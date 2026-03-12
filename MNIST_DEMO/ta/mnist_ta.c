/*
 * MNIST Demo Trusted Application
 *
 * Simplified version without NEON for debugging
 */

#include <tee_internal_api.h>
#include <tee_internal_api_extensions.h>
#include <string.h>
#include "include/mnist_demo.h"

/* Global storage for LoRO keys */
static struct {
    float *B[NUM_LAYERS];
    float *A[NUM_LAYERS];
    int input_dim[NUM_LAYERS];
    int output_dim[NUM_LAYERS];
    int rank[NUM_LAYERS];
    int keys_loaded;
    int keys_locked;
} loro_keys = {0};

/* Simple GEMM without NEON: C = A @ B */
static void simple_gemm(const float *A, const float *B, float *C, int M, int K, int N)
{
    int i, j, k;

    for (i = 0; i < M * N; i++) {
        C[i] = 0.0f;
    }

    for (i = 0; i < M; i++) {
        for (k = 0; k < K; k++) {
            float a = A[i * K + k];
            for (j = 0; j < N; j++) {
                C[i * N + j] += a * B[k * N + j];
            }
        }
    }
}

/* GEMM with transposed B: C = A @ B.T where B is (N x K) stored row-major */
static void gemm_Bt(const float *A, const float *B, float *C, int M, int K, int N)
{
    int i, j, k;

    for (i = 0; i < M * N; i++) {
        C[i] = 0.0f;
    }

    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            float sum = 0.0f;
            for (k = 0; k < K; k++) {
                /* B is (N x K), B.T is (K x N), B.T[k,j] = B[j,k] */
                sum += A[i * K + k] * B[j * K + k];
            }
            C[i * N + j] = sum;
        }
    }
}

static void free_keys(void)
{
    int i;
    for (i = 0; i < NUM_LAYERS; i++) {
        if (loro_keys.B[i]) {
            TEE_Free(loro_keys.B[i]);
            loro_keys.B[i] = NULL;
        }
        if (loro_keys.A[i]) {
            TEE_Free(loro_keys.A[i]);
            loro_keys.A[i] = NULL;
        }
    }
    loro_keys.keys_loaded = 0;
}

/*
 * Generate random keys
 */
static TEE_Result generate_keys(uint32_t param_types, TEE_Param params[4])
{
    uint32_t exp_param_types = TEE_PARAM_TYPES(
        TEE_PARAM_TYPE_MEMREF_INPUT,
        TEE_PARAM_TYPE_NONE,
        TEE_PARAM_TYPE_NONE,
        TEE_PARAM_TYPE_NONE
    );

    int *layer_configs;
    int i;
    size_t B_size, A_size;
    uint32_t j;
    float *B_ptr, *A_ptr;

    DMSG("generate_keys called");

    if (param_types != exp_param_types) {
        EMSG("Bad param types");
        return TEE_ERROR_BAD_PARAMETERS;
    }

    layer_configs = (int *)params[0].memref.buffer;

    /* Validate input size */
    if (params[0].memref.size != NUM_LAYERS * 3 * sizeof(int)) {
        EMSG("Wrong input size: %u", params[0].memref.size);
        return TEE_ERROR_BAD_PARAMETERS;
    }

    free_keys();

    for (i = 0; i < NUM_LAYERS; i++) {
        loro_keys.input_dim[i] = layer_configs[i * 3 + 0];
        loro_keys.output_dim[i] = layer_configs[i * 3 + 1];
        loro_keys.rank[i] = layer_configs[i * 3 + 2];

        /* B: (output_dim, rank), A: (rank, input_dim) */
        B_size = loro_keys.output_dim[i] * loro_keys.rank[i] * sizeof(float);
        A_size = loro_keys.rank[i] * loro_keys.input_dim[i] * sizeof(float);

        DMSG("Layer %d: allocating B=%zu, A=%zu bytes", i, B_size, A_size);

        loro_keys.B[i] = TEE_Malloc(B_size, TEE_MALLOC_FILL_ZERO);
        loro_keys.A[i] = TEE_Malloc(A_size, TEE_MALLOC_FILL_ZERO);

        if (!loro_keys.B[i] || !loro_keys.A[i]) {
            EMSG("Out of memory at layer %d", i);
            free_keys();
            return TEE_ERROR_OUT_OF_MEMORY;
        }

        /* Generate random values */
        B_ptr = loro_keys.B[i];
        A_ptr = loro_keys.A[i];

        for (j = 0; j < B_size / sizeof(float); j++) {
            B_ptr[j] = 0.01f * (float)(j % 100);  /* Simple deterministic values */
        }
        for (j = 0; j < A_size / sizeof(float); j++) {
            A_ptr[j] = 0.01f * (float)(j % 100);
        }

        DMSG("Layer %d done", i);
    }

    loro_keys.keys_loaded = 1;
    IMSG("Keys generated successfully");

    return TEE_SUCCESS;
}

/*
 * Export keys
 */
static TEE_Result export_keys(uint32_t param_types, TEE_Param params[4])
{
    uint32_t exp_param_types = TEE_PARAM_TYPES(
        TEE_PARAM_TYPE_MEMREF_OUTPUT,
        TEE_PARAM_TYPE_NONE,
        TEE_PARAM_TYPE_NONE,
        TEE_PARAM_TYPE_NONE
    );

    float *buffer;
    size_t total_size = 0;
    size_t offset = 0;
    int i;

    DMSG("export_keys called");

    if (param_types != exp_param_types) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    if (!loro_keys.keys_loaded) {
        EMSG("Keys not loaded");
        return TEE_ERROR_BAD_STATE;
    }

    if (loro_keys.keys_locked) {
        EMSG("Keys are locked - export denied");
        return TEE_ERROR_ACCESS_DENIED;
    }

    /* Calculate total size */
    for (i = 0; i < NUM_LAYERS; i++) {
        total_size += loro_keys.output_dim[i] * loro_keys.rank[i] * sizeof(float);
        total_size += loro_keys.rank[i] * loro_keys.input_dim[i] * sizeof(float);
    }

    if (params[0].memref.size < total_size) {
        params[0].memref.size = total_size;
        return TEE_ERROR_SHORT_BUFFER;
    }

    buffer = (float *)params[0].memref.buffer;
    offset = 0;

    /* Copy all matrices */
    for (i = 0; i < NUM_LAYERS; i++) {
        size_t B_size = loro_keys.output_dim[i] * loro_keys.rank[i] * sizeof(float);
        size_t A_size = loro_keys.rank[i] * loro_keys.input_dim[i] * sizeof(float);

        TEE_MemMove(buffer + offset, loro_keys.B[i], B_size);
        offset += B_size / sizeof(float);

        TEE_MemMove(buffer + offset, loro_keys.A[i], A_size);
        offset += A_size / sizeof(float);
    }

    params[0].memref.size = total_size;
    IMSG("Exported %zu bytes", total_size);

    return TEE_SUCCESS;
}

/*
 * Get key status
 */
static TEE_Result get_key_status(uint32_t param_types, TEE_Param params[4])
{
    uint32_t exp_param_types = TEE_PARAM_TYPES(
        TEE_PARAM_TYPE_VALUE_OUTPUT,
        TEE_PARAM_TYPE_NONE,
        TEE_PARAM_TYPE_NONE,
        TEE_PARAM_TYPE_NONE
    );

    if (param_types != exp_param_types) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    params[0].value.a = loro_keys.keys_loaded;
    params[0].value.b = loro_keys.keys_locked;

    return TEE_SUCCESS;
}

/*
 * Lock keys after provisioning
 */
static TEE_Result lock_keys(uint32_t param_types, TEE_Param params[4])
{
    uint32_t exp_param_types = TEE_PARAM_TYPES(
        TEE_PARAM_TYPE_NONE,
        TEE_PARAM_TYPE_NONE,
        TEE_PARAM_TYPE_NONE,
        TEE_PARAM_TYPE_NONE
    );

    (void)params;

    if (param_types != exp_param_types) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    if (!loro_keys.keys_loaded) {
        EMSG("No keys to lock");
        return TEE_ERROR_BAD_STATE;
    }

    loro_keys.keys_locked = 1;
    IMSG("Keys locked - export disabled");

    return TEE_SUCCESS;
}

/*
 * Perform LoRO inference
 */
static TEE_Result loro_inference(uint32_t param_types, TEE_Param params[4])
{
    uint32_t exp_param_types = TEE_PARAM_TYPES(
        TEE_PARAM_TYPE_MEMREF_INPUT,
        TEE_PARAM_TYPE_MEMREF_INOUT,
        TEE_PARAM_TYPE_VALUE_OUTPUT,
        TEE_PARAM_TYPE_NONE
    );

    TEE_Time start, end;
    int elapsed;
    float *input_matrix;
    float *intermediate;
    float *output_matrix;
    int *dims;
    int batch_size, layer_idx;
    int input_h, rank, output_h;
    size_t input_size, intermediate_size, output_size;

    DMSG("loro_inference called");

    if (param_types != exp_param_types) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    if (!loro_keys.keys_loaded) {
        EMSG("Keys not loaded");
        return TEE_ERROR_BAD_STATE;
    }

    dims = (int *)params[1].memref.buffer;
    batch_size = dims[0];
    layer_idx = dims[1];

    if (layer_idx < 0 || layer_idx >= NUM_LAYERS) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    input_h = loro_keys.input_dim[layer_idx];
    rank = loro_keys.rank[layer_idx];
    output_h = loro_keys.output_dim[layer_idx];

    input_size = batch_size * input_h * sizeof(float);
    intermediate_size = batch_size * rank * sizeof(float);
    output_size = batch_size * output_h * sizeof(float);

    input_matrix = TEE_Malloc(input_size, TEE_MALLOC_FILL_ZERO);
    intermediate = TEE_Malloc(intermediate_size, TEE_MALLOC_FILL_ZERO);
    output_matrix = TEE_Malloc(output_size, TEE_MALLOC_FILL_ZERO);

    if (!input_matrix || !intermediate || !output_matrix) {
        TEE_Free(input_matrix);
        TEE_Free(intermediate);
        TEE_Free(output_matrix);
        return TEE_ERROR_OUT_OF_MEMORY;
    }

    TEE_MemMove(input_matrix, params[0].memref.buffer, input_size);

    TEE_GetREETime(&start);

    /* Compute correction = x @ A.T @ B.T */
    /* A is (rank, input_dim), B is (output_dim, rank) */

    /* Step 1: IR = x @ A.T where A.T is (input_dim, rank) */
    /* x is (batch, input_dim), IR is (batch, rank) */
    gemm_Bt(input_matrix, loro_keys.A[layer_idx], intermediate,
            batch_size, input_h, rank);

    /* Step 2: correction = IR @ B.T where B.T is (rank, output_dim) */
    /* IR is (batch, rank), correction is (batch, output_dim) */
    gemm_Bt(intermediate, loro_keys.B[layer_idx], output_matrix,
            batch_size, rank, output_h);

    TEE_GetREETime(&end);

    elapsed = (end.seconds - start.seconds) * 1000 + (end.millis - start.millis);

    TEE_MemMove(params[1].memref.buffer, output_matrix, output_size);
    params[2].value.a = elapsed;

    TEE_Free(input_matrix);
    TEE_Free(intermediate);
    TEE_Free(output_matrix);

    return TEE_SUCCESS;
}

/*
 * TA Entry Points
 */
TEE_Result TA_CreateEntryPoint(void)
{
    DMSG("TA_CreateEntryPoint");
    memset(&loro_keys, 0, sizeof(loro_keys));
    return TEE_SUCCESS;
}

void TA_DestroyEntryPoint(void)
{
    DMSG("TA_DestroyEntryPoint");
    free_keys();
}

TEE_Result TA_OpenSessionEntryPoint(uint32_t param_types,
                                    TEE_Param params[4],
                                    void **sess_ctx)
{
    (void)params;
    (void)sess_ctx;

    if (param_types != TEE_PARAM_TYPES(TEE_PARAM_TYPE_NONE,
                                        TEE_PARAM_TYPE_NONE,
                                        TEE_PARAM_TYPE_NONE,
                                        TEE_PARAM_TYPE_NONE)) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    IMSG("MNIST TA session opened");

    return TEE_SUCCESS;
}

void TA_CloseSessionEntryPoint(void *sess_ctx)
{
    (void)sess_ctx;
    IMSG("MNIST TA session closed");
}

TEE_Result TA_InvokeCommandEntryPoint(void *sess_ctx,
                                      uint32_t cmd_id,
                                      uint32_t param_types,
                                      TEE_Param params[4])
{
    (void)sess_ctx;

    DMSG("Command: 0x%x", cmd_id);

    switch (cmd_id) {
    case TA_MNIST_CMD_LORO_INFERENCE:
        return loro_inference(param_types, params);
    case TA_MNIST_CMD_GENERATE_KEYS:
        return generate_keys(param_types, params);
    case TA_MNIST_CMD_GET_KEY_STATUS:
        return get_key_status(param_types, params);
    case TA_MNIST_CMD_EXPORT_KEYS:
        return export_keys(param_types, params);
    case TA_MNIST_CMD_LOCK_KEYS:
        return lock_keys(param_types, params);
    default:
        EMSG("Unknown cmd: 0x%x", cmd_id);
        return TEE_ERROR_BAD_PARAMETERS;
    }
}
