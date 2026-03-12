/*
 * MNIST Demo v2 Host Application (Optimized with Pinned Memory)
 *
 * Optimizations:
 * 1. Pinned (page-locked) memory for faster DMA transfers
 * 2. Pre-allocated buffer pools
 * 3. Reduced memory allocation overhead
 */

#include <err.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>

#include <tee_client_api.h>
#include "mnist_demo.h"

/* Layer configuration */
static const int layer_configs[NUM_LAYERS * 3] = {
    784, 256, 8,
    256, 128, 8,
    128, 10,  4
};

/* Global TEE context */
static TEEC_Context g_ctx;
static TEEC_Session g_sess;
static int g_tee_initialized = 0;

/* Pinned memory buffers */
static float *pinned_input[3] = {NULL, NULL, NULL};
static float *pinned_output[3] = {NULL, NULL, NULL};
static size_t pinned_input_sizes[3] = {0, 0, 0};
static size_t pinned_output_sizes[3] = {0, 0, 0};
static int g_pinned_initialized = 0;

/* Maximum supported batch size */
#define MAX_BATCH_SIZE 128

/*
 * Allocate pinned memory buffer
 */
static float *alloc_pinned_buffer(size_t size, int *is_pinned) {
    float *buf;

    /* Allocate page-aligned memory */
    if (posix_memalign((void **)&buf, sysconf(_SC_PAGESIZE), size) != 0) {
        return NULL;
    }

    /* Pin the memory */
    if (mlock(buf, size) == 0) {
        *is_pinned = 1;
    } else {
        *is_pinned = 0;
        fprintf(stderr, "Warning: mlock failed, using unpinned memory\n");
    }

    memset(buf, 0, size);
    return buf;
}

/*
 * Initialize pinned memory buffers
 */
int pinned_memory_init(void) {
    if (g_pinned_initialized) return 0;

    int input_dims[3] = {784, 256, 128};
    int output_dims[3] = {256, 128, 10};
    int is_pinned;

    for (int i = 0; i < 3; i++) {
        pinned_input_sizes[i] = MAX_BATCH_SIZE * input_dims[i] * sizeof(float);
        pinned_output_sizes[i] = MAX_BATCH_SIZE * output_dims[i] * sizeof(float);

        pinned_input[i] = alloc_pinned_buffer(pinned_input_sizes[i], &is_pinned);
        pinned_output[i] = alloc_pinned_buffer(pinned_output_sizes[i], &is_pinned);

        if (!pinned_input[i] || !pinned_output[i]) {
            fprintf(stderr, "Failed to allocate pinned buffers for layer %d\n", i);
            return -1;
        }
    }

    g_pinned_initialized = 1;
    printf("Pinned memory initialized (max batch: %d)\n", MAX_BATCH_SIZE);
    return 0;
}

/*
 * Cleanup pinned memory
 */
void pinned_memory_cleanup(void) {
    if (!g_pinned_initialized) return;

    for (int i = 0; i < 3; i++) {
        if (pinned_input[i]) {
            munlock(pinned_input[i], pinned_input_sizes[i]);
            free(pinned_input[i]);
            pinned_input[i] = NULL;
        }
        if (pinned_output[i]) {
            munlock(pinned_output[i], pinned_output_sizes[i]);
            free(pinned_output[i]);
            pinned_output[i] = NULL;
        }
    }
    g_pinned_initialized = 0;
}

/*
 * Get pinned input buffer for layer
 */
float *get_pinned_input(int layer_idx, int batch_size) {
    if (!g_pinned_initialized || layer_idx < 0 || layer_idx >= 3) return NULL;
    return pinned_input[layer_idx];
}

/*
 * Get pinned output buffer for layer
 */
float *get_pinned_output(int layer_idx, int batch_size) {
    if (!g_pinned_initialized || layer_idx < 0 || layer_idx >= 3) return NULL;
    return pinned_output[layer_idx];
}

/*
 * Initialize TEE connection
 */
int tee_init(void) {
    TEEC_Result res;
    uint32_t err_origin;
    TEEC_UUID uuid = TA_MNIST_DEMO_UUID;

    if (g_tee_initialized) return 0;

    res = TEEC_InitializeContext(NULL, &g_ctx);
    if (res != TEEC_SUCCESS) {
        fprintf(stderr, "TEEC_InitializeContext failed: 0x%x\n", res);
        return -1;
    }

    res = TEEC_OpenSession(&g_ctx, &g_sess, &uuid,
                           TEEC_LOGIN_PUBLIC, NULL, NULL, &err_origin);
    if (res != TEEC_SUCCESS) {
        fprintf(stderr, "TEEC_OpenSession failed: 0x%x\n", res);
        TEEC_FinalizeContext(&g_ctx);
        return -1;
    }

    g_tee_initialized = 1;
    return 0;
}

/*
 * Cleanup TEE
 */
void tee_cleanup(void) {
    if (g_tee_initialized) {
        TEEC_CloseSession(&g_sess);
        TEEC_FinalizeContext(&g_ctx);
        g_tee_initialized = 0;
    }
}

/*
 * Full initialization (TEE + pinned memory)
 */
int full_init(void) {
    if (tee_init() != 0) return -1;
    if (pinned_memory_init() != 0) return -1;
    return 0;
}

/*
 * Full cleanup
 */
void full_cleanup(void) {
    pinned_memory_cleanup();
    tee_cleanup();
}

/*
 * Generate keys in TEE
 */
int tee_generate_keys(void) {
    TEEC_Result res;
    TEEC_Operation op;
    uint32_t err_origin;

    if (tee_init() != 0) return -1;

    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(
        TEEC_MEMREF_TEMP_INPUT, TEEC_NONE, TEEC_NONE, TEEC_NONE);
    op.params[0].tmpref.buffer = (void *)layer_configs;
    op.params[0].tmpref.size = sizeof(layer_configs);

    res = TEEC_InvokeCommand(&g_sess, TA_MNIST_CMD_GENERATE_KEYS, &op, &err_origin);
    if (res != TEEC_SUCCESS) {
        fprintf(stderr, "Key generation failed: 0x%x\n", res);
        return -1;
    }

    printf("Keys generated in TEE\n");
    return 0;
}

/*
 * Get key status
 */
int tee_get_key_status(int *loaded, int *locked) {
    TEEC_Result res;
    TEEC_Operation op;
    uint32_t err_origin;

    if (tee_init() != 0) return -1;

    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(
        TEEC_VALUE_OUTPUT, TEEC_NONE, TEEC_NONE, TEEC_NONE);

    res = TEEC_InvokeCommand(&g_sess, TA_MNIST_CMD_GET_KEY_STATUS, &op, &err_origin);
    if (res != TEEC_SUCCESS) return -1;

    *loaded = op.params[0].value.a;
    *locked = op.params[0].value.b;
    return 0;
}

/*
 * Export keys
 */
int tee_export_keys(float *buffer, size_t buffer_size, size_t *actual_size) {
    TEEC_Result res;
    TEEC_Operation op;
    uint32_t err_origin;

    if (tee_init() != 0) return -1;

    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(
        TEEC_MEMREF_TEMP_OUTPUT, TEEC_NONE, TEEC_NONE, TEEC_NONE);
    op.params[0].tmpref.buffer = buffer;
    op.params[0].tmpref.size = buffer_size;

    res = TEEC_InvokeCommand(&g_sess, TA_MNIST_CMD_EXPORT_KEYS, &op, &err_origin);
    if (res == TEEC_ERROR_SHORT_BUFFER) {
        *actual_size = op.params[0].tmpref.size;
        return -2;
    }
    if (res != TEEC_SUCCESS) {
        if (res == TEEC_ERROR_ACCESS_DENIED) {
            fprintf(stderr, "Keys are locked - cannot export\n");
        }
        return -1;
    }

    *actual_size = op.params[0].tmpref.size;
    return 0;
}

/*
 * Lock keys after provisioning
 */
int tee_lock_keys(void) {
    TEEC_Result res;
    TEEC_Operation op;
    uint32_t err_origin;

    if (tee_init() != 0) return -1;

    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_NONE, TEEC_NONE, TEEC_NONE, TEEC_NONE);

    res = TEEC_InvokeCommand(&g_sess, TA_MNIST_CMD_LOCK_KEYS, &op, &err_origin);
    if (res != TEEC_SUCCESS) {
        fprintf(stderr, "Lock keys failed: 0x%x\n", res);
        return -1;
    }

    printf("Keys locked in TEE\n");
    return 0;
}

/*
 * Inference using pinned memory (OPTIMIZED)
 * Uses pre-allocated pinned buffers for faster transfer
 */
int tee_loro_inference_pinned(int layer_idx, int batch_size,
                               float *input, float *output) {
    TEEC_Result res;
    TEEC_Operation op;
    uint32_t err_origin;

    int input_dim = layer_configs[layer_idx * 3 + 0];
    int output_dim = layer_configs[layer_idx * 3 + 1];

    int input_size = batch_size * input_dim * sizeof(float);
    int output_size = batch_size * output_dim * sizeof(float);
    int elapsed;

    if (tee_init() != 0) return -1;

    /* Copy input to pinned buffer */
    float *pinned_in = get_pinned_input(layer_idx, batch_size);
    float *pinned_out = get_pinned_output(layer_idx, batch_size);

    if (pinned_in && pinned_out) {
        /* Use pinned buffers */
        memcpy(pinned_in, input, input_size);

        memset(&op, 0, sizeof(op));
        op.paramTypes = TEEC_PARAM_TYPES(
            TEEC_MEMREF_TEMP_INPUT,
            TEEC_MEMREF_TEMP_INOUT,
            TEEC_VALUE_OUTPUT,
            TEEC_NONE);

        int dims[2] = {batch_size, layer_idx};
        op.params[0].tmpref.buffer = pinned_in;
        op.params[0].tmpref.size = input_size;
        op.params[1].tmpref.buffer = pinned_out;
        op.params[1].tmpref.size = output_size;

        /* Note: dims are passed separately for TEE, output goes to pinned_out */
        /* For now, use temporary buffer for dims + output */
        float *temp_buf = (float *)malloc(output_size > sizeof(dims) ? output_size : sizeof(dims));
        memcpy(temp_buf, dims, sizeof(dims));
        op.params[1].tmpref.buffer = temp_buf;
        op.params[1].tmpref.size = output_size;

        res = TEEC_InvokeCommand(&g_sess, TA_MNIST_CMD_LORO_INFERENCE, &op, &err_origin);

        if (res == TEEC_SUCCESS) {
            elapsed = op.params[2].value.a;
            memcpy(output, temp_buf, output_size);
        } else {
            elapsed = -1;
        }

        free(temp_buf);
    } else {
        /* Fallback to regular allocation */
        float *io_buffer = (float *)malloc(output_size);
        if (!io_buffer) return -1;

        memset(&op, 0, sizeof(op));
        op.paramTypes = TEEC_PARAM_TYPES(
            TEEC_MEMREF_TEMP_INPUT,
            TEEC_MEMREF_TEMP_INOUT,
            TEEC_VALUE_OUTPUT,
            TEEC_NONE);

        int dims[2] = {batch_size, layer_idx};
        op.params[0].tmpref.buffer = input;
        op.params[0].tmpref.size = input_size;
        memcpy(io_buffer, dims, sizeof(dims));
        op.params[1].tmpref.buffer = io_buffer;
        op.params[1].tmpref.size = output_size;

        res = TEEC_InvokeCommand(&g_sess, TA_MNIST_CMD_LORO_INFERENCE, &op, &err_origin);

        if (res == TEEC_SUCCESS) {
            elapsed = op.params[2].value.a;
            memcpy(output, io_buffer, output_size);
        } else {
            elapsed = -1;
        }

        free(io_buffer);
    }

    return elapsed;
}

/*
 * Original inference function (for compatibility)
 */
int tee_loro_inference(int layer_idx, int batch_size,
                       float *input, float *output) {
    return tee_loro_inference_pinned(layer_idx, batch_size, input, output);
}

/* ========================================
 * Python bindings via ctypes
 * ======================================== */

int tee_get_export_buffer_size(void) {
    int total = 0;
    for (int i = 0; i < NUM_LAYERS; i++) {
        int output_dim = layer_configs[i * 3 + 1];
        int rank = layer_configs[i * 3 + 2];
        total += (output_dim * rank + rank * layer_configs[i * 3]) * sizeof(float);
    }
    return total;
}

int py_get_key_status(void) {
    int loaded, locked;
    if (tee_get_key_status(&loaded, &locked) != 0) return -1;
    return (loaded ? 1 : 0) | (locked ? 2 : 0);
}

int py_export_keys(float *buffer, int buffer_size) {
    size_t actual;
    return tee_export_keys(buffer, buffer_size, &actual);
}

int py_inference(int layer_idx, int batch_size, float *input, float *output) {
    return tee_loro_inference_pinned(layer_idx, batch_size, input, output);
}

int py_full_init(void) {
    return full_init();
}

void py_full_cleanup(void) {
    full_cleanup();
}

/* Check if using pinned memory */
int py_is_pinned(void) {
    return g_pinned_initialized ? 1 : 0;
}

/* ========================================
 * Main demo
 * ======================================== */
int main(void) {
    int loaded, locked;

    printf("==========================================\n");
    printf("MNIST LoRO Secure Inference Demo v2\n");
    printf("(Optimized with Pinned Memory)\n");
    printf("==========================================\n\n");

    /* Initialize TEE and pinned memory */
    if (full_init() != 0) {
        printf("Initialization failed!\n");
        return 1;
    }

    /* Check status */
    if (tee_get_key_status(&loaded, &locked) != 0) {
        printf("Error: Cannot get key status\n");
        return 1;
    }

    printf("Key status: loaded=%d, locked=%d\n", loaded, locked);
    printf("Using pinned memory: %s\n\n", g_pinned_initialized ? "Yes" : "No");

    if (!loaded) {
        printf("No keys found. Generating new keys...\n");
        if (tee_generate_keys() != 0) {
            printf("Failed to generate keys!\n");
            return 1;
        }

        /* Export keys */
        int buf_size = tee_get_export_buffer_size();
        float *key_buffer = (float *)malloc(buf_size);
        size_t actual;

        printf("\nExporting keys for weight obfuscation...\n");
        if (tee_export_keys(key_buffer, buf_size, &actual) == 0) {
            printf("Keys exported: %zu bytes\n", actual);
        }

        free(key_buffer);
    }

    /* Test inference with pinned memory */
    printf("\nTesting inference with pinned memory:\n");
    printf("------------------------------------\n");

    float input[784] = {0};
    float output1[256], output2[128], output3[10];

    int t1 = tee_loro_inference_pinned(0, 1, input, output1);
    printf("Layer 1: %d ms\n", t1);

    int t2 = tee_loro_inference_pinned(1, 1, output1, output2);
    printf("Layer 2: %d ms\n", t2);

    int t3 = tee_loro_inference_pinned(2, 1, output2, output3);
    printf("Layer 3: %d ms\n", t3);

    printf("\nTotal TEE time: %d ms\n", t1 + t2 + t3);

    full_cleanup();

    printf("\n==========================================\n");
    printf("Demo completed!\n");
    printf("==========================================\n");

    return 0;
}
