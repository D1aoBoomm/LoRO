#include <tee_internal_api.h>
#include <tee_internal_api_extensions.h>

#include <arm_neon.h> // make sure your machine support neon
#include <demo.h>

//int s = 512;
//int h = 768;
//int r = 128;

/*
 * Called when the instance of the TA is created. This is the first call in
 * the TA.
 */
TEE_Result TA_CreateEntryPoint(void)
{
	DMSG("has been called");

	return TEE_SUCCESS;
}

void neon_gemm(float *A, float *B, float *C, int M, int K, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j += 4) {
            float32x4_t c = vmovq_n_f32(0.0);

            for (int k = 0; k < N; k++) {
                float32x4_t b = vld1q_f32(&B[k * K + j]);
                float a = A[i * N + k];
                c = vmlaq_n_f32(c, b, a);
            }
            vst1q_f32(&C[i * K + j], c);
        }
    }
}

/*
 * Called when the instance of the TA is destroyed if the TA has not
 * crashed or panicked. This is the last call in the TA.
 */
void TA_DestroyEntryPoint(void)
{
	DMSG("has been called");
}

/*
 * Called when a new session is opened to the TA. *sess_ctx can be updated
 * with a value to be able to identify this session in subsequent calls to the
 * TA. In this function you will normally do the global initialization for the
 * TA.
 */
TEE_Result TA_OpenSessionEntryPoint(uint32_t param_types,
		TEE_Param __maybe_unused params[4],
		void __maybe_unused **sess_ctx)
{
	uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_NONE,
						   TEE_PARAM_TYPE_NONE,
						   TEE_PARAM_TYPE_NONE,
						   TEE_PARAM_TYPE_NONE);

	DMSG("has been called");

	if (param_types != exp_param_types)
		return TEE_ERROR_BAD_PARAMETERS;

	/* Unused parameters */
	(void)&params;
	(void)&sess_ctx;

	/*
	 * The DMSG() macro is non-standard, TEE Internal API doesn't
	 * specify any means to logging from a TA.
	 */
	IMSG("Hello!\n");

	/* If return value != TEE_SUCCESS the session will not be created. */
	return TEE_SUCCESS;
}

/*
 * Called when a session is closed, sess_ctx hold the value that was
 * assigned by TA_OpenSessionEntryPoint().
 */
void TA_CloseSessionEntryPoint(void __maybe_unused *sess_ctx)
{
	(void)&sess_ctx; /* Unused parameter */
	IMSG("Goodbye!\n");
}

static TEE_Result inference(uint32_t param_types,
	TEE_Param params[4])
{
	uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INOUT,
					TEE_PARAM_TYPE_MEMREF_INOUT,
					TEE_PARAM_TYPE_VALUE_INOUT,
					TEE_PARAM_TYPE_NONE);
	size_t input_sz;
	float *input_matrix;
	size_t A_r_sz;
	float *A_r;
	size_t IR_sz;
	float *IR;
	size_t B_r_sz;
	float *B_r;
	size_t output_sz;
	float *output_matrix;
	size_t W_sz;
	float *W;

	int *dimension_matrix;
	int elapsed;

	dimension_matrix = TEE_Malloc(5*sizeof(int), 0);
	TEE_MemMove(dimension_matrix, params[1].memref.buffer, 5*sizeof(int));

	int s = dimension_matrix[0];
	int h = dimension_matrix[1];
	int r = dimension_matrix[2];
	int output_h = dimension_matrix[3];
	int inference_type = dimension_matrix[4];

	//int s = 512;
	//int h = 768;
	//int r = 16;
	//int output_h = 768;

	DMSG("has been called");

	if (param_types != exp_param_types)
		return TEE_ERROR_BAD_PARAMETERS;

	input_sz = s * h * sizeof(float);
	input_matrix = TEE_Malloc(input_sz, 0);
	if(!input_matrix){
		return TEE_ERROR_OUT_OF_MEMORY;
	}

	TEE_Time start, end;
	
	//TEE_GetSystemTime(&start);

	if(inference_type==0){
		// parameters are supposed to be loaded already
		A_r_sz = h * r * sizeof(float);
		A_r = TEE_Malloc(A_r_sz, 0);
		if(!A_r){
			return TEE_ERROR_OUT_OF_MEMORY;
		}

		IR_sz = s * r * sizeof(float);
		IR = TEE_Malloc(IR_sz, 0);
		if(!IR){
			return TEE_ERROR_OUT_OF_MEMORY;
		}

		TEE_GetREETime(&start);
		neon_gemm(input_matrix, A_r, IR, s, h, r);
		TEE_GetREETime(&end);

		B_r_sz = r * output_h * sizeof(float);
		B_r = TEE_Malloc(B_r_sz, 0);
		if(!B_r){
			return TEE_ERROR_OUT_OF_MEMORY;
		}

		elapsed = (end.seconds - start.seconds) * 1000 + (end.millis - start.millis);

		// if your secure memory is not enough
		//TEE_Free(A_r);
		//TEE_Free(input_matrix);

		output_sz = s * output_h * sizeof(float);
		output_matrix = TEE_Malloc(output_sz, 0);
		if(!output_matrix){
			return TEE_ERROR_OUT_OF_MEMORY;
		}

		TEE_GetREETime(&start);
		neon_gemm(IR, B_r, output_matrix, s, r, h);
		// computing complexity eqauls to OTP
		for (int i=0; i<output_sz; i++){
			*(output_matrix+i) += *(output_matrix+i);
		}

		TEE_GetREETime(&end);

		elapsed += (end.seconds - start.seconds) * 1000 + (end.millis - start.millis);

		
		TEE_MemMove(params[0].memref.buffer, output_matrix, output_sz);
		
	}
	else if(inference_type==1){
		W_sz = h * output_h * sizeof(float); // you may need to recompile the OPTEE to support such large matrix on LLMs. at least we do.
		W = TEE_Malloc(W_sz, 0);
		if(!W){
			return TEE_ERROR_OUT_OF_MEMORY;
		}
		output_sz = s * output_h * sizeof(float);
		output_matrix = TEE_Malloc(output_sz, 0);
		if(!output_matrix){
			return TEE_ERROR_OUT_OF_MEMORY;
		}

		TEE_GetREETime(&start);
		neon_gemm(input_matrix, W, output_matrix, s, h, output_h);
		TEE_MemMove(params[0].memref.buffer, output_matrix, output_sz);
		TEE_GetREETime(&end);

		elapsed = (end.seconds - start.seconds) * 1000 + (end.millis - start.millis);
	}
	
	//TEE_GetSystemTime(&end);
	
	// return time
	params[2].value.a = elapsed;

	//to test whether the result is correct
	//for (int i=0; i<4; i++){
        //*(input_matrix+i) = i;
        //}

	return TEE_SUCCESS;
}
/*
 * Called when a TA is invoked. sess_ctx hold that value that was
 * assigned by TA_OpenSessionEntryPoint(). The rest of the paramters
 * comes from normal world.
 */
TEE_Result TA_InvokeCommandEntryPoint(void __maybe_unused *sess_ctx,
			uint32_t cmd_id,
			uint32_t param_types, TEE_Param params[4])
{
	(void)&sess_ctx; /* Unused parameter */

	switch (cmd_id) {
	case TA_DEMO_INFERENCE:
		return inference(param_types, params);
	default:
		return TEE_ERROR_BAD_PARAMETERS;
	}
}
