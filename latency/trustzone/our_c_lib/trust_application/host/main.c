#include <err.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/* OP-TEE TEE client API (built by optee_client) */
#include <tee_client_api.h>

/* For the UUID (found in the TA's h-file(s)) */
#include <demo.h>

void initialize_matrix(float *matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = (float)(rand() % 100) / 10.0f;
    }
}

int one_inference_TEE(int s, int h, int r, int output_h, int inference_type){
	TEEC_Result res;
	TEEC_Context ctx;
	TEEC_Session sess;
	TEEC_Operation op;
	TEEC_UUID uuid = TA_DEMO_UUID;
	uint32_t err_origin;
	//printf("%d\n", 1);
	//printf("%d, %d, %d\n", s, h ,r);
	
	//int64_t time;
	int matrix_size;

	int *dimension_matrix = (int *) malloc(5 * sizeof(int));
	dimension_matrix[0] = s;
	dimension_matrix[1] = h;
	dimension_matrix[2] = r;
	dimension_matrix[3] = output_h;
	dimension_matrix[4] = inference_type;

	if (inference_type==0){
		matrix_size = s*h;
	}
	else if (inference_type==1){
		matrix_size = s*output_h;
	}
		 

	float *R = (float *) malloc(matrix_size * sizeof(float));

	/* Initialize a context connecting us to the TEE */
	res = TEEC_InitializeContext(NULL, &ctx);
	if (res != TEEC_SUCCESS)
		errx(1, "TEEC_InitializeContext failed with code 0x%x", res);

	/*
	 * Open a session to the "hello world" TA, the TA will print "hello
	 * world!" in the log when the session is created.
	 */
	res = TEEC_OpenSession(&ctx, &sess, &uuid,
			       TEEC_LOGIN_PUBLIC, NULL, NULL, &err_origin);
	if (res != TEEC_SUCCESS)
		errx(1, "TEEC_Opensession failed with code 0x%x origin 0x%x",
			res, err_origin);

	/*
	 * Execute a function in the TA by invoking it, in this case
	 * we're incrementing a number.
	 *
	 * The value of command ID part and how the parameters are
	 * interpreted is part of the interface provided by the TA.
	 */

	/* Clear the TEEC_Operation struct */
	memset(&op, 0, sizeof(op));

	// resigter shared memory

	/*
	 * Prepare the argument. Pass a value in the first parameter,
	 * the remaining three parameters are unused.
	 */
	op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INOUT, TEEC_MEMREF_TEMP_INOUT, TEEC_VALUE_INOUT, TEEC_NONE);
	//op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INOUT, TEEC_NONE, TEEC_NONE, TEEC_NONE);
	op.params[0].tmpref.buffer = R;
	op.params[0].tmpref.size = matrix_size * sizeof(float);
	op.params[1].tmpref.buffer = dimension_matrix;
	op.params[1].tmpref.size = 5 * sizeof(int);
	op.params[2].value.a = 0;

	/*
	 * TA_HELLO_WORLD_CMD_INC_VALUE is the actual function in the TA to be
	 * called.
	 */
	// printf("%s\n", "Passing input to TA");
	res = TEEC_InvokeCommand(&sess, TA_DEMO_INFERENCE, &op,
				 &err_origin);
	if (res != TEEC_SUCCESS)
		errx(1, "TEEC_InvokeCommand failed with code 0x%x origin 0x%x",
			res, err_origin);
	// printf("%s\n", "Completed");

	// printf("%d\n", op.params[2].value.a);


	/*
	 * We're done with the TA, close the session and
	 * destroy the context.
	 *
	 * The TA will print "Goodbye!" in the log when the
	 * session is closed.
	 */

	TEEC_CloseSession(&sess);

	TEEC_FinalizeContext(&ctx);

	return op.params[2].value.a;
}

float test_add(float a, float b){
	return a+b;
}

int main(void)
{
	TEEC_Result res;
	TEEC_Context ctx;
	TEEC_Session sess;
	TEEC_Operation op;
	TEEC_UUID uuid = TA_DEMO_UUID;
	uint32_t err_origin;
	//printf("%d\n", 1);
	//printf("%d, %d, %d\n", s, h ,r);
	
	//int64_t time;
	int matrix_size;

	int s = 512;
	int h = 768;
	int r = 16;
	int output_h = 768;
	int inference_type = 0;

	int *dimension_matrix = (int *) malloc(5 * sizeof(int));
	dimension_matrix[0] = s;
	dimension_matrix[1] = h;
	dimension_matrix[2] = r;
	dimension_matrix[3] = output_h;
	dimension_matrix[4] = inference_type;

	if (inference_type==0){
		matrix_size = s*h;
	}
	else if (inference_type==1){
		matrix_size = s*output_h;
	}
		 

	float *R = (float *) malloc(matrix_size * sizeof(float));

	/* Initialize a context connecting us to the TEE */
	res = TEEC_InitializeContext(NULL, &ctx);
	if (res != TEEC_SUCCESS)
		errx(1, "TEEC_InitializeContext failed with code 0x%x", res);

	/*
	 * Open a session to the "hello world" TA, the TA will print "hello
	 * world!" in the log when the session is created.
	 */
	res = TEEC_OpenSession(&ctx, &sess, &uuid,
			       TEEC_LOGIN_PUBLIC, NULL, NULL, &err_origin);
	if (res != TEEC_SUCCESS)
		errx(1, "TEEC_Opensession failed with code 0x%x origin 0x%x",
			res, err_origin);

	/*
	 * Execute a function in the TA by invoking it, in this case
	 * we're incrementing a number.
	 *
	 * The value of command ID part and how the parameters are
	 * interpreted is part of the interface provided by the TA.
	 */

	/* Clear the TEEC_Operation struct */
	memset(&op, 0, sizeof(op));

	// resigter shared memory

	/*
	 * Prepare the argument. Pass a value in the first parameter,
	 * the remaining three parameters are unused.
	 */
	op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INOUT, TEEC_MEMREF_TEMP_INOUT, TEEC_VALUE_INOUT, TEEC_NONE);
	//op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INOUT, TEEC_NONE, TEEC_NONE, TEEC_NONE);
	op.params[0].tmpref.buffer = R;
	op.params[0].tmpref.size = matrix_size * sizeof(float);
	op.params[1].tmpref.buffer = dimension_matrix;
	op.params[1].tmpref.size = 5 * sizeof(int);
	op.params[2].value.a = 0;

	/*
	 * TA_HELLO_WORLD_CMD_INC_VALUE is the actual function in the TA to be
	 * called.
	 */
	printf("%s\n", "Passing input to TA");
	res = TEEC_InvokeCommand(&sess, TA_DEMO_INFERENCE, &op,
				 &err_origin);
	if (res != TEEC_SUCCESS)
		errx(1, "TEEC_InvokeCommand failed with code 0x%x origin 0x%x",
			res, err_origin);
	printf("%s\n", "Completed");

	printf("%d\n", op.params[2].value.a);


	/*
	 * We're done with the TA, close the session and
	 * destroy the context.
	 *
	 * The TA will print "Goodbye!" in the log when the
	 * session is closed.
	 */

	TEEC_CloseSession(&sess);

	TEEC_FinalizeContext(&ctx);

	return;
}
