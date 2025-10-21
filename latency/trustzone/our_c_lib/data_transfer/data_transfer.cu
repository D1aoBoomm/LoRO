#include <stdio.h>
#include <cuda_runtime.h>
#include <assert.h>

// if you are in windows, you need to change this. I have not tried yet.
extern "C" {
    float transfer(int s, int h);
}

inline cudaError_t checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}

float profileCopies(float* h_a, float* h_b, float* d, unsigned int n, char* desc, FILE *file){
    //printf("\n%s transfers\n", desc);

    unsigned int bytes = n * sizeof(float);

    cudaEvent_t startEvent, stopEvent;

    checkCuda(cudaEventCreate(&startEvent));
    checkCuda(cudaEventCreate(&stopEvent));

    // Host to Device
    checkCuda(cudaEventRecord(startEvent, 0));
    checkCuda(cudaMemcpy(d, h_a, bytes, cudaMemcpyHostToDevice));
    checkCuda(cudaEventRecord(stopEvent, 0));
    checkCuda(cudaEventSynchronize(stopEvent));

    float time_0;
    checkCuda(cudaEventElapsedTime(&time_0, startEvent, stopEvent));
    //printf("Host to Device Time to transfer %u bytes of data (%s): %f ms\n", bytes, desc, time_0);
    //printf("Host to Device bandwidth GB/s: %f\n", bytes*1e-6/time_0);

    if (file && strcmp(desc, "Pinned") == 0) {
        fprintf(file, "%f\n", time_0);
    }

    // Device to Host
    checkCuda(cudaEventRecord(startEvent, 0));
    checkCuda(cudaMemcpy(h_b, d, bytes, cudaMemcpyDeviceToHost));
    checkCuda(cudaEventRecord(stopEvent, 0));
    checkCuda(cudaEventSynchronize(stopEvent));

    float time_1;
    checkCuda(cudaEventElapsedTime(&time_1, startEvent, stopEvent));
    //printf("Device to Host Time to transfer %u bytes of data (%s): %f ms\n", bytes, desc, time_1);
    //printf("Device to Host bandwidth GB/s: %f\n", bytes*1e-6/time_1);

    if (file && strcmp(desc, "Pinned") == 0) {
        fprintf(file, "%f\n", time_1);
    }

    checkCuda(cudaEventDestroy(startEvent));
    checkCuda(cudaEventDestroy(stopEvent));

    return time_0+time_1;
}

float transfer(int s, int h) {
    unsigned int nElement = 1*s*h;
    const unsigned int bytes = nElement * sizeof(float);

    float *h_aPageable, *h_bPageable;
    float *h_aPinned, *h_bPinned;

    float *d_a;

    h_aPageable = (float*)malloc(bytes);
    h_bPageable = (float*)malloc(bytes);

    checkCuda(cudaMallocHost((void**)&h_aPinned, bytes));
    checkCuda(cudaMallocHost((void**)&h_bPinned, bytes));

    checkCuda(cudaMalloc((void**)&d_a, bytes));

    for (int i=0; i<nElement; i++){
        h_aPageable[i] = i;
    }

    memcpy(h_aPinned, h_aPageable, bytes);
    memset(h_bPageable, 0, bytes);
    memset(h_bPinned, 0, bytes);

    cudaDeviceProp prop;
    checkCuda(cudaGetDeviceProperties(&prop, 0));

    //printf("\nDevice: %s\n", prop.name);
    //printf("Transfer size (MB): %d \n", bytes/1024/1024);

    // Open file to write the transfer times for pinned memory
    // FILE *file = fopen("data_transfer.txt", "w");
    // if (!file) {
    //     perror("Unable to open file for writing");
    //     return 1;
    // }

    float time_0 = profileCopies(h_aPageable, h_bPageable, d_a, nElement, "Pageable", NULL);
    float time_1 = profileCopies(h_aPinned, h_bPinned, d_a, nElement, "Pinned", NULL);

    //printf("\n");

    // fclose(file);

    cudaFree(d_a);
    cudaFreeHost(h_aPinned);
    cudaFreeHost(h_bPinned);
    free(h_aPageable);
    free(h_bPageable);

    return time_1;
}

int main() {
    float time = transfer(512, 768);
    printf("\nTotal time: %f\n", time);
    return 1;
}