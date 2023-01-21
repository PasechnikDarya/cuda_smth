#include <stdio.h>

const int N = 1024 * 1024;

__global__ void kernel(float* arr) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float x = 2.0f * 3.1415926 * (float) i / (float) N;

  arr[i] = sinf(sqrtf(x));
}

int main() {
  
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  float timerValueGRU;

  float *arr, *cuda_arr;

  cudaEventRecord(start, 0);

  arr = (float *) calloc (N, sizeof(float));
  // pinned memory
  // cudaHostAlloc((void **) &arr, N * sizeof(float), cudaHostAllocDefault);

  cudaMalloc((void**) &cuda_arr, N * sizeof(float));

  kernel <<< N / 256, 256 >>> (cuda_arr);

  cudaMemcpy(arr, cuda_arr, N * sizeof(float), cudaMemcpyDeviceToHost);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&timerValueGRU, start, stop);

  printf("\n GPU computation time: %f ms\n", timerValueGRU);

  free(arr);
  // cudaFreeHost(arr);
  cudaFree(cuda_arr);

  return 0; 
}
