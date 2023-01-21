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

  float *shared_arr;

  cudaEventRecord(start, 0);

  cudaMallocManaged((void**) &shared_arr, N * sizeof(float));

  kernel <<< N / 256, 256 >>> (shared_arr);

  cudaDeviceSynchronize();

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&timerValueGRU, start, stop);

  printf("\n GPU computation time: %f ms\n", timerValueGRU);

  cudaFree(shared_arr);

  return 0; 
}
