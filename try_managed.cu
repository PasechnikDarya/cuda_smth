#include <stdio.h>

const int N = 1024 * 1024;
__device__ __managed__ int managed_arr[N];


__global__ void kernel(void) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float x = 2.0f * 3.1415926 * (float) i / (float) N;

  managed_arr[i] = sinf(sqrtf(x));
}

int main() {
  
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  float timerValueGRU;
  cudaEventRecord(start, 0);

  kernel <<< N / 256, 256 >>> ();
  cudaDeviceSynchronize();

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&timerValueGRU, start, stop);

  printf("\n GPU computation time: %f ms\n", timerValueGRU);

  return 0; 
}
