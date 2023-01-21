#include <iostream>

__global__ void kernel(int i, int j, ){

}

int main(void) {

  kernel<<<1, 1>>>();
  std::cout << "Hello, world!" << std::endl;

  return 0;
}