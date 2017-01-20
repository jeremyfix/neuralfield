// g++ -o test test.cpp `pkg-config --libs --cflags fftw3` -std=c++11

#include <algorithm>
#include <iostream>
#include <cstring>

#include "../src/convolution_fftw.h"

void print_array(double* values, int size) {
  for(unsigned int i = 0 ; i < size; ++i, ++values) {
    std::cout << *values << " ";
  }
  std::cout << std::endl;
}

int main(int argc, char * argv[]) {

  bool toric = atoi(argv[1]);
  
  int N = 10;
  
  
  FFTW_Convolution::Workspace ws;
  unsigned int k_size;
  if(!toric) { // 0
    k_size = 2 * N - 1;
    FFTW_Convolution::init_workspace(ws, FFTW_Convolution::LINEAR_SAME, 1, N, 1, k_size);
  }
  else { // 1
    k_size = N;
    FFTW_Convolution::init_workspace(ws, FFTW_Convolution::CIRCULAR_SAME, 1, N, 1, k_size);
  }
  
  double* src = new double[N];
  std::fill(src , src+N, 0.);
  src[0] = 1;
  
  double* kernel = new double[k_size];
  std::fill(kernel , kernel+k_size, 0.);
  std::fill(kernel+(N/2-2), kernel + (N/2+2), 1.);
  
  double* res = new double[N];
  std::fill(res , res+N, 0.);
  

  FFTW_Convolution::convolve(ws, src, kernel);
  memcpy(res, ws.dst, N* sizeof(double));

  std::cout << "Input : " << std::endl;
  print_array(src, N);

  std::cout << "kernel : " << std::endl;
  print_array(kernel, k_size);

  std::cout << "result : " << std::endl;
  print_array(res, N);
  
  FFTW_Convolution::clear_workspace(ws);

  delete[] src;
  delete[] kernel;
  delete[] res;
}
