#include "stdafx.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda.h>
#include "kernels.h"
#include "kernels_cpu.h"
#include <cuda_runtime.h>

using namespace std;

cudaError_t cudaErr;
#define T double




int main(char** argv, int argc)
{
	int N=10;
	int* A=new int[N*N];
	int* b=new int[N];

	load_matrix<int>(&N, A, b, "../diplomka/mat1.txt");
	cout << N << endl;
	//vypsat_mat<int>(N, A, b);
	

	/*
	int* cuda_A;
	cudaMalloc((void**)&cuda_A, N*N*sizeof(int));
	cudaMemcpy(cuda_A, A, N*N*sizeof(int), cudaMemcpyHostToDevice);
	int* cuda_b=NULL;
	cudaMalloc((void**)&cuda_b, N*sizeof(int));
	cudaErr=cudaMemcpy(cuda_b, b, N*sizeof(int), cudaMemcpyHostToDevice);
	*/



	//cudaFree(cuda_b);
	
	delete[] A;
	delete[] b;

#ifdef _DEBUG
	cin.get();
#endif
}


