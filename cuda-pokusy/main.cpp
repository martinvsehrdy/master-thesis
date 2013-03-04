#include "stdafx.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda.h>
#include "kernels.h"
#include "kernels_cpu.h"
#include <cuda_runtime.h>

using namespace std;



int main(char** argv, int argc)
{
	int N=10;
	int modul=1073741824;
	int* A=new int[N*N];
	int* b=new int[N];
	int* jm=new int[N];
	//for(int i=0;i<N*N;i++) A[i]=0;

	load_matrix(&N, &A, &b, "../diplomka/mat-int.txt");
	cout << N << endl;
	vypsat_mat(N, A, b);
	
	// inicializace CUDA
	int* cuda_A;
	cudaMalloc((void**)&cuda_A, N*N*sizeof(int));
	cudaMemcpy(cuda_A, A, N*N*sizeof(int), cudaMemcpyHostToDevice);
	int* cuda_b=NULL;
	cudaMalloc((void**)&cuda_b, N*sizeof(int));
	cudaMemcpy(cuda_b, b, N*sizeof(int), cudaMemcpyHostToDevice);
	int* cuda_jm=NULL;
	cudaMalloc((void**)&cuda_jm, N*sizeof(int));
	
	// vypocet v CUDA
	cuda_gauss_jordan_elim(N, modul, cuda_A, cuda_b, cuda_jm);
	
	// kopirovani z GPU
	cudaMemcpy(b, cuda_b, N*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(jm, cuda_jm, N*sizeof(int), cudaMemcpyDeviceToHost);


	vypsat_vys(N, b, jm);

	print_gpus_info();
	cudaFree(cuda_A);
	cudaFree(cuda_b);
	cudaFree(cuda_jm);
	

#ifdef _DEBUG
	cin.get();
#else
	delete[] A;
	delete[] b;
	delete[] jm;
#endif
}