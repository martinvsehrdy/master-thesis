#include "stdafx.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda.h>
//#include "kernels.h"
#include "kernels_cpu.h"

using namespace std;



int main(char** argv, int argc)
{
	int N=10;
	int* A=new int[N*N];
	int* b=new int[N];
	int* jm=new int[N];
	//for(int i=0;i<N*N;i++) A[i]=0;

	load_matrix(&N, &A, &b, "../diplomka/mat-int.txt");
	cout << N << endl;
	vypsat_mat(N, A, b);
	
	gauss_jordan_elim_for(N, 100001, A, b, jm);

	vypsat_vys(N, b, jm);
	/*
	int* cuda_A;
	cudaMalloc((void**)&cuda_A, N*N*sizeof(int));
	cudaMemcpy(cuda_A, A, N*N*sizeof(int), cudaMemcpyHostToDevice);
	int* cuda_b=NULL;
	cudaMalloc((void**)&cuda_b, N*sizeof(int));
	cudaErr=cudaMemcpy(cuda_b, b, N*sizeof(int), cudaMemcpyHostToDevice);
	*/



	//cudaFree(cuda_b);
	

#ifdef _DEBUG
	cin.get();
#else
	delete[] A;
	delete[] b;
	delete[] jm;
#endif
}