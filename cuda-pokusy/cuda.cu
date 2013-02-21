#ifndef _CUDA_CU_
#define _CUDA_CU_

#include "stdafx.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda.h>
//#include "kernels.cu"
#include <cuda_runtime.h>

using namespace std;


void print_gpus_info(void);
void print_cuda_err(cudaError_t cudaErr);

#define T double

__global__ void modulovat(int N, int* pole, int modul)
{
	int tid=threadIdx.x;
	if(tid<N) pole[tid]=pole[tid] % modul;
}

cudaError_t cudaErr;

int main(char** argv, int argc)
{
	int N=10;
	int* A=new int[N*N];
	int* b=new int[N];

	for(int i=0;i<N;i++)
	{
		b[i] = rand()-15000;
		cout << b[i] << "\t";
	}
	

	int* cuda_A;
	cudaMalloc((void**)&cuda_A, N*N*sizeof(int));
	cudaMemcpy(cuda_A, A, N*N*sizeof(int), cudaMemcpyHostToDevice);
	int* cuda_b=NULL;
	cudaMalloc((void**)&cuda_b, N*sizeof(int));
	cudaErr=cudaMemcpy(cuda_b, b, N*sizeof(int), cudaMemcpyHostToDevice);
	print_cuda_err(cudaErr);
	modulovat<<<1,1024>>>(N, cuda_b, 11);
	cout << endl << endl;
	for(int i=0;i<N;i++) b[i] = 0;

	cudaErr=cudaMemcpy(b, cuda_b, N*sizeof(int), cudaMemcpyDeviceToHost);
	print_cuda_err(cudaErr);
	
	cudaFree(cuda_b);
	for(int i=0;i<N;i++) cout << b[i] << "\t";

	delete[] b;

#ifdef _DEBUG
	cin.get();
#endif
}


void gauss_jordan_elim(int modul, int* m_matice, int* m_prava_strana, int* m_vys_citatel, int* m_vys_jmenovatel)
{

}

void print_gpus_info(void)
{
	cudaDeviceProp prop;
    int count;
 
    cudaGetDeviceCount( &count);

	for (int i=0; i< count; i++)
	{
		cudaGetDeviceProperties( &prop, i);

		printf( " --- General Information for device %d ---\n", i );
		printf( "Name: %s\n", prop.name );
		printf( "Compute capability: %d.%d\n", prop.major, prop.minor );
		printf( "Clock rate: %d\n", prop.clockRate );
		printf( "Device copy overlap: " );
       
		if (prop.deviceOverlap)
			printf( "Enabled\n" );
		else
			printf( "Disabled\n" );
		printf( "Kernel execition timeout : " );


		if (prop.kernelExecTimeoutEnabled)
			printf( "Enabled\n" );
		else
			printf( "Disabled\n" );
    
		printf( " --- Memory Information for device %d ---\n", i );
		printf( "Total global mem: %ld\n", prop.totalGlobalMem );
		printf( "Total constant Mem: %ld\n", prop.totalConstMem );
		printf( "Max mem pitch: %ld\n", prop.memPitch );
		printf( "Texture Alignment: %ld\n", prop.textureAlignment );
		printf( " --- MP Information for device %d ---\n", i );
		printf( "Multiprocessor count: %d\n",
		prop.multiProcessorCount );
		printf( "Shared mem per mp: %ld\n", prop.sharedMemPerBlock );
		printf( "Registers per mp: %d\n", prop.regsPerBlock );
		printf( "Threads in warp: %d\n", prop.warpSize );
		printf( "Max threads per block: %d\n",
		prop.maxThreadsPerBlock );
		printf( "Max thread dimensions: (%d, %d, %d)\n",
		prop.maxThreadsDim[0], prop.maxThreadsDim[1],
		prop.maxThreadsDim[2] );
		printf( "Max grid dimensions: (%d, %d, %d)\n",
		prop.maxGridSize[0], prop.maxGridSize[1],
		prop.maxGridSize[2] );
		printf( "\n" );
	}
}

void print_cuda_err(cudaError_t cudaErr)
{
	switch(cudaErr)
	{
	case cudaSuccess: cout << "cudaSuccess";
		break;
	case cudaErrorInvalidValue: cout << "cudaErrorInvalidValue";
		break;
	case cudaErrorInvalidDevicePointer: cout << "cudaErrorInvalidDevicePointer";
		break;
	case cudaErrorInvalidMemcpyDirection: cout << "cudaErrorInvalidMemcpyDirection";
		break;
	}
}

#endif /* _CUDA_CU_ */
