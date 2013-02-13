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
__global__ void kernel1(double* pole, int N)
{
	int id=threadIdx.x;
	while( id<N )
	{
		pole[id]=1.0;
		id+=1024;
	}
}

int main(char** argv, int argc)
{
	int N=100;
	int sum=0;
	double* A=new double[N];
	for(int i=0;i<N;i++)
	{
		A[i]=i;
		sum+=i;
	}
	for(int i=0;i<N;i++) cout << A[i] << "\t";
	double* cudaA;
	cudaMalloc((void**)&cudaA, N*sizeof(double));
	cudaMemcpy(cudaA, A, N*sizeof(double), cudaMemcpyHostToDevice);
	kernel1<<<1,1024>>>(cudaA, N);
	cout << endl;
	cudaMemcpy(A, cudaA, N*sizeof(double), cudaMemcpyDeviceToHost);
	cudaFree(cudaA);
	for(int i=0;i<N;i++) cout << A[i] << "\t";

	delete[] A;


#ifdef _DEBUG
	cin.get();
#endif
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

#endif /* _CUDA_CU_ */
