#include <stdio.h>
#include <iostream>

#include <cuda_runtime.h>


using namespace std;

__global__ void cuda_kernel(int N, int modul,  int* m_matice, int* m_prava_strana, int* m_vys_jmenovatel);
void cuda_gauss_jordan_elim(int N, int modul, int* m_matice, int* m_prava_strana, int* m_vys_jmenovatel);
__device__ int cuda_get_index(int X, int Y, int N);

__device__ int cuda_m_inv(int modulo, int cislo)
{
	// TODO
	if(cislo==0) return 0;
	int i;
	for(i=1;i<=modulo;i++)
	{
		if( (cislo*i)% modulo==1 ) break;
	}
	return i;
}

template<class T>
__global__ void modulovat(int N, T* poleIn, int* poleOut, int modul)
{
	int tid=threadIdx.x;
	while(tid<N)
	{
		poleOut[tid]=((int)poleIn[tid]) % modul;
		tid+=blockDim.x;
	}
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
