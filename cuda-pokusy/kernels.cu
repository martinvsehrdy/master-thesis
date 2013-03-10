#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

static cudaDeviceProp gpu_property;

using namespace std;

__device__ int cuda_get_index(int X, int Y, int N)
{
	return X*N+Y;
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
__global__ void kernel(int N, int* pole, int cislo)
{
	int tid=threadIdx.x;
	while(tid<N)
	{
		pole[tid]=5;
		tid+=blockDim.x;
	}
}
__global__ void kernel_gauss_jordan_elim(int N, int modul,  int* m_matice, int* m_prava_strana, int* m_vys_jmenovatel, int* retval)
{
	int tid=threadIdx.x;
	int itid;
	for(int ipivot=0;ipivot<N;ipivot++)
	{
		// deleni nulou => nasobeni inverznim prvkem
		if(m_matice[cuda_get_index(ipivot, ipivot, N)]==0)
		{
			// v 'ipivot'-tem radku na diagonále je nula => vymena s jinym radkem
			int novy_pivot=ipivot;
			do{
				novy_pivot++;
			}while(m_matice[cuda_get_index(ipivot, novy_pivot, N)]==0 && novy_pivot<N);

			if(m_matice[cuda_get_index(ipivot, novy_pivot, N)]!=0 && novy_pivot<N)		// nasel jsem radek s nenulovym prvkem ve sloupci ipivot
			{
				// vymena radku ipivot a novy_pivot
				int pom;
				itid=tid;
				while(itid<=N)
				{
					if(itid==N)
					{
						pom=m_prava_strana[ipivot];
						m_prava_strana[ipivot]=m_prava_strana[novy_pivot];
						m_prava_strana[novy_pivot]=pom;
					}else
					{
						pom=m_matice[cuda_get_index(itid, ipivot, N)];
						m_matice[cuda_get_index(itid, ipivot, N)]=m_matice[cuda_get_index(itid, novy_pivot, N)];
						m_matice[cuda_get_index(itid, novy_pivot, N)]=pom;
					}
					itid+=blockDim.x;
				}
			}else
			{
				// matice nema v 'ipivot'-tem sloupci nenulovy prvek => je singularni
				*retval=1;
				//cout << "singularni" << endl;
				itid=tid;
				while(itid<=N)	// singularni matice => vysledky jsou nulove = nepouzitelne, nemusi to tu byt
				{
					m_prava_strana[itid]=0;
					m_vys_jmenovatel[itid]=1;
					itid+=blockDim.x;
				}
				return;
			}
		}
		int multipl1 = m_matice[cuda_get_index(ipivot, ipivot, N)];
		//*/
		itid=tid;
		while(itid<N)	// prochazi jednotlive radky
		{
			if(itid==ipivot)
			{
				itid+=blockDim.x;
				continue;
			}
			int pom;
			int multipl2 = m_matice[cuda_get_index(ipivot, itid, N)];
			for(int iX=0;iX<N;iX++)	// prochazi cisla v i1-tem radku
			{
				int m1=m_matice[cuda_get_index(iX, itid, N)];
				int m2=m_matice[cuda_get_index(iX, ipivot, N)];
				pom = multipl1*m1-multipl2*m2;
				pom=pom % modul;
				m_matice[cuda_get_index(iX, itid, N)]=pom;
			}
			pom = multipl1*m_prava_strana[itid]-multipl2*m_prava_strana[ipivot];
			m_prava_strana[itid]=pom % modul;
			itid+=blockDim.x;
		}
		/*/
		for(int iY=0;iY<N;iY++)	// prochazi jednotlive radky
		{
			if(iY==ipivot) continue;
			int pom;
			int multipl2 = m_matice[cuda_get_index(ipivot, iY, N)];
			itid=tid;
			while(itid<N)	// prochazi cisla v i1-tem radku
			{
				int m1=m_matice[cuda_get_index(itid, iY, N)];
				int m2=m_matice[cuda_get_index(itid, ipivot, N)];
				// TODO: jak cuda moduluje hlavne zaporny cisla? potrebuju interval <0;modul)
				pom = multipl1*m1-multipl2*m2;
				pom=pom % modul;
				//if(pom<0) pom+=modul;
				m_matice[cuda_get_index(itid, iY, N)]=pom;
				itid+=blockDim.x;
			}
			pom = multipl1*m_prava_strana[iY]-multipl2*m_prava_strana[ipivot];
			// TODO: jak cuda moduluje hlavne zaporny cisla? potrebuju interval <0;modul)
			m_prava_strana[iY]=pom % modul;
			//if(m_prava_strana[iY]<0) m_prava_strana[iY]+=modul;
		}//*/
		
		// TODO: _syncthread();
	}
	// ulozit diagonalu do m_vys_jmenovatel
	itid=tid;
	while(itid<N)
	{
		m_vys_jmenovatel[itid]=m_matice[cuda_get_index(itid, itid, N)];
		itid+=blockDim.x;
	}
	*retval=0;
}
void cuda_gauss_jordan_elim(int N, int modul, int* m_matice, int* m_prava_strana, int* m_vys_jmenovatel, int* retval)
{
	// TODO: posouvat cisla v radcich doleva, kvuli CUDA, aby se pristupovalo stale na ty stejna mista v pameti, 
	//       vysledek bude v prvnim sloupci matice
	kernel_gauss_jordan_elim<<<1,gpu_property.maxThreadsPerBlock>>>(N, modul, m_matice, m_prava_strana, m_vys_jmenovatel, retval);

	//kernel<<<1, gpu_property.maxThreadsPerBlock>>>(N, m_prava_strana, 5);
	
}
void init_gpu_compute(void)
{
	int count;
    cudaGetDeviceCount( &count);
	if (0<count) cudaGetDeviceProperties( &gpu_property, 0);
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
