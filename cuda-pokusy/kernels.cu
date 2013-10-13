#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include "kernels.h"
#include "time_measure.h"

static cudaDeviceProp gpu_property;

#define S_DELENIM

using namespace std;

__device__ int cuda_get_index(int X, int Y, int N)	// SLOUPEC, RADEK
{
	return Y*N+X;
}
__device__ unsigned int cuda_compute_inverse_eukleides(unsigned int cislo, unsigned int modul)
{
	unsigned int a, b, a1, a2, q, r;
	a = cislo;
	b = modul;
	a1 = 0;
	a2 = 1;
	int plus = 1;

	while( b!=0 )
	{
		q = a / b;
		r = a % b;
		a = b;
		b = r;
		r = a1;
		a1 = a2 + r*q;
		a2 = r;
		plus=-plus;
	}
	if( a==1 )
	{
		if( 0<plus )
		{
			return (unsigned int)a2;
		}else
		{
			return (unsigned int)(modul-a2);
		}
	}
	return (unsigned int)0;
}

// elementarni uprava s delenim
__device__ unsigned int cuda_elem_uprava_s_delenim(unsigned int modul, unsigned int a_xy, unsigned int a_xp, unsigned int a_py)
// \STATE $a_{xy} := a_{xy} - a_xp \cdot a_py$
{
	unsigned long long m1;
	unsigned long long pom;

	pom = a_xy;
	m1 = a_xp;
	
	m1 *= a_py;
	if(pom >= m1)
	{
		pom -= m1;
		pom %= modul;
	}else
	{
		m1 -= pom;
		m1 %= modul;
		pom = modul-m1;
	}
	return ((unsigned int)pom);
}
// elementarni uprava bez deleni
__device__ unsigned int cuda_elem_uprava_bez_deleni(unsigned int modul, unsigned int a_xy, unsigned int a_pp, unsigned int a_xp, unsigned int a_py)
// \STATE $a_{xy} := a_{xy} \cdot a_pp - a_xp \cdot a_py$
{
	unsigned long long m1;
	unsigned long long pom;

	pom = a_xy;
	m1 = a_xp;
	
	pom *= a_pp;
	m1 *= a_py;
	if(pom >= m1)
	{
		pom -= m1;
		pom %= modul;
	}else
	{
		m1 -= pom;
		m1 %= modul;
		pom = modul-m1;
	}
	return ((unsigned int)pom);
}
/* nacte/ulozi podmatici z globalni p. do sdilene nebo zpet
 * Sx, Sy - velikost podmatice, mela by se vejit do sdilene pameti
 * sx, sy - souradnice zvolene podmatice v matici, sx \in [0; ceil(N/Sx)]
 * mat_A, mat_B - zdrojova nebo cilova adresa
 */
//#define COPY_MAT_B_GLOB_TO_A_SH	1
//#define COPY_MAT_A_SH_TO_B_GLOB	2
//#define COPY_MAT_A_SH_TO_B_SH 	3
__device__ void cuda_copy_podmatice(int N, int sx, int sy, int Sx, int Sy, unsigned int* mat_A, unsigned int* mat_B, unsigned int* prava_str, int copy_to)
{
	int tid=threadIdx.x;
	int bdim=blockDim.x;
	int itid=tid;
	while(itid<Sy)
	{
		for(int ix=0;ix<Sx;ix++)
		{
			int glob_x=sx*Sx+ix;
			int glob_y=sy*Sy+itid;
			if(glob_x<=N && glob_y<N)
			{
				if(glob_x<N)
				{
					switch(copy_to)
					{
					case COPY_MAT_A_SH_TO_B_GLOB:
						mat_B[cuda_get_index(glob_x, glob_y, N)] = mat_A[cuda_get_index(ix, itid, Sx)];
						break;
					case COPY_MAT_B_GLOB_TO_A_SH:
						mat_A[cuda_get_index(ix, itid, Sx)] = mat_B[cuda_get_index(glob_x, glob_y, N)];
						break;
					}
				}else
				{
					switch(copy_to)
					{
					case COPY_MAT_A_SH_TO_B_GLOB:
						prava_str[glob_y] = mat_A[cuda_get_index(ix, itid, Sx)];
						break;
					case COPY_MAT_B_GLOB_TO_A_SH:
						mat_A[cuda_get_index(ix, itid, Sx)] = prava_str[glob_y];
						break;
					}
				}
			}else
			{
				if(copy_to == COPY_MAT_B_GLOB_TO_A_SH)
				{
					//if( sx==sy && ix==itid )
					//mat_A[get_index(ix, itid, Sx)] = 1;
					//else
					mat_A[cuda_get_index(ix, itid, Sx)] = 0;
				}
			}
		}
		itid+=bdim;
	}
}
/* 
 * gauss-jordanova eliminace, jednovlaknova, ve while-cyklech, primo na datech ve vstupnim poli, 
 * bez deleni - nasobim oba mergujici radky, po vypoctu kazde bunky se moduluje, 
 * dva pristupy k matici: ipivot prochazi pres matici pres radky/sloupce
 * void gauss_jordan_elim_while(int Sx, int Sy, unsigned int modul, unsigned int* m_matice)
 */
__device__ void gauss_jordan_elim_while_kernel(int Sx, int Sy, unsigned int modul, unsigned int* m_matice)
{
	// TODO: posouvat cisla v radcich doleva, kvuli CUDA, aby se pristupovalo stale na ty stejna mista v pameti, 
	//       vysledek bude v prvnim sloupci matice
	int Smin=min(Sx, Sy);
	int tid=threadIdx.x;
	int bdim=blockDim.x;
	int itid;
	for(int ipivot=0;ipivot<Smin;ipivot++)
	{
		__shared__ int novy_pivot;	// CUDA: shared
		__syncthreads();
		if(tid==0)
		{
			novy_pivot=ipivot;
			// deleni nulou => nasobeni inverznim prvkem
			if(m_matice[cuda_get_index(ipivot, ipivot, Sx)]==0)
			{
				// v 'ipivot'-tem radku na diagonále je nula => vymena s jinym radkem
				do{
					novy_pivot++;
				}while(m_matice[cuda_get_index(ipivot, novy_pivot, Sx)]==0 && novy_pivot<Smin);
			}
		}
		__syncthreads();
		// matice je singularni
		if(novy_pivot>=Smin)
		{
			// matice nema v 'ipivot'-tem sloupci nenulovy prvek => je singularni
			//cout << "singularni" << endl;
			itid=tid;
			// singularni matice => vysledky jsou nulove (nepouzitelne)
			//while(itid<=N)
			{
					
				itid+=bdim;
			}
			return;
		}
		// musim prehodit pivotni radek s jinym
		if(novy_pivot>ipivot)
		{
			// vymena radku ipivot a novy_pivot
			itid=tid;
			unsigned int pom;
			while(itid<=Sx)
			{
				pom=m_matice[cuda_get_index(itid, ipivot, Sx)];
				m_matice[cuda_get_index(itid, ipivot, Sx)]=m_matice[cuda_get_index(itid, novy_pivot, Sx)];
				m_matice[cuda_get_index(itid, novy_pivot, Sx)]=pom;
				itid+=bdim;
			}
		}

		__syncthreads();
#ifdef S_DELENIM
		unsigned int a_pp_inv = cuda_compute_inverse_eukleides(m_matice[cuda_get_index(ipivot, ipivot, Sx)], modul);
		// vydelit cely ipivot-ty radek cislem a_pp
		itid=tid;
		while(itid<Sx)
		{
			unsigned long long pom = m_matice[cuda_get_index(itid, ipivot, Sx)];
			pom *= a_pp_inv;
			pom %= modul;
			m_matice[cuda_get_index(itid, ipivot, Sx)]=(unsigned int)pom;

			itid+=bdim;
		}
#else
		unsigned int a_pp = m_matice[cuda_get_index(ipivot, ipivot, Sx)];
#endif

		 /*
		itid=tid;
		while(itid<Sy)	// prochazi jednotlive radky
		{
			if(itid!=ipivot)
			{
				unsigned int a_py = m_matice[cuda_get_index(ipivot, itid, Sx)];

				for(int iX=0;iX<Sx;iX++)	// prochazi cisla v i1-tem radku
				{
					unsigned int a_xy = m_matice[cuda_get_index(iX, itid, Sx)];
					unsigned int a_xp = m_matice[cuda_get_index(iX, ipivot, Sx)];
#ifdef S_DELENIM
					m_matice[cuda_get_index(iX, itid, Sx)] = cuda_elem_uprava_s_delenim(modul, a_xy, a_xp, a_py);
#else
					m_matice[cuda_get_index(iX, itid, Sx)] = cuda_elem_uprava_bez_deleni(modul, a_xy, a_pp, a_xp, a_py);
#endif
				}
			}
			itid+=bdim;
		}
		/*/
		for(int iY=0;iY<Sy;iY++)	// prochazi jednotlive radky
		{
			if(iY!=ipivot)
			{
				unsigned int a_py = m_matice[cuda_get_index(ipivot, iY, Sx)];
				// DEBUG
				itid=tid;
				while(itid<Sx)	// prochazi cisla v i1-tem radku
				{
					unsigned int a_xy = m_matice[cuda_get_index(itid, iY, Sx)];
					unsigned int a_xp = m_matice[cuda_get_index(itid, ipivot, Sx)];
#ifdef S_DELENIM
					m_matice[cuda_get_index(itid, iY, Sx)] = cuda_elem_uprava_s_delenim(modul, a_xy, a_xp, a_py);
#else
					m_matice[cuda_get_index(itid, iY, Sx)] = cuda_elem_uprava_bez_deleni(modul, a_xy, a_pp, a_xp, a_py);
#endif
					itid+=bdim;
				}
			}
			__syncthreads();
		}//*/
	}
#ifndef S_DELENIM
	unsigned long long pom;
	itid=tid;
	while(itid<Smin)
	{
		pom = m_matice[cuda_get_index(Sx-1, itid, Sx)];
		pom *= cuda_compute_inverse_eukleides(m_matice[cuda_get_index(itid, itid, Sx)], modul);
		pom %= modul;
		m_matice[cuda_get_index(Sx-1, itid, Sx)] = (unsigned int)pom;
		itid+=bdim;
	}
#endif
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
__global__ void cuda_GJE_podmatice(int N, int modul, unsigned int* g_matice, unsigned int* g_prava_strana)
{
	int Sx=N+1;
	int Sy=N;
	__shared__ unsigned int s_mat[4000];
	cuda_copy_podmatice(N, 0, 0, Sx, Sy, s_mat, g_matice, g_prava_strana, COPY_MAT_B_GLOB_TO_A_SH);
	gauss_jordan_elim_while_kernel(Sx, Sy, modul, s_mat);
	cuda_copy_podmatice(N, 0, 0, Sx, Sy, s_mat, g_matice, g_prava_strana, COPY_MAT_A_SH_TO_B_GLOB);
}
void cuda_GJE_while(int N, int modul, unsigned int* m_matice, unsigned int* m_prava_strana)
{
	// TODO: dynamicky alokovana sdilena pamet
	unsigned int *g_matice, *g_prava_strana;
	cudaMalloc((void**)&g_matice, (N*N)*sizeof(unsigned int));
	cudaMalloc((void**)&g_prava_strana, N*sizeof(unsigned int));
	cudaMemcpy(g_matice, m_matice, (N*N)*sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy(g_prava_strana, m_prava_strana, N*sizeof(unsigned int), cudaMemcpyHostToDevice);
	unsigned int start_time=get_milisec_from_startup();
	cuda_GJE_podmatice<<<1,32>>>(N, modul, g_matice, g_prava_strana);
	cudaThreadSynchronize();
	gpu_time=(get_milisec_from_startup()-start_time);
	cudaMemcpy(m_matice, g_matice, (N*N)*sizeof(unsigned int), cudaMemcpyDeviceToHost);
	cudaMemcpy(m_prava_strana, g_prava_strana, N*sizeof(unsigned int), cudaMemcpyDeviceToHost);
	cudaFree(g_matice);
	cudaFree(g_prava_strana);

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
	printf("Pocet CUDA zarizeni: %d\n", count);
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
