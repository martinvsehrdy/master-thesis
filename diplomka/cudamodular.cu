#ifndef _CUDAMODULAR_CU_
#define _CUDAMODULAR_CU_

#include "stdafx.h"
#include "global.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

cudaError_t cudaErr;

/*
 * vynasobi radky tak, aby kazde cislo melo za desetinou carkou pouze nuly
 */
void cuda_vstupni_slr(int N, TYPE* matice, TYPE* prava_strana)
{
	
}
/*
 * spocita hadamarduv odhad a modul M
 */
void cuda_exec_hadamard(int N,  TYPE* matice, TYPE* prava_strana, mpz_class* hadamard, TYPE* modul_M)
{
	
}
/*
 * rozlozi modul M na soucin jednotlivych modulu vzajemne nesoudelnych
 * r - pocet jednotlivych modulu
 * moduly - pole jednotlivych modulu
 * M - vstupni modul M
 */
void cuda_exec_moduly(int N,  TYPE* matice, TYPE* prava_strana, mpz_class hadamard, TYPE modul_M, int* r, int** moduly)
{
	
}
/*
 * vytvori SLR v modulu "modul", zmoduluje vstupní SLR
 * modul - vybrany jednotlivy modul
 * m_matice - tady bude matice modularnich zbytku
 * m_prava_strana - prava strana   - || -
 */
void cuda_rozklad_slr_mod(int N,  TYPE* matice, TYPE* prava_strana, int modul, int* m_matice, int* m_prava_strana)
{
	return;
}
__global__ void kernel1(int N, int modul,  int* m_matice, int* m_prava_strana, int* m_vys_jmenovatel)
{
	int tid=threadIdx.x;
	if(tid<N)
	{
		m_vys_jmenovatel[tid]=m_prava_strana[tid];
		m_prava_strana[tid]=1;
	}
}
void cuda_gauss_jordan_elim(int N, int modul, int* m_matice, int* m_prava_strana, int* m_vys_jmenovatel)
{
	// TODO: posouvat cisla v radcich doleva, kvuli CUDA, aby se pristupovalo stale na ty stejna mista v pameti, 
	//       vysledek bude v prvnim sloupci matice
	kernel1<<<1,N>>>(N, modul, m_matice, m_prava_strana, m_vys_jmenovatel);
}
/*
 * r - pocet jednotlivych modulu
 */
void cuda_zpetny_prevod(int r, int** vys_citatel, int** vys_jmenovatel, TYPE* vysledek)
{
}

void cuda_do_modular(int N,  TYPE* matice, TYPE* prava_strana)
{
	if(matice==NULL || prava_strana==NULL) return;
		
	
}

#endif /* _CUDAMODULAR_CU_ */