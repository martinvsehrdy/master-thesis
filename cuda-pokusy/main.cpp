#include "stdafx.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <list>
#include "kernels_cpu.h"
#include "templates_functions.h"
#include "kernels.h"
#include "time_measure.h"
#include "common.h"
#include <Windows.h>

using namespace std;

#define POC_OPAKOVANI 10
//extern unsigned int measured_time;

void statistic(list<float> l, float* quartal1, float* quartal2, float* quartal3, float* avg)
{
	l.sort();
	unsigned int poc=0;
	*avg=0.0;
	list<float> l1, l2;
	while(l.size()>0)
	{
		*avg+=l.front();
		poc++;
		if(l.size()>1)	// pro pripad, ze v seznamu zbude jeden posledni prvek
		{
			*avg+=l.back();
			poc++;
		}
		l1.push_back(l.front());
		l2.push_front(l.back());
		l.pop_back();
		l.pop_front();
	}
	if(poc>0) *avg/=poc;
	if(l1.size()>0 && l2.size()>0) *quartal2=(float)((l1.back() + l2.front())/2.0);
	else *quartal2=0;
	while(l1.size()>2 && l2.size()>2)
	{
		l1.pop_back();
		l1.pop_front();
		l2.pop_back();
		l2.pop_front();
	}
	poc=0;
	*quartal1=0;
	*quartal3=0;
	for(list<float>::iterator iter=l1.begin();iter!=l1.end();iter++)
	{
		*quartal1+=*iter;
		poc++;
	}
	if(poc>0) *quartal1/=poc;
	poc=0;
	for(list<float>::iterator iter=l2.begin();iter!=l2.end();iter++)
	{
		*quartal3+=*iter;
		poc++;
	}
	if(poc>0) *quartal3/=poc;
	
}
int main(int argc, char** argv)
// argv[0] <N> <modul>
{
	int N=50;
	unsigned int modul=0x10000003; //(~(unsigned int)0);
	/*modul = 0x1003;	// 4099 je prvocislo
	cout << "Modul = " << modul << endl;
	unsigned int* M=new unsigned int[N*N];
	unsigned int* P=new unsigned int[N];
	for(int y=0;y<N;y++) for(int x=0;x<N;x++) M[get_index(x, y, N)]=10*x+y;
	for(int y=0;y<N;y++) P[y]=800+y;

	unsigned int setting=strtol("00111", NULL, 2);
	hilbert_matrix(N, M, P);
	gauss_jordan_elim_for(N, modul, M, P, setting);
	save_matrix(N, M, P, "outmat-for");

	hilbert_matrix(N, M, P);
	vypsat_mat(N, N, M, P);
	init_gpu_compute();
	//cuda_GJE_podmatice(N, modul, M, P, setting);
	unsigned int* S=new unsigned int[N*N+N];
	copy_podmatice(N, 0, 0, N+1, N, S, M, P, COPY_MAT_B_GLOB_TO_A_SH);
	cuda_GJE_global(N, modul, S, setting);
	copy_podmatice(N, 0, 0, N+1, N, S, M, P, COPY_MAT_A_SH_TO_B_GLOB);
	free(S);

	save_matrix(N, M, P, "outmat-GJE");
	vypsat_mat(N, N, M, P);
	
	delete M;
	delete P;
#ifdef _DEBUG
	cin.get();
#else
#endif
	return 0; /*/
	////////////////////////////////////////////////////////
	int zpusob=0;
	if(argc>2)
	{
		N=atoi(argv[1]);
		zpusob=strtol(argv[2], NULL, 2);
	}else
	{
		// TODO: (ne)vyuzivat sdilenou pamet; modulo v elementarni uprave;
		cout << "#Program spustte ve tvaru:" << argv[0] << " <N> <zpusob zpracovani>" << endl;
		cout << "#zpusob zpracovani: 0. bit \tfor/while(0) while/for(1)" << endl;
		cout << "#(pocitano zprava)  1. bit \tbez deleni(0) s delenim(1)" << endl;
		cout << "#                   2.3.bit \t1(00) 32(01) 128(10) vlaken" << endl;
		cout << "#                   4.bit \tGPU(0) CPU(1)" << endl;
		cout << "#                   5.bit \tmatice v sdilene(0), globalni(1) pameti" << endl;


		cout << "#Vystup: <velikost N> <na GPU [ms]>\t<z GPU [ms]>\tprumer\tnejrychlejsi\t1.quartal\tmedian\t3.quartal\tnejpomalejsi\t<celkem [ms]>" << endl;
		return 0;
	}
	// TODO: vypocet
	unsigned int* A=new unsigned int[N*N];
	unsigned int* b=new unsigned int[N];

#ifndef _DEBUG
	list<float> times;
	times.clear();
	float sum=0.0;
	for(int i=0;i<POC_OPAKOVANI;i++)
	{
#endif
		hilbert_matrix<unsigned int>(N, A, b);
		gauss_jordan_elim_for(N, modul, A, b, zpusob);
		save_matrix(N, A, b, "outmat-for");
		hilbert_matrix<unsigned int>(N, A, b);
#ifdef _DEBUG
	vypsat_mat<unsigned int>(N, N, A, b);
#endif
		float tt=0;
		if(zpusob & ZPUSOB_CPU)
		{
			gauss_jordan_elim_for(N, modul, A, b, zpusob);
			tt = get_measured_time();
		}else
		{
			init_gpu_compute();
			if(zpusob & ZPUSOB_GLOBAL_MEM)
			{
				unsigned int* S=new unsigned int[N*N+N];
				copy_podmatice(N, 0, 0, N+1, N, S, A, b, COPY_MAT_B_GLOB_TO_A_SH);
				cuda_GJE_global(N, modul, S, zpusob);
				copy_podmatice(N, 0, 0, N+1, N, S, A, b, COPY_MAT_A_SH_TO_B_GLOB);
				free(S);
			}else
			{
				cuda_GJE_podmatice(N, modul, A, b, zpusob);
			}
			tt = cuda_get_measured_time();
		}
		save_matrix(N, A, b, "outmat-computed");
#ifndef _DEBUG
		times.push_back(tt);
		sum += tt;
		if( !(zpusob & ZPUSOB_CPU) ) cudaDeviceReset();
		Sleep(100);
	}
	float q1, q2, q3, prumer;
	statistic(times, &q1, &q2, &q3, &prumer);
	times.sort();
	//<velikost N> <na GPU [ms]>\t<z GPU [ms]>\tprumer\tnejrychlejsi\t1.quartal\tmedian\t3.quartal\tnejpomalejsi\t<celkem [ms]>
	cout << N << "\t?\t?\t" << prumer << "\t" << times.front() << "\t" << q1 << "\t" << q2 << "\t" << q3 << "\t" << times.back() << "\t" << sum << endl;
#else
	cout << tt << "ms" << endl;
#endif
	//vypsat_mat<unsigned int>(N, N, A, b);
#ifdef _DEBUG
	vypsat_mat<unsigned int>(N, N, A, b);
	cout << "===================================================" << endl;
#endif
	
#ifdef _DEBUG
	cin.get();
#else
	
#endif
	free(A);
	free(b);
	return 0;
	//*/
}
