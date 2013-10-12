#include "stdafx.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <list>
#include "kernels_cpu.h"
#include "templates_functions.h"
#include "kernels.h"
#include "time_measure.h"

using namespace std;

#define POC_OPAKOVANI 10
float cuda_time1, cuda_time2, cuda_time3;

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
	print_gpus_info();
	int N=4;
	unsigned int modul=0x10000001; //(~(unsigned int)0);
	modul |= rand();
	//modul = 0x1003;	// 4099 je prvocislo
	
	unsigned int* V=new unsigned int[N*N];
	unsigned int* M=new unsigned int[N];

	//load_matrix<unsigned int>(&N, &V, &M, "mat-int.txt");
	hilbert_matrix<unsigned int>(N, V, M);
	vypsat_mat<unsigned int>(N, N, V, M);
	//GJE_podmatice(N, modul, V, M, NULL);
	gauss_jordan_elim_for(N, modul, V, M, NULL);
	vypsat_mat<unsigned int>(N, N, V, M);
	
	cout << "===================================================" << endl;
	unsigned int* S=new unsigned int[N*N+N];
	//load_matrix<unsigned int>(&N, &V, &M, "mat-int.txt");
	hilbert_matrix<unsigned int>(N, V, M);
	vypsat_mat<unsigned int>(N, N, V, M);
	//*
	// na GPU
	cuda_GJE_while(N, modul, V, M);
	/*/
	// na CPU
	copy_podmatice(N, 0, 0, N+1, N, S, V, M, COPY_MAT_B_GLOB_TO_A_SH);
	gauss_jordan_elim_while(N+1, N, modul, S);
	copy_podmatice(N, 0, 0, N+1, N, S, V, M, COPY_MAT_A_SH_TO_B_GLOB);
	//*/
	vypsat_mat<unsigned int>(N, N, V, M);
	
#ifdef _DEBUG
	cin.get();
#endif
	free(V);
	free(M);
	free(S);
	return 0;//*/
	////////////////////////////////////////////////////////

	if(argc>2)
	{
		N=atoi(argv[1]);
		modul=atoi(argv[2]);
	}else
	{
		cout << "#Program spustte ve tvaru:" << argv[0] << " <N> <modul>" << endl;
		cout << "#Vystup: <velikost N> <na GPU [ms]>\t<z GPU [ms]>\tprumer\tnejrychlejsi\t1.quartal\tmedian\t3.quartal\tnejpomalejsi\t<celkem [ms]>" << endl;
		return 0;
	}
	// TODO: vypocet


#ifdef _DEBUG
	cin.get();
#else
	
#endif
}