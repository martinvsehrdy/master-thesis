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
	int N=8;
	unsigned int modul=0x10000003; //(~(unsigned int)0);
	modul |= rand();
	modul = 0x1003;	// 4099 je prvocislo
	cout << "Modul = " << modul << endl;
	unsigned int* M=new unsigned int[N*N];
	unsigned int* P=new unsigned int[N];
	hilbert_matrix(N, M, P);
	vypsat_mat(N, N, M, P);
	GJE_podmatice(N, modul, M, P, NULL);
	vypsat_mat(N, N, M, P);
#ifdef _DEBUG
	cin.get();
#endif
	////////////////////////////////////////////////////////
	int zpusob=0;
	if(argc>2)
	{
		N=atoi(argv[1]);
		zpusob=atoi(argv[2]);
	}else
	{
		cout << "#Program spustte ve tvaru:" << argv[0] << " <N> <zpusob zpracovani>" << endl;
		cout << "zpusob zpracovani: 1 - na CPU s delenim" << endl;
		cout << "                   2 - na CPU bez deleni" << endl;
		cout << "                   3 - na GPU s deleni" << endl;
		cout << "                   4 - na GPU bez deleni" << endl;
		cout << "#Vystup: <velikost N> <na GPU [ms]>\t<z GPU [ms]>\tprumer\tnejrychlejsi\t1.quartal\tmedian\t3.quartal\tnejpomalejsi\t<celkem [ms]>" << endl;
		return 0;
	}
	// TODO: vypocet
	unsigned int* A=new unsigned int[N*N];
	unsigned int* b=new unsigned int[N];

	hilbert_matrix<unsigned int>(N, A, b);
#ifdef _DEBUG
	vypsat_mat<unsigned int>(N, N, A, b);
#endif

	switch(zpusob)
	{
	case 1:
		gauss_jordan_elim_for(N, modul, A, b, NULL);
		break;
	case 2:
		gauss_jordan_elim_for(N, modul, A, b, NULL);
		break;
	case 3:
		cuda_GJE_while(N, modul, A, b);
		break;
	case 4:
		cuda_GJE_while(N, modul, A, b);
		break;
	}
	cout << measured_time;

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
}
