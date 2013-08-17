#include "stdafx.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <list>
#include "kernels_cpu.h"
#include "templates_functions.h"

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
	int N=17;
	int modul=0;
	unsigned int* V=new unsigned int[N*N];
	unsigned int* M=new unsigned int[N];

	GJE_podmatice(N, modul, V, M, NULL);
	/*unsigned int* V=new unsigned int[N*N];
	unsigned int* M=new unsigned int[Sx*Sy];
	for(int i=0;i<N*N;i++) V[i]=100;
	for(int y=0;y<ceil((double)N/Sy);y++)
	{
		for(int x=0;x<ceil((double)N/Sx);x++)
		{
			for(int i=0;i<Sx*Sy;i++) M[i]=i;
			copy_podmatice(N, x, y, Sx, Sy, M, V, COPY_TO_GLOBAL_MEM);
		}
	}

	for(int y=0;y<N;y++)
	{
		for(int x=0;x<N;x++)
		{
			printf("%4u", V[get_index(x, y, N)]);
		}
		printf("\n");
	} //*/
	cin.get();
	free(V);
	free(M);
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
	unsigned int* A=new unsigned int[N*N];
	unsigned int* b=new unsigned int[N];
	unsigned int* jm=new unsigned int[N];
	load_matrix<unsigned int>(&N, &A, &b, "../diplomka/mat-int.txt");
	vypsat_mat<unsigned int>(N, N, A, b);
	gauss_jordan_elim_while(N, modul, A, b, jm);


	cout << endl << "-------------------------------" << endl;
	load_matrix(&N, &A, &b, "../diplomka/mat-int.txt");
	vypsat_mat(N, N, A, b);
	gauss_jordan_elim_part(N, modul, A, b, jm);

#ifdef _DEBUG
	vypsat_vys<unsigned int>(N, b, jm);
	cin.get();
#else
	delete[] A;
	delete[] b;
	delete[] jm;
#endif
}