// diplomka.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <string>
#include <iostream>
#include <sstream>
#include <iomanip>
#include "cudamodular.cu"
#include "matice.h"
#include <list>

#include <cuda_runtime.h>

using namespace std;

double* A=NULL;
double* b=NULL;
double* x=NULL;

int main(int argc, char** argv)
// argv[0]
{
	N=10;
	int modul=13;
	// inicializace mA, mb
	int* mA=new int[N*N];
	int* mb=new int[N];
	int* jmen=new int[N];
	int* vysc=new int[N];
	int* vysj=new int[N];
	int* cvysc=new int[N];
	int* cvysj=new int[N];
	for(int y=0;y<N;y++)
	{
		for(int x=0;x<N;x++)
		{
			mA[get_index(x,y)]=rand() % modul;
		}
		mb[y]=rand() % modul;
		jmen[y]=modul;
	}
	vypsat1<int>(N, mA, mb);

	// nahrani mA, mb na gpu
	int* cuda_mA=NULL;
	cudaMalloc((void**)&cuda_mA, N*N*sizeof(int));
	cudaMemcpy(cuda_mA, mA, N*N*sizeof(int), cudaMemcpyHostToDevice);
	int* cuda_mb=NULL;
	cudaMalloc((void**)&cuda_mb, N*sizeof(int));
	cudaErr=cudaMemcpy(cuda_mb, mb, N*sizeof(int), cudaMemcpyHostToDevice);
	int* cuda_jmen=NULL;
	cudaMalloc((void**)&cuda_jmen, N*sizeof(int));
	// TODO: pridat nahrani do sdileny pameti na gpu
	// vypocet na gpu
	//cuda_gauss_jordan_elim(N, modul, cuda_mA, cuda_mb, cuda_jmen);
	// vypocet na cpu
	gauss_jordan_elim(N, modul, mA, mb, jmen);
	for(int i=0;i<N;i++)
	{
		vysc[i]=mb[i];
		vysj[i]=jmen[i];
	}
	cudaMemcpy(cvysc, cuda_mb, N*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(cvysj, cuda_jmen, N*sizeof(int), cudaMemcpyDeviceToHost);


	// vypsat vysledky
	cout << endl << "vysledky z CPU" << endl;
	for(int i=0;i<N;i++)
	{
		cout << vysc << "/" << vysj << "\t";
	}
	cout << endl << "vysledky z GPU" << endl;
	for(int i=0;i<N;i++)
	{
		cout << cvysc << "/" << cvysj << "\t";
	}

	cudaFree(cuda_mA);
	cudaFree(cuda_mb);
	cudaFree(cuda_jmen);
	delete[] mA;
	delete[] mb;
	delete[] jmen;
	delete[] vysc;
	delete[] vysj;
	delete[] cvysc;
	delete[] cvysj;
	cin.get();
	return 0;
	/*/

	int metoda=0;
	try
	{
		if(argc>2)
		{
			metoda=atoi(argv[1]);
			N=atoi(argv[2]);
		}else throw;
	}catch(...)
	{
		printf("nekde se stala chyba :-(\nspravne volani programu: %s ", argv[0]);
		return 0;
	}


	A = new double[N*N];
	b = new double[N];
	
	fill_hilbert(N, A, b);
	vypsat(N, A, b);
	
	switch(metoda)
	{
	case 1: do_gauss(N, A, b);
		break;
	case 2: do_modular(N, A, b);
		break;
	}
	cout << "VYSLEDEK:" << endl;
	vypsat(N, A, b);
	
	delete A;
	delete b;
	cin.get();
	return 0;//*/
}
