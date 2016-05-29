#include "stdafx.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <list>
#include <sstream>
#include "kernels_cpu.h"
#include "templates_functions.h"
#include "kernels.h"
#include "time_measure.h"
#include "common.h"
#include <Windows.h>

using namespace std;

#define POC_OPAKOVANI 1
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

// spocita na CPU a na GPU (lib. metoda) a porovna vysledky
void main1(int argc, char** argv, int N, unsigned int settings)
{
	stringstream ss;
	if(argc>2)
	{
#ifndef _DEBUG
		N=atoi(argv[1]);
#endif
	}
	unsigned int modul=0x10000003; //(~(unsigned int)0);
	modul = 0x1003;	// 4099 je prvocislo
	cout << "Modul = " << modul << endl;

	unsigned int* M=new unsigned int[N*N];
	unsigned int* P=new unsigned int[N];
	unsigned int* Pfor=new unsigned int[N];

	ss.str("");
	ss.clear();
	ss << "outmat-for";
	ss << N;
	fstream file;
	file.open(ss.str().c_str(), fstream::in);
	bool ok=false;
	if(file.is_open())
	{
		int i;
		file >> i;
		if(i==N)
		{
			i=0;
			while(!file.eof() && i<N)
			{
				file >> Pfor[i];
				i++;
			}
			if(i==N) ok=true;
		}
	}
	if(!ok)
	{
		if(settings & ZPUSOB_HILBERT_MAT)
		{
			hilbert_matrix<unsigned int>(N, M, Pfor);
		}else
		{
			tridiag_matrix<unsigned int>(N, M, Pfor);
		}
		gauss_jordan_elim_for(N, modul, M, Pfor, settings | ZPUSOB_S_DELENIM);
		save_vys<unsigned int>(N, Pfor, (char*)ss.str().c_str());
	}

	if(settings & ZPUSOB_HILBERT_MAT)
	{
		hilbert_matrix<unsigned int>(N, M, P);
	}else
	{
		tridiag_matrix<unsigned int>(N, M, P);
	}
	vypsat_mat(N, N, M, P);
	init_gpu_compute();
	cuda_GJE_radky(N, modul, M, P, settings);
	/*unsigned int* S=new unsigned int[N*N+N];
	copy_podmatice(N, 0, 0, N+1, N, S, M, P, COPY_MAT_B_GLOB_TO_A_SH);
	cuda_GJE_global(N, modul, S, settings);
	copy_podmatice(N, 0, 0, N+1, N, S, M, P, COPY_MAT_A_SH_TO_B_GLOB);
	free(S);//*/

	ss.str("");
	ss.clear();
	ss << "outmat-GJE";
	//ss << N;
	save_vys<unsigned int>(N, P, (char*)ss.str().c_str());
	vypsat_mat<unsigned int>(N, N, M, P);

	bool v=true;
	for(int y=0;y<N;y++)
	{
		if( Pfor[y]!=P[y] )
	{
			v=false;
			break;
	}
	}
	if( v ) cout << endl << "SPRAVNE" << endl;
	else cout << endl << "SPATNE" << endl;

	delete M;
	delete P;
	delete Pfor;
#ifdef _DEBUG
	cin.get();
#endif
}
// zmeri cas vypoctu jednoho kroku = nulovani jednoho sloupce, ipivot
void main2(int argc, char** argv)
{
	int N=200;
	unsigned int zpusob=0;
	unsigned int modul=0x10000003; //(~(unsigned int)0);
	modul = 0x40000003;	// nejmensi prvocislo v [2^30+1;2^31-1]
	modul = 0x7FFFFFED;	// nejvetsi prvocislo v [2^30+1;2^31-1]
	if(argc>2)
	{

		N=atoi(argv[1]);
		zpusob=atoi(argv[2]);
	}else
	{
		cout << "#Program spustte ve tvaru:" << argv[0] << " <N> <zpusob zpracovani> [-DG]" << endl;
		cout << "#zpusob zpracovani: 8 - find_inverse" << endl;
		cout << "#                   9 - GJE_radky_kernel, ipivot=0" << endl;
		cout << "#                  10 - GJE_radky_kernel, ipivot=1/4 * N" << endl;
		cout << "#                  11 - GJE_radky_kernel, ipivot=1/2 * N" << endl;
		cout << "#                  12 - GJE_radky_kernel, ipivot=3/4 * N" << endl;
		cout << "#                  13 - GJE_radky_kernel, ipivot= N-1" << endl;
		cout << "#    G - vlakno zpracovava sloupec matice" << endl;
		cout << "#    D - elementarni uprava bude v realnych cislech s CUDA funkcema" << endl;
		cout << "#Vystup: <N> <zpusob> <shared_size> <cas vypoctu> <poc_bloku> <poc_vlaken> <gd>" << endl;
		return;
	}
	for(int i=1;i<argc;i++)
	{
		if(argv[i][0]=='-')
		{
			int j=1;
			while(argv[i][j]!=0)
			{
				switch(argv[i][j])
				{
				case 'g':
				case 'G': zpusob |= ZPUSOB_GLOB_PRISTUP;
					break;
				case 'd':
				case 'D': zpusob |= ZPUSOB_CUDA_UPRAVA;
					break;
				}
				j++;
			}
		}
	}
	init_gpu_compute();

	cout << N << "\t" << (zpusob & 0xFF) << "\t-\t";
	modul = 0x40000003;
	float tt=0;
	int pocet=0;
	test_GJE_radky(N, zpusob);
	
		tt+=cuda_get_measured_time();

	int poc_vlaken;
	int poc_bloku;
	get_pocty(&poc_bloku, &poc_vlaken, NULL, NULL);
	cout << tt << "\t";
	cout << poc_bloku << "\t" << poc_vlaken << "\t";
	if(zpusob & ZPUSOB_GLOB_PRISTUP) cout << "G";
	else cout << "g";
	if(zpusob & ZPUSOB_CUDA_UPRAVA)  cout << "D";
	else cout << "d";
	cout << endl;
	
	
#ifdef _DEBUG
	cin.get();
#else
#endif
}
int main(int argc, char** argv)
// argv[0] <N> <modul>
{
	
	//main2(argc, argv);
	/*main1(argc, argv, 100, ZPUSOB_S_DELENIM | ZPUSOB_GLOB_PRISTUP | ZPUSOB_CUDA_UPRAVA | ZPUSOB_HILBERT_MAT);
	
	 /*/
	////////////////////////////////////////////////////////
	int N=4;
	unsigned int modul=0x7FFFFFED; //(~(unsigned int)0);
	int zpusob=0;
	if(argc>2)
	{
		N=atoi(argv[1]);
		zpusob=strtol(argv[2], NULL, 2);
	}else
	{
		
		print_gpus_info();
		// TODO: (ne)vyuzivat sdilenou pamet; modulo v elementarni uprave;
		cout << "#Program spustte ve tvaru:" << argv[0] << " <N> <zpusob zpracovani>" << endl;
		cout << "#zpusob zpracovani: 0. bit \tfor/while(0) while/for(1)" << endl;
		cout << "#(pocitano zprava)  1. bit \tbez deleni(0) s delenim(1)" << endl;
		cout << "#                   2.3.bit \t1(00) 32(01) 128(10) vlaken" << endl;
		cout << "#                   4.bit \tGPU(0) CPU(1)" << endl;
		cout << "#                   5.bit \tmatice v sdilene(0), globalni(1) pameti" << endl;
		cout << "#                   6.bit \tmetoda: podmatice(0), radky(1)" << endl;
		cout << "#    G - vlakno zpracovava sloupec matice" << endl;
		cout << "#    D - elementarni uprava bude v realnych cislech s CUDA funkcema" << endl;
		cout << "#    M - hilbertova matice, jinak bude tridiagonalni" << endl;
		cout << "#    S - Podmatice: pomer stran podmatice; Radkova: pocet bloku/SM (1)" << endl;
		cout << "#Vystup: <velikost N> <na GPU [ms]>\t<z GPU [ms]>\tprumer\tnejrychlejsi\t1.quartal\tmedian\t3.quartal\tnejpomalejsi\t<celkem [ms]>\t <poc_bloku> <poc_vlaken> <Sx> <Sy> <gdm>" << endl;
		return 0;
	}
	for(int i=1;i<argc;i++)
	{
		if(argv[i][0]=='-')
		{
			int j=1;
			while(argv[i][j]!=0)
			{
				switch(argv[i][j])
				{
				case 'g':
				case 'G': zpusob |= ZPUSOB_GLOB_PRISTUP;
					break;
				case 'd':
				case 'D': zpusob |= ZPUSOB_CUDA_UPRAVA;
					break;
				case 'm':
				case 'M': zpusob |= ZPUSOB_HILBERT_MAT;
					break;
				case 'a':
				case 'A': zpusob |= ZPUSOB_ASTR;
					break;
				case 'S':
				case 's':
					if('0'<=argv[i][j+1] && argv[i][j+1]<='9')
					{
						zpusob |= (atoi(&argv[i][j+1]))<<16;
					}
				}
				j++;
			}
		}
	}
	// TODO: vypocet
	unsigned int* A=new unsigned int[N*N];
	unsigned int* b=new unsigned int[N];

#ifndef _DEBUG
	list<float> times;
	times.clear();
	float sum=0.0;
	for(int opakovani=0;opakovani<POC_OPAKOVANI;opakovani++)
	{
#endif
		if(zpusob & ZPUSOB_HILBERT_MAT)
		{
			hilbert_matrix<unsigned int>(N, A, b);
		}else
		{
			tridiag_matrix<unsigned int>(N, A, b);
		}
		if(zpusob & ZPUSOB_ASTR)
		{
			delete A;
			delete b;
			fstream fileA, fileb;
			fileA.open("s000001.txt", fstream::in);
			fileb.open("r000001.txt", fstream::in);
			fileA >> N;
			A=new unsigned int[N*N];
			b=new unsigned int[N];
			double* pom=new double[N+1];
			double a;
			double min=0;
			for(int y=0;y<N;y++)
			{
				for(int x=0;x<N;x++)
				{
					if(!fileA.eof()) fileA >> a;
					else a=0.0;
					if( (min==0 || abs(a)<abs(min)) && a!=0) min=abs(a);
					pom[x]=a;
				}
				if(!fileb.eof()) fileb >> a;
				else a=0.0;
				if( (min==0 || abs(a)<abs(min)) && a!=0) min=abs(a);
				pom[N]=a;

				for(int x=0;x<N;x++)
				{
					a = pom[x]/min;
					a = fmod(a, (double)modul);
					if( a>=0 ) A[get_index(x, y, N)]=(unsigned int)a;
					else A[get_index(x, y, N)]=modul - (unsigned int)a;
				}
				a = pom[N]/min;
				a = fmod(a, (double)modul);
				if( a>=0 ) b[y]=(unsigned int)a;
				else b[y]=modul - (unsigned int)a;
				if(y%1000 == 0) cout << ".";
			}
			cout << endl;
			delete pom;
		}
	
#ifdef _DEBUG
	vypsat_mat<unsigned int>(N, N, A, b);
#endif
		float tt=0;
		if(zpusob & ZPUSOB_CPU)
		{
			gauss_jordan_elim_for(N, modul, A, b, zpusob);
			tt = get_measured_time();
#ifndef _DEBUG
			if(opakovani>0) break;
#endif
		}else
		{
			init_gpu_compute();
			
			if( zpusob & ZPUSOB_RADKY )
			{
				cuda_GJE_radky(N, modul, A, b, zpusob);
			}else
			{
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
			}
			tt = cuda_get_measured_time();
		}
#ifndef _DEBUG
		times.push_back(tt);
		sum += tt;
		//if( !(zpusob & ZPUSOB_CPU) ) cudaDeviceReset();
		Sleep(100);
	}
	float q1, q2, q3, prumer;
	statistic(times, &q1, &q2, &q3, &prumer);
	times.sort();
	//<velikost N> <na GPU [ms]>\t<z GPU [ms]>\tprumer\tnejrychlejsi\t1.quartal\tmedian\t3.quartal\tnejpomalejsi\t<celkem [ms]>
	cout << N << "\t?\t?\t" << prumer << "\t" << times.front() << "\t" << q1 << "\t" << q2 << "\t" << q3 << "\t" << times.back() << "\t" << sum << "\t";
	int poc_vlaken = 0;
	int poc_bloku = 0;
	int poc_Sx = 0;
	int poc_Sy = 0;
	get_pocty(&poc_bloku, &poc_vlaken, &poc_Sx, &poc_Sy);
	cout << poc_bloku << "\t" << poc_vlaken << "\t" << poc_Sx << "\t" << poc_Sy << "\t";

	if(zpusob & ZPUSOB_GLOB_PRISTUP) cout << "G";
	else cout << "g";
	if(zpusob & ZPUSOB_CUDA_UPRAVA)  cout << "D";
	else cout << "d";
	if(zpusob & ZPUSOB_HILBERT_MAT)  cout << "M";
	else cout << "m";
	int max_vys=0;
	for(int i=0;i<N;i++) if(b[i]>max_vys) max_vys=b[i];
	cout << "\t" << max_vys;
	cout << endl;
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
