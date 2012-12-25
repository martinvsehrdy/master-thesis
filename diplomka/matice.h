#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <iostream>
#include <fstream>

using namespace std; 

template<class T>
class matice
{
private:
	T* pointer;
	int N;
	T* prava_strana;
public:
	matice(void)
	{
		N=0;
		this->pointer=NULL;
		this->prava_strana=NULL;
	}
	matice(int size)
	{
		this->N=size;
		pointer=new T[size*size];
		prava_strana=new T[size];
	}
	matice(int size, T value)
	{
		this->N=size;
		pointer=new T[size*size];
		prava_strana=new T[size];

		for(int i=0;i<N*N;i++)
		{
			pointer[i]=value;
		}
		for(int i=0;i<N;i++)
		{
			prava_strana[i]=value;
		}
	}
	~matice(void)
	{
		if(pointer != NULL)
		{
			delete pointer;
		}
		if(prava_strana != NULL)
		{
			delete prava_strana;
		}
	}
	
	int get_size()
	{
		return N;
	}
	int set_cell(int X, int Y, T value)
	{
		if(pointer != NULL && 0<=X && X<N && 0<=Y && Y<N)
		{
			pointer[X*N+Y]=value;
			return 0;
		}
		return 1;
	}
	T get_cell(int X, int Y)	// SLOUPEC, RADEK
	{
		if(pointer != NULL && 0<=X && X<N && 0<=Y && Y<N)
		{
			return pointer[X*N+Y];
		}
		return 0;
	}
	int set_cell1(int ind, T value)
	{
		if(prava_strana != NULL && 0<=ind && ind<N)
		{
			prava_strana[ind]=value;
			return 0;
		}
		return 1;
	}
	T get_cell1(int ind)
	{
		if(prava_strana != NULL && 0<=ind && ind<N)
		{
			return prava_strana[ind];
		}
		return 1;
	}
	void fill_random(void)
	{
		srand( time(NULL) );
		for(int i=0;i<N*N;i++)
		{
			pointer[i]=(rand() % 1000 + 1)/10.0;
		}
		for(int i=0;i<N;i++)
		{
			prava_strana[i]=(rand() % 1000 + 1)/10.0;
		}
	}
	void save_matrix(char* filename)
	{
		FILE* file;
		file=fopen(filename, "w");
		fprintf(file, "%d\n", N);
		for(int y=0;y<N;y++)
		{
			for(int x=0;x<N;x++)
			{
				fprintf(file, "%7.2f", pointer[x*N+y]);
			}
			fprintf(file, " | %7.2f\n", prava_strana[y]);
		}
		fclose(file);
	}
	int load_matrix(char* filename)
	{
		//FILE* file;
		fstream file;
		file.open(filename, fstream::in);
		if(!file.is_open()) return 1;
		int newN=0;
		file >> newN;
		if(newN<=0) return 2;	// nekladna cisla nechci
	
		if(N!=newN)	// stara a nova matice maji ruznou velikost => musim zmenit velikost
		{
			N=newN;
			if(pointer!=NULL) delete pointer;
			pointer=new double[N*N];
			if(prava_strana!=NULL) delete prava_strana;
			prava_strana=new double[N];
		}
		double a;
		for(int y=0;y<N;y++)
		{
			for(int x=0;x<N;x++)
			{
				if(!file.eof()) file >> a;
				else a=0.0;
				pointer[x*N+y]=a;
			}
			if(!file.eof()) file >> a;
			else a=0.0;
			prava_strana[y]=a;
		}

		file.close();
		return 0;
	}
	void do_gauss(void)
	{
		for(int ipivot=0;ipivot<N;ipivot++)
		{
			if(get_cell(ipivot, ipivot)==0)
			{
				// v 'ipivot'-tem radku na diagonále je nula => vymena s jinym radkem
				int novy_pivot=ipivot;
				do{
					novy_pivot++;
				}while(get_cell(ipivot, novy_pivot)==0 && novy_pivot<N);

				if(get_cell(ipivot, novy_pivot)==0)		// nasel jsem radek s nenulovym prvkem ve sloupci ipivot
				{
					// vymena radku ipivot a novy_pivot
					double pom;
					for(int iX=0;iX<N;iX++)
					{
						pom=get_cell(iX, ipivot);
						set_cell(iX, ipivot, get_cell(iX, novy_pivot));
						set_cell(iX, novy_pivot, pom);
					}
					pom=prava_strana[ipivot];
					prava_strana[ipivot]=prava_strana[novy_pivot];
					prava_strana[novy_pivot]=pom;
				}else
				{
					// matice nema v 'ipivot'-tem sloupci nenulovy prvek => je singularni
				}
			}
			for(int iY=0;iY<N;iY++)	// prochazi jednotlive radky
			{
				if(iY==ipivot) continue;
				double multipl=get_cell(ipivot, iY)/get_cell(ipivot, ipivot);
				for(int iX=0;iX<N;iX++)	// prochazi cisla v i1-tem radku
				{
					set_cell(iX, iY, get_cell(iX, iY)-multipl*get_cell(iX, ipivot));
				}
				prava_strana[iY]=prava_strana[iY]-multipl*prava_strana[ipivot];
			}
		}
		// znormovani na jednicku
		for(int ipivot=0;ipivot<N;ipivot++)
		{
			prava_strana[ipivot]/=get_cell(ipivot, ipivot);
			set_cell(ipivot, ipivot, 1.0);
		}
	}
	void do_modular(void)
	{
		if(pointer==NULL || prava_strana==NULL) return;
		// frexp, ldexp
		int min_exponent;		// exponent nejmensiho cisla
		T max_a=get_cell1(0);
		T max_y=prava_strana[0];
		frexp(max_a, &min_exponent);
		for(int i=1;i<N*N;i++)
		{
			int exponent;
			frexp(get_cell1(i), &exponent);
			if(min_exponent>exponent)
			{
				min_exponent=exponent;
			}
			if(max_a<get_cell1(i))
			{
				max_a=get_cell1(i);
			}
			if(i<N)
			{
				if(max_y<prava_strana[i])
				{
					max_y=prava_strana[i];
				}
			}
		}
		T M1=pow(N, N/2)*pow(max_a, N);
		T M2=N*pow(N-1, (N-1)/2)*pow(max_a, N-1)*max_y;

		long M=(long)(2*max(M1, M2));
		// TODO: funkce gcd(M,D), zvolit m_1, m_2... m_r


		double D_hadamard=1.0;
		for(int iX=0;iX<N;iX++)
		{
			double souc=0.0;
			for(int iY=0;iY<N;iY++)
			{
				souc+=get_cell(iX, iY)*get_cell(iX, iY);
			}
			D_hadamard*=souc;
		}
		D_hadamard=sqrt(D_hadamard);


	}
	void vypsat()
	{
		printf("\n");
		for(int y=0;y<N;y++)
		{
			for(int x=0;x<N;x++)
			{
				printf("%7.2f", pointer[x*N+y]);
			}
			printf(" | %7.2f\n", prava_strana[y]);
		}
	}
};

