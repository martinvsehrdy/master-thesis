#include "maticeR.h"
#include "matice.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include <cfloat>

using namespace std;

template<class T>
maticeR<T>::maticeR()
{
	N=0;
	this->pointer=NULL;
	this->prava_strana=NULL;
}
template<class T>
maticeR<T>::maticeR(int size)
{
	this->N=size;
	pointer=new double[size*size];
	prava_strana=new double[size];
}
template<class T>
maticeR<T>::maticeR(int size, double value)
{
	this->N=size;
	pointer=new double[size*size];
	prava_strana=new double[size];

	for(int i=0;i<N*N;i++)
	{
		pointer[i]=value;
	}
	for(int i=0;i<N;i++)
	{
		prava_strana[i]=value;
	}
}

template<class T>
maticeR<T>::~maticeR(void)
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
template<class T>
int maticeR<T>::get_size()
{
	return N;
}
template<class T>
int maticeR<T>::set_cell(int X, int Y, double value)
{
	if(pointer != NULL && 0<=X && X<N && 0<=Y && Y<N)
	{
		pointer[X*N+Y]=value;
		return 0;
	}
	return 1;
}
template<class T>
double maticeR<T>::get_cell(int X, int Y)
{
	if(pointer != NULL && 0<=X && X<N && 0<=Y && Y<N)
	{
		return pointer[X*N+Y];
	}
	return 0;
}

template<class T>
int maticeR<T>::set_cell1(int ind, double value)
{
	if(prava_strana != NULL && 0<=ind && ind<N)
	{
		prava_strana[ind]=value;
		return 0;
	}
	return 1;
}
template<class T>
double maticeR<T>::get_cell1(int ind)
{
	if(prava_strana != NULL && 0<=ind && ind<N)
	{
		return prava_strana[ind];
	}
	return 1;
}
template<class T>
void maticeR<T>::fill_random()
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

template<class T>
void maticeR<T>::save_matrix(char* filename)
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

template<class T>
int maticeR<T>::load_matrix(char* filename)
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

template<class T>
void maticeR<T>::execute()
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



void maticeR<T>::vypsat()
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

