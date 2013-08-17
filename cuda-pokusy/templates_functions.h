#ifndef _TEMPLATES_FUNCTIONS_H_
#define _TEMPLATES_FUNCTIONS_H_
#include <iostream>
#include <fstream>
#include "kernels_cpu.h"


template<class TYPE>
int load_matrix(int* N, TYPE** matice, TYPE** prava_strana, char* filename)
{
	TYPE* matice1=*matice;
	TYPE* prava_strana1=*prava_strana;

	fstream file;
	file.open(filename, fstream::in);
	if(!file.is_open()) return 1;
	int newN=0;
	file >> newN;
	if(newN<=0) return 2;	// nekladna cisla nechci
	
	if((*N)!=newN)	// stara a nova matice maji ruznou velikost => musim zmenit velikost
	{
		(*N)=newN;
		if(matice1!=NULL) delete matice1;
		matice1=NULL;
		if(prava_strana1!=NULL) delete prava_strana1;
		prava_strana1=NULL;
	}
	if(matice1==NULL) matice1=new TYPE[(*N)*(*N)];
	if(prava_strana1==NULL) prava_strana1=new TYPE[*N];
	TYPE a;
	for(int y=0;y<(*N);y++)
	{
		for(int x=0;x<(*N);x++)
		{
			if(!file.eof()) file >> a;
			else a=(TYPE)0.0;
			matice1[get_index(x, y, (*N))]=a;
		}
		if(!file.eof()) file >> a;
		else a=(TYPE)0.0;
		prava_strana1[y]=a;
	}
	file.close();
	*matice=matice1;
	*prava_strana=prava_strana1;
	return 0;
}

template<class TYPE>
void vypsat_mat(int nx, int ny, TYPE* matice, TYPE* prava_strana)
{
	cout << endl;
	for(int y=0;y<min(ny,12);y++)
	{
		int x;
		for(x=0;x<min(nx,10);x++)
		{
			cout.precision(5);
			cout << matice[get_index(x, y, max(nx, ny))] << "\t";
		}
		if(x<nx-1) cout << "...";
		cout << "| ";
		if(prava_strana!=NULL) cout << prava_strana[y];
		cout << endl;
	}
}

template<class TYPE>
void vypsat_matlab(int nx, int ny, TYPE* matice, TYPE* prava_strana)
{
	cout << endl << "A=[";
	for(int y=0;y<ny;y++)
	{
		int x;
		for(x=0;x<nx;x++)
		{
			cout << matice[get_index(x, y, max(nx, ny))];
			if(x<nx-1) cout << ",";
		}
		if(y<ny-1) cout << ";";
	}
	cout << "];" << endl << "b=[";
	for(int y=0;y<ny;y++)
	{
		cout << prava_strana[y];
		if(y<ny-1) cout << ";";
	}
	cout << "];" << endl;
}

template<class TYPE>
void vypsat_vys(int N, TYPE* citatel, TYPE* jmenovatel)
{
	cout << endl;
	//cout.precision(7);
	int i;
	for(i=0;i<min(N,30);i++)
	{
		cout << citatel[i] << "/" << jmenovatel[i] << "\t";
	}
	if(i<N-1) cout << "...";
}


#endif /*  _TEMPLATES_FUNCTIONS_H_ */
