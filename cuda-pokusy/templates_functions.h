#ifndef _TEMPLATES_FUNCTIONS_H_
#define _TEMPLATES_FUNCTIONS_H_
#include <iostream>
#include <fstream>
#include <stdio.h>
#include "kernels_cpu.h"

template<class TYPE>
void hilbert_matrix(int N, TYPE* matice, TYPE* prava_strana)
{
	TYPE a;
	double zaklad = pow(10.0,1+ceil(log10((double)N)));
	for(int y=0;y<N;y++)
	{
		for(int x=0;x<N;x++)
		{
			a = (TYPE)(zaklad/((double)(x+y+2)));
			matice[get_index(x, y, N)]=a;
		}
		prava_strana[y]=y+1;
	}
}
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
int save_matrix(int N, TYPE* matice, TYPE* prava_strana, char* filename)
{
	/*fstream file;
	file.open(filename, fstream::out);
	if(!file.is_open()) return 1;*/
	FILE* f=fopen(filename, "w");
	 if (f==NULL) return 1;
	//file << N << endl;
	fprintf(f, "%d\n", N);
	
	for(int y=0;y<N;y++)
	{
		int x;
		for(x=0;x<N;x++)
		{
			//file << matice[get_index(x, y, N)] << "\t";
			fprintf(f, "%8u\t", matice[get_index(x, y, N)]);
		}
		if(prava_strana!=NULL)
		{
			//file << "| " << prava_strana[y];
			fprintf(f, "| %u", prava_strana[y]);
		}
		//file << endl;
		fprintf(f, "\n");
	}
	//file.close();
	fclose(f);
	return 0;
}

template<class TYPE>
int save_vys(int N, TYPE* prava_strana, char* filename)
{
	/*fstream file;
	file.open(filename, fstream::out);
	if(!file.is_open()) return 1;*/
	FILE* f=fopen(filename, "w");
	 if (f==NULL) return 1;
	//file << N << endl;
	fprintf(f, "%d\n", N);
	
	for(int y=0;y<N;y++)
	{
		if(prava_strana!=NULL)
		{
			//file << "| " << prava_strana[y];
			fprintf(f, "| %u", prava_strana[y]);
		}
	}
	//file.close();
	fclose(f);
	return 0;
}

template<class TYPE>
void vypsat_mat(int nx, int ny, TYPE* matice, TYPE* prava_strana)
{
	//cout << endl;
	printf("\n");
	for(int y=0;y<min(ny,12);y++)
	{
		int x;
		for(x=0;x<min(nx,8);x++)
		{
			TYPE a=matice[get_index(x, y, nx)];
			//cout.precision(8);
			//cout << a << "\t";
			printf("%6u\t", a);
		}
		if(x<nx-1)
		{
			//cout << "...";
			printf("...");
		}
		//cout << "| ";
		printf("| ");
		if(prava_strana!=NULL)
		{
			//cout << prava_strana[y];
			printf("%u", prava_strana[y]);
		}
		//cout << endl;
		printf("\n");
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
			cout << matice[get_index(x, y, nx)];
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
		if(citatel!=NULL) printf("%u", citatel[i]);
		cout << "/";
		if(jmenovatel!=NULL) printf("%u", jmenovatel[i]);
		printf("\t");
	}
	if(i<N-1) printf("...");
}


#endif /*  _TEMPLATES_FUNCTIONS_H_ */
