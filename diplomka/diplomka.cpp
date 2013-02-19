// diplomka.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <string>
#include <iostream>
#include <sstream>
#include <iomanip>
#include "matice.h"
#include <list>

using namespace std;

double* A=NULL;
double* b=NULL;
double* x=NULL;

int main(int argc, char** argv)
// argv[0]
{
	/*int modulo=3;
	for(int a=-10;a<10;a++)
	{
		int mod=a % modulo;
		if(mod<0) mod+=modulo;
		cout << a << " -> " << mod << " -> " << m_inv(modulo, mod) << endl;	// TODO: moduluje pouze kladne
	}
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
