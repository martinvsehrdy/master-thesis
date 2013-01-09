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

	int size=0;
	try
	{
		if(argc>1)
		{
			size=atoi(argv[1]);
		}
	}catch(...)
	{
		printf("nekde se stala chyba :-(\nspravne volani programu: %s ", argv[0]);
		return 0;
	}


	matice<double> mat(size);
	
	mat.fill_hilbert();
	mat.load_matrix("mat1.txt");
	
	mat.vypsat();
	mat.do_modular();
	//mat.do_gauss();
	cout << "VYSLEDEK:" << endl;
	mat.vypsat();
	
	cin.get();
	return 0;//*/
}
