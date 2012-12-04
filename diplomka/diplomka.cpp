// diplomka.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "matice.h"
#include <string>
#include <iostream>
#include <iomanip>

using namespace std;

int _tmain(int argc, _TCHAR* argv[])
{
	matice mat(4);
	mat.load_matrix("mat1.txt");
	//mat.save_matrix("mat1.txt");
	mat.vypsat();
	mat.do_gauss();
	cout << "VYSLEDEK:" << endl;
	mat.vypsat();

	getchar();getchar();
	return 0;
}

