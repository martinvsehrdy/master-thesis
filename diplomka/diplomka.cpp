// diplomka.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <string>
#include <iostream>
#include <iomanip>
#include "matice.h"
#include <list>

using namespace std;

int main(int argc, char* argv[])
{

	matice<double> mat(4);
	mat.load_matrix("mat1.txt");
	mat.vypsat();
	mat.do_modular();
	cout << "VYSLEDEK:" << endl;
	mat.vypsat();
	
	cin.get();
	return 0;
}

