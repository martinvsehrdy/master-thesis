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
	/*mpz_class aBigPO2;

    aBigPO2 = 1073741824; //2^30
    aBigPO2*=aBigPO2; //2^60
    aBigPO2*=aBigPO2; //2^120
    aBigPO2*=aBigPO2; //2^240
    aBigPO2*=aBigPO2; //2^480
    aBigPO2*=aBigPO2; //2^960
    aBigPO2*=aBigPO2; //2^1920

    cout << aBigPO2 << endl;
	//*/

	bool yes=false;
	unsigned short s=65530;
	if(s+30>s)
		yes=true;

	float x=1.0/(float)(1 << 20);
	
	int e;
	float a=frexp(x, &e);
	int v=(int)(a*(1 << numeric_limits<float>::digits));
	e-=numeric_limits<float>::digits;
	yes=false;
	if(x==ldexp((float)v, e))
		yes=true;


	
	matice<float> mat(4);
	mat.load_matrix("mat2.txt");
	mat.vypsat();
	mat.do_modular();
	cout << "VYSLEDEK:" << endl;
	mat.vypsat();
	//*/
	cin.get();
	return 0;
}

