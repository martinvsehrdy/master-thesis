#include "StdAfx.h"
#include "maticeM.h"


template<class T>
maticeM::maticeM(void)
{

}
template<class T>
maticeM<T>::maticeM(int size)
{

}
template<class T>
maticeM(int size, T value)
{

}
template<class T>
maticeM::~maticeM(void)
{

}
template<class T>	
int maticeM<T>::get_size()
{
}
int set_cell(int X, int Y, T value);
T get_cell(int X, int Y);	// SLOUPEC, RADEK
void fill_random(void);
void fill_hilbert(void);
void save_matrix(char* filename);
int load_matrix(char* filename);
void execute();
void vypsat();

template<class T>
void maticeR<T>::execute()
{
	// frexp, ldexp
	int min_exponent=get_cell1(0);		// exponent nejmensiho cisla
	for(int i=1;i<N*N;i++)
	{
		int exponent;
		frexp(get_cell1(i), &exponent);
		if(min_exponent>exponent) min_exponent=exponent;
	}



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