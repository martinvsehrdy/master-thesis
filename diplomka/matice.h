#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <iostream>
#include <fstream>

using namespace std;

#pragma warning(disable: 4800; disable: 4244)
#include <mpirxx.h>
#pragma warning(default: 4244; default: 4800)

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
			pointer=new T[N*N];
			if(prava_strana!=NULL) delete prava_strana;
			prava_strana=new T[N];
		}
		T a;
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
	int hadamard()
	{
		
	}
	void do_modular(void)
	{
		if(pointer==NULL || prava_strana==NULL) return;
		// prenasobeni radku matice, aby za desetinou carkou byly jen nuly
		for(int y=0;y<N;y++)
		{
			int min_exponent;
			ifrexp(prava_strana[y], &min_exponent);
			int exponent;
			for(int x=0;x<N;x++)
			{
				ifrexp(get_cell(x,y), &exponent);
				if(min_exponent>exponent)
				{
					min_exponent=exponent;
				}
			}
			T pom=frexp(prava_strana[y], &exponent);
			prava_strana[y]=ldexp(pom, exponent - min_exponent);
			for(int x=0;x<N;x++)
			{
				pom=frexp(get_cell(x,y), &exponent);
				set_cell(x,y, ldexp(pom, exponent - min_exponent));
			}
		}

		// spocitat 'M'
		T max_a=get_cell(0,0);
		T max_y=prava_strana[0];
		for(int y=0;y<N;y++)
		{
			for(int x=0;x<N;x++)
			{
				if(max_a<get_cell(x,y))
				{
					max_a=get_cell(x,y);
				}
			}
			if(max_y<prava_strana[y])
			{
				max_y=prava_strana[y];
			}
		}
		T M1=pow((T)N, N/2)*pow(max_a, N);
		T M2=N*pow((T)(N-1), (N-1)/2)*pow(max_a, N-1)*max_y;
		T M=2*max(M1, M2);

		int poc_sieve;
		frexp(M, &poc_sieve);
		poc_sieve=poc_sieve*poc_sieve;
		// vygenerovat pole prvocisel
		int* sieve = new int[poc_sieve];
		for(int i = 0; i <= poc_sieve; i ++)
		{
			sieve[i] = true;
		}
		//true == je prvocislo
		//false == je slozene
		sieve[0] = sieve[1] = false; //nula a jedna nejsou prvocisla
		for(int i = 2; i <= sqrt((double)poc_sieve); i++)
		{
			//if( i%2==0 ) sieve[i]=false;	// suda cisla nejsou prvocisla a 2 je nevyhovujici
			if(sieve[i] == false) continue;
			for(int j = 2*i; j <= poc_sieve; j+=i)
			{ //samotne citani
				sieve[j] = false;	// nemuze byt z definice prvocislem (je nasobkem jineho cisla)
			}
		}
		
		// hadamard
		mpz_class D_hadamard1=1;
		for(int iX=0;iX<N;iX++)
		{
			mpz_class souc=0;
			for(int iY=0;iY<N;iY++)
			{
				souc+=((long)get_cell(iX, iY))*((long)get_cell(iX, iY));
			}
			double dh=D_hadamard1.get_d();
			double ds=souc.get_d();
			// kontrola jestli by slo ze 'souc' vytknout nejake x^2, ktere vyleze pred odmocninu, a ktere deli 'D_hadamard' => nesmi byt m_i
			mpz_class delitel;
			mpz_gcd(delitel.get_mpz_t(), souc.get_mpz_t(), D_hadamard1.get_mpz_t());
			int idelitel=(int)delitel.get_d();
			//sieve[idelitel]=false;	// nevyhovujici prvocislo/slozene cislo
			//D_hadamard1/=delitel;
			//souc/=delitel;
			D_hadamard1*=souc;
		}
		
		mpz_class D_hadamard;
		mpz_sqrt(D_hadamard.get_mpz_t(), D_hadamard1.get_mpz_t());

		cout << "\nprvocisla: ";
		for(int i = 0; i <= poc_sieve; i ++)
		{
			if(sieve[i]) cout << i << ", ";
		}
		cout << "\nD = " << ((long)D_hadamard.get_d()) << "\n";

		

		// TODO: funkce gcd(M,D), zvolit m_1, m_2... m_r
		//       nasobit prvocisla, ktera nedeli D_hadamard, do te doby nez budou vetsi nez M
		int M_akt=1;
		cout << endl << M << " = 1 ";
		for(int i=0;i<poc_sieve;i++)
		{
			if(sieve[i])
			{
				mpz_class delitel;
				mpz_class prvoc=i;
				mpz_gcd(delitel.get_mpz_t(), prvoc.get_mpz_t(), D_hadamard.get_mpz_t());
				int idelitel=(int)delitel.get_d();
				
				if(idelitel==1)
				{
					M_akt*=i;
					cout << i << " * ";
				}else
				{
					cout << "(" << i << ") * ";
				}
			
				if( M_akt > M )
				{
					break;
				}
			}
		}
		

		//delete[] sieve;
		//*/
	}
	void vypsat()
	{
		printf("\n");
		for(int y=0;y<N;y++)
		{
			for(int x=0;x<N;x++)
			{
				printf("%7.2f", get_cell(x, y));
			}
			printf(" | %7.2f\n", prava_strana[y]);
		}
	}
};

// frexp, ldexp
template<class T>
int ifrexp(T X, int * Y)
{
	float a=frexp(X, Y);
	int exponent;
	for(exponent=1;exponent<numeric_limits<T>::digits;exponent++)
	{
		a=2.0*a;
		if( a == floor(a) ) break;
	}
	(*Y)-=exponent;
	return (int)a;
}

unsigned int gcd(unsigned int u, unsigned int v)
{
  // simple cases (termination)
  if (u == v)
    return u;
  if (u == 0)
    return v;
  if (v == 0)
    return u;
 
  // look for factors of 2
  if (~u & 1) // u is even
    if (v & 1) // v is odd
      return gcd(u >> 1, v);
    else // both u and v are even
      return gcd(u >> 1, v >> 1) << 1;
  if (~v & 1) // u is odd, v is even
    return gcd(u, v >> 1);
 
  // reduce larger argument
  if (u > v)
    return gcd((u - v) >> 1, v);
  return gcd((v - u) >> 1, u);
}