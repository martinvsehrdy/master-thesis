#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include <list>

using namespace std;

#pragma warning(disable: 4800; disable: 4244)
#include <mpirxx.h>
#pragma warning(default: 4244; default: 4800)

#define T	double

int N;

int ifrexp(T X, int * Y);
int m_inv(int modulo, int cislo);


int get_index(int X, int Y)	// SLOUPEC, RADEK
{
	return X*N+Y;
}
void fill_random(int N, T* matice, T* prava_strana)
{
	srand( time(NULL) );
	for(int i=0;i<N*N;i++)
	{
		matice[i]=(rand() % 1000 + 1)/10.0;
	}
	for(int i=0;i<N;i++)
	{
		prava_strana[i]=(rand() % 1000 + 1)/10.0;
	}
}
void fill_hilbert(int N, T* matice, T* prava_strana)
{
	for(int y=0;y<N;y++)
	{
		for(int x=0;x<N;x++)
		{
			T val=1.0/((T)(1+x+y));
			matice[get_index(x, y)] = val;
		}
		prava_strana[y]=(y+1)*(y+1);
	}
}
void save_matrix(int N, T* matice, T* prava_strana, char* filename)
{
	fstream file;
	file.open(filename, fstream::out);
	file << N << endl;
	for(int y=0;y<N;y++)
	{
		for(int x=0;x<N;x++)
		{
			file << matice[get_index(x, y)] << "\t";
		}
		file << prava_strana[y] << endl;
	}
	file.close();
}
int load_matrix(int N, T* matice, T* prava_strana, char* filename)
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
		if(matice!=NULL) delete matice;
		matice=new T[N*N];
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
			matice[x*N+y]=a;
		}
		if(!file.eof()) file >> a;
		else a=0.0;
		prava_strana[y]=a;
	}

	file.close();
	return 0;
}
void do_gauss(int N, T* matice, T* prava_strana)
{
	for(int ipivot=0;ipivot<N;ipivot++)
	{
		if(matice[get_index(ipivot, ipivot)]==0)
		{
			// v 'ipivot'-tem radku na diagonále je nula => vymena s jinym radkem
			int novy_pivot=ipivot;
			do{
				novy_pivot++;
			}while(matice[get_index(ipivot, novy_pivot)]==0 && novy_pivot<N);

			if(matice[get_index(ipivot, novy_pivot)]==0)		// nasel jsem radek s nenulovym prvkem ve sloupci ipivot
			{
				// vymena radku ipivot a novy_pivot
				double pom;
				for(int iX=0;iX<N;iX++)
				{
					pom=matice[get_index(iX, ipivot)];
					matice[get_index(iX, ipivot)] = matice[get_index(iX, novy_pivot)];
					matice[get_index(iX, novy_pivot)] = pom;
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
			double multipl=matice[get_index(ipivot, iY)]/matice[get_index(ipivot, ipivot)];
			for(int iX=0;iX<N;iX++)	// prochazi cisla v i1-tem radku
			{
				matice[get_index(iX, iY)]-=multipl*matice[get_index(iX, ipivot)];
			}
			prava_strana[iY]-=multipl*prava_strana[ipivot];
		}
	}
	// znormovani na jednicku
	for(int ipivot=0;ipivot<N;ipivot++)
	{
		prava_strana[ipivot]/=matice[get_index(ipivot, ipivot)];
		matice[get_index(ipivot, ipivot)]=1.0;
	}
}
/*
 * vynasobi radky tak, aby kazde cislo melo za desetinou carkou pouze nuly
 */
void vstupni_slr(int N, T* matice, T* prava_strana)
{
	// prenasobeni radku matice, aby za desetinou carkou byly jen nuly
	for(int y=0;y<N;y++)
	{
		int min_exponent;
		ifrexp(prava_strana[y], &min_exponent);
		int exponent;
		for(int x=0;x<N;x++)
		{
			ifrexp(matice[get_index(x,y)], &exponent);
			if(min_exponent>exponent)
			{
				min_exponent=exponent;
			}
		}
		T pom=frexp(prava_strana[y], &exponent);
		prava_strana[y]=ldexp(pom, exponent - min_exponent);
		for(int x=0;x<N;x++)
		{
			pom=frexp(matice[get_index(x,y)], &exponent);
			matice[get_index(x,y)] = ldexp(pom, exponent - min_exponent);
		}
	}
}
/*
 * spocita hadamarduv odhad a modul M
 */
void exec_hadamard(int N,  T* matice, T* prava_strana, mpz_class* hadamard, T* modul_M)
{
	// hadamard
	mpz_class D_hadamard1=1;
	for(int iX=0;iX<N;iX++)
	{
		mpz_class souc=0;
		for(int iY=0;iY<N;iY++)
		{
			souc+=(matice[get_index(iX, iY)])*(matice[get_index(iX, iY)]);
		}
		double dh=D_hadamard1.get_d();
		double ds=souc.get_d();
		// kontrola jestli by slo ze 'souc' vytknout nejake x^2, ktere vyleze pred odmocninu, a ktere deli 'D_hadamard' => nesmi byt m_i
		mpz_class delitel;
		mpz_gcd(delitel.get_mpz_t(), souc.get_mpz_t(), D_hadamard1.get_mpz_t());
		int idelitel=(int)delitel.get_d();
		if(delitel>1) cout << delitel << " ";
		//sieve[idelitel]=false;	// nevyhovujici prvocislo/slozene cislo
		//D_hadamard1/=delitel;
		//souc/=delitel;
		D_hadamard1*=souc;
	}
	double dh=D_hadamard1.get_d();
	*hadamard=sqrt(D_hadamard1);

	// spocitat 'M'
	T max_a=matice[get_index(0,0)];
	T max_y=prava_strana[0];
	for(int y=0;y<N;y++)
	{
		for(int x=0;x<N;x++)
		{
			if(max_a<matice[get_index(x,y)])
			{
				max_a=matice[get_index(x,y)];
			}
		}
		if(max_y<prava_strana[y])
		{
			max_y=prava_strana[y];
		}
	}
	T M1=pow((T)N, N/2)*pow(max_a, N);
	T M2=N*pow((T)(N-1), (N-1)/2)*pow(max_a, N-1)*max_y;
	*modul_M=2*max(M1, M2);
}
/*
 * rozlozi modul M na soucin jednotlivych modulu vzajemne nesoudelnych
 * r - pocet jednotlivych modulu
 * moduly - pole jednotlivych modulu
 * M - vstupni modul M
 */
void exec_moduly(int N,  T* matice, T* prava_strana, mpz_class hadamard, T modul_M, int* r, int** moduly)
{
	int poc_sieve;
	frexp(modul_M, &poc_sieve);
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

	// zvolit m_1, m_2... m_r
	// nasobit prvocisla, ktera nedeli D_hadamard, do te doby nez budou vetsi nez M
	mpz_class M_akt=1;
	list<int> m_prvocisla;
	m_prvocisla.clear();
	for(int i=0;i<poc_sieve;i++)
	{
		if(sieve[i])
		{
			// TODO: zjistit proc se 3 vyskytuje mezi m_prvocisla kdyz 3^2 deli D_hadamard1
			mpz_class delitel;
			mpz_class prvoc=i;
			mpz_gcd(delitel.get_mpz_t(), prvoc.get_mpz_t(), hadamard.get_mpz_t());
			int idelitel=(int)delitel.get_d();
				
			if(delitel==1)	// podminka: gcd(hadamard, M) == 1
			{
				M_akt*=i;
				m_prvocisla.push_front(i);
			}
			
			if( M_akt > modul_M )
			{
				break;
			}
		}
	}
	cout << endl << "M_akt = " << M_akt << endl;
	// TODO: vymazat zbytecna prvocisla napr. 2, 3..
	/*do
	{
		M_akt*=(*m_iter);
		cout << (*m_iter);
		if(M_akt > M)
		{
			list<int>::iterator rem=m_iter;
			m_iter++;
			m_prvocisla.remove((*rem));
			m_iter--;
			//break;
		}else cout << " * ";
	}while(
		*/
	*r = m_prvocisla.size();
	int* pole = new int[*r];
	int i=0;
	for(list<int>::iterator iter=m_prvocisla.begin();iter!=m_prvocisla.end();iter++)
	{
		pole[i] = (*iter);
		i++;
	}

#ifndef _DEBUG
	delete[] sieve;
#endif
	*moduly=pole;
}
/*
 * vytvori SLR v modulu "modul", zmoduluje vstupní SLR
 * modul - vybrany jednotlivy modul
 * m_matice - tady bude matice modularnich zbytku
 * m_prava_strana - prava strana   - || -
 */
void rozklad_slr_mod(int N,  T* matice, T* prava_strana, int modul, int* m_matice, int* m_prava_strana)
{
	for(int x=0;x<N;x++)	// TODO: paralelizovat tenhle nebo
	{
		int pom;
		for(int y=0;y<N;y++)	// tenhle cyklus
		{
			pom = ((long)matice[get_index(x,y)]) % modul;
			m_matice[get_index(x,y)] = pom;
		}
		pom = ((long)prava_strana[x]) % modul;
		m_prava_strana[x] = pom;
	}
}

void gauss_jordan_elim(int N, int modul, int* m_matice, int* m_prava_strana, int* m_vys_jmenovatel)
{
	// TODO: posouvat cisla v radcich doleva, kvuli CUDA, aby se pristupovalo stale na ty stejna mista v pameti, 
	//       vysledek bude v prvnim sloupci matice
	for(int ipivot=0;ipivot<N;ipivot++)
	{
		// deleni nulou => nasobeni inverznim prvkem
		if(m_matice[get_index(ipivot, ipivot)]==0)
		{
			// v 'ipivot'-tem radku na diagonále je nula => vymena s jinym radkem
			int novy_pivot=ipivot;
			do{
				novy_pivot++;
			}while(m_matice[get_index(ipivot, novy_pivot)]==0 && novy_pivot<N);

			if(m_matice[get_index(ipivot, novy_pivot)]!=0 && novy_pivot<N)		// nasel jsem radek s nenulovym prvkem ve sloupci ipivot
			{
				// vymena radku ipivot a novy_pivot
				int pom;
				for(int iX=0;iX<N;iX++)
				{
					pom=m_matice[get_index(iX, ipivot)];
					m_matice[get_index(iX, ipivot)]=m_matice[get_index(iX, novy_pivot)];
					m_matice[get_index(iX, novy_pivot)]=pom;
				}
				pom=m_prava_strana[ipivot];
				m_prava_strana[ipivot]=m_prava_strana[novy_pivot];
				m_prava_strana[novy_pivot]=pom;
			}else
			{
				// matice nema v 'ipivot'-tem sloupci nenulovy prvek => je singularni
				//cout << "singularni" << endl;
				for(int i=0;i<N;i++)	// singularni matice => vysledky jsou nulove (nepouzitelne)
				{
					m_prava_strana[i]=0;
					m_vys_jmenovatel[i]=1;
				}
				return;
			}
		}
		int inverzni=m_inv(modul, m_matice[get_index(ipivot, ipivot)]);

		for(int iY=0;iY<N;iY++)	// prochazi jednotlive radky
		{
			if(iY==ipivot) continue;
			int pom;
			int multipl=(m_matice[get_index(ipivot, iY)]*inverzni) % modul;
			for(int iX=0;iX<N;iX++)	// prochazi cisla v i1-tem radku
			{
				int m1=m_matice[get_index(iX, iY)];
				int m2=m_matice[get_index(iX, ipivot)];
				// TODO: jak cuda moduluje hlavne zaporny cisla? potrebuju interval <0;modul)
				pom = m1-multipl*m2;
				pom=pom % modul;
				if(pom<0) pom+=modul;
				m_matice[get_index(iX, iY)]=pom;
			}
			pom = m_prava_strana[iY]-multipl*m_prava_strana[ipivot];
			// TODO: jak cuda moduluje hlavne zaporny cisla? potrebuju interval <0;modul)
			m_prava_strana[iY]=pom % modul;
			if(m_prava_strana[iY]<0) m_prava_strana[iY]+=modul;
		}
		// ulozit diagonalu do m_vys_jmenovatel
		for(int iX=0;iX<N;iX++)
		{
			m_vys_jmenovatel[iX]=m_matice[get_index(iX, iX)];
		}
				
	}
}
/*
 * r - pocet jednotlivych modulu
 */
void zpetny_prevod(int r, int** vys_citatel, int** vys_jmenovatel, T* vysledek)
{
}

void do_modular(int N,  T* matice, T* prava_strana)
{
	if(matice==NULL || prava_strana==NULL) return;
		
	vstupni_slr(N, matice, prava_strana);
		
	cout << "\nhadamard = ";
	mpz_class D_hadamard;
	T M;
	exec_hadamard(N, matice, prava_strana, &D_hadamard, &M);
	cout << D_hadamard << "; M = " << M << endl;

	int m_r;
	int* m_moduly;
	exec_moduly(N, matice, prava_strana, D_hadamard, M, &m_r, &m_moduly);

	for(int i=0;i<m_r;i++) cout << m_moduly[i] << "\t";
	// TODO: spustit reseni v jednotlivych modulech
	int** m_prava_strana=new int*[m_r];
	int** m_diagonala=new int*[m_r];
	int* m_matice=new int[N*N];
	int m_i=0;
	for(m_i=0;m_i<m_r;m_i++)
	{
		m_prava_strana[m_i]=new int[N];
		m_diagonala[m_i]=new int[N];
		for(int i=0;i<N;i++)
		{
			m_prava_strana[m_i][i]=i;
			m_diagonala[m_i][i]=i;
		}

		// zkopirovat a modulovat
		rozklad_slr_mod(N, matice, prava_strana, m_moduly[m_i], m_matice, m_prava_strana[m_i]);

		// spocitat gauss-jordanovu eliminaci
		gauss_jordan_elim(N, m_moduly[m_i], m_matice, m_prava_strana[m_i], m_diagonala[m_i]);


	}
		
	// vypsat vysledky m_prava_strana
	cout << "VYSLEDKY: " << endl;
	delete[] m_matice;

	for(int i=0;i<m_r;i++)
	{
		cout << m_moduly[i] << "\t";
	}
	cout << endl;
	for(int y=0;y<N;y++)
	{
		m_i=0;
		for(int i=0;i<m_r;i++)
		{
			cout << m_prava_strana[i][y] << "/" << m_diagonala[i][y] << "\t";
		}
		cout << endl;
	}
	// z vysledku m_prava_strana ziskat celociselne vysledky - mixed-radix
	// zpetny_prevod(m_r, m_prava_strana, m_diagonala, 

		


	delete[] *m_prava_strana;
	delete[] *m_diagonala;
	delete[] m_prava_strana;
	delete[] m_diagonala;
	//*/
}
void vypsat(int N, T* matice, T* prava_strana)
{
	printf("\n");
	for(int y=0;y<min(N,6);y++)
	{
		int x;
		for(x=0;x<min(N,5);x++)
		{
			printf("%7.2f ", matice[get_index(x, y)]);
		}
		if(x<N-1) printf("...");
		printf("| %7.2f\n", prava_strana[y]);
	}
}

// frexp, ldexp
int ifrexp(T X, int * Y)
{
	T a=frexp(X, Y);
	int exponent;
	for(exponent=1;exponent<numeric_limits<T>::digits;exponent++)
	{
		a=2.0*a;
		if( a == floor(a) ) break;
	}
	(*Y)-=exponent;
	return (int)a;
}

int m_inv(int modulo, int cislo)
{
	// TODO
	if(cislo==0) return 0;
	int i;
	for(i=1;i<=modulo;i++)
	{
		if( (cislo*i)% modulo==1 ) break;
	}
	return i;
}