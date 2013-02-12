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
		if(size>0)
		{
			this->N=size;
			pointer=new T[size*size];
			prava_strana=new T[size];
		}
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
	int get_index(int X, int Y)
	{
		return X*N+Y;
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
	void fill_hilbert(void)
	{
		for(int y=0;y<N;y++)
		{
			for(int x=0;x<N;x++)
			{
				T val=1.0/((T)(1+x+y));
				set_cell(x, y, val);
			}
			prava_strana[y]=(y+1)*(y+1);
		}
	}
	void save_matrix(char* filename)
	{
		fstream file;
		file.open(filename, fstream::out);
		file << N << endl;
		for(int y=0;y<N;y++)
		{
			for(int x=0;x<N;x++)
			{
				file << get_cell(x, y) << "\t";
			}
			file << prava_strana[y] << endl;
		}
		file.close();
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
		cout << "\nhadamard ";
		mpz_class D_hadamard1=1;
		for(int iX=0;iX<N;iX++)
		{
			mpz_class souc=0;
			for(int iY=0;iY<N;iY++)
			{
				souc+=(get_cell(iX, iY))*(get_cell(iX, iY));
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
		mpz_class D_hadamard=sqrt(D_hadamard1);
		

		cout << "\nprvocisla: ";
		for(int i = 0; i <= poc_sieve; i ++)
		{
			//if(sieve[i]) cout << i << ", ";
		}
		cout << "\nD = " << D_hadamard << "\n";

		

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
				mpz_gcd(delitel.get_mpz_t(), prvoc.get_mpz_t(), D_hadamard.get_mpz_t());
				int idelitel=(int)delitel.get_d();
				
				if(delitel==1)
				{
					M_akt*=i;
					m_prvocisla.push_front(i);
				}
			
				if( M_akt > M )
				{
					break;
				}
			}
		}
		cout << "M     = " << ((mpz_class)M) << " = ";
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
		cout << endl << "M_akt = " << M_akt << endl;
		// TODO: spustit reseni v jednotlivych modulech
		int** m_prava_strana=new int*[m_prvocisla.size()];
		int** m_diagonala=new int*[m_prvocisla.size()];
		int* m_matice=new int[N*N];
		int m_i=0;
		for(m_i=0;m_i<m_prvocisla.size();m_i++)
		{
			m_prava_strana[m_i]=new int[N];
			m_diagonala[m_i]=new int[N];
			for(int i=0;i<N;i++)
			{
				m_prava_strana[m_i][i]=i;
				m_diagonala[m_i][i]=i;
			}
		}
		m_i=0;
		for(list<int>::iterator m_iter=m_prvocisla.begin();m_iter!=m_prvocisla.end();m_iter++)
		{
			for(int i=0;i<N;i++)
			{
				m_prava_strana[m_i][i]=9;
				m_diagonala[m_i][i]=8;
			}
			//cout << endl;
			// zkopirovat a modulovat
			for(int y=0;y<N;y++)
			{
				int pom;
				for(int x=0;x<N;x++)
				{
					pom=((long)get_cell(x,y)) % (*m_iter);
					//if( pom==0 ) pom=(*m_iter);
					m_matice[get_index(x,y)]=pom;
				}
				pom=((long)prava_strana[y]) % (*m_iter);
				//if( pom==0 ) pom=(*m_iter);
				m_prava_strana[m_i][y]=pom;
			}


			// spocitat gauss-jordanovu eliminaci
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
						double pom;
						for(int iX=0;iX<N;iX++)
						{
							pom=m_matice[get_index(iX, ipivot)];
							m_matice[get_index(iX, ipivot)]=m_matice[get_index(iX, novy_pivot)];
							m_matice[get_index(iX, novy_pivot)]=pom;
						}
						pom=m_prava_strana[m_i][ipivot];
						m_prava_strana[m_i][ipivot]=m_prava_strana[m_i][novy_pivot];
						m_prava_strana[m_i][novy_pivot]=pom;
					}else
					{
						// matice nema v 'ipivot'-tem sloupci nenulovy prvek => je singularni
						//cout << "singularni" << endl;
						for(int i=0;i<N;i++)	// singularni matice => vysledky jsou nulove (nepouzitelne)
						{
							m_prava_strana[m_i][i]=0;
							m_diagonala[m_i][i]=0;
						}
						//break;
					}
				}
				int inverzni=m_inv((*m_iter), m_matice[get_index(ipivot, ipivot)]);

				for(int iY=0;iY<N;iY++)	// prochazi jednotlive radky
				{
					if(iY==ipivot) continue;
					int pom;
					int multipl=(m_matice[get_index(ipivot, iY)]*inverzni) % (*m_iter);
					for(int iX=0;iX<N;iX++)	// prochazi cisla v i1-tem radku
					{
						int m1=m_matice[get_index(iX, iY)];
						int m2=m_matice[get_index(iX, ipivot)];
						pom = m1-multipl*m2;
						pom=pom % (*m_iter);
						if(pom<0) pom+=(*m_iter);
						m_matice[get_index(iX, iY)]=pom;
					}
					pom = m_prava_strana[m_i][iY]-multipl*m_prava_strana[m_i][ipivot];
					m_prava_strana[m_i][iY]=pom % (*m_iter);
					if(m_prava_strana[m_i][iY]<0) m_prava_strana[m_i][iY]+=(*m_iter);
				}
				// ulozit diagonalu do m_diagonala
				for(int iX=0;iX<N;iX++)
				{
					m_diagonala[m_i][iX]=m_matice[get_index(iX, iX)];
				}
				
			}
			//*/
			
			m_i++;
		}
		
		delete[] m_matice;
		// vypsat vysledky m_prava_strana
		cout << "VYSLEDKY: " << endl;
		for(list<int>::iterator m_iter=m_prvocisla.begin();m_iter!=m_prvocisla.end();m_iter++)
		{
			cout << (*m_iter) << "\t";
		}
		cout << endl;
		for(int y=0;y<N;y++)
		{
			m_i=0;
			for(list<int>::iterator m_iter=m_prvocisla.begin();m_iter!=m_prvocisla.end();m_iter++)
			{
				cout << m_prava_strana[m_i][y] << "/" << m_diagonala[m_i][y] << "\t";
				m_i++;
			}
			cout << endl;
		}
		// z vysledku m_prava_strana ziskat celociselne vysledky - mixed-radix
		// TODO: chybne
		for(int y=0;y<N;y++)
		{
			mpz_class citatel=0;
			mpz_class jmenovatel=0;
			mpz_class radix=1;
			m_i=0;
			for(list<int>::iterator m_iter=m_prvocisla.begin();m_iter!=m_prvocisla.end();m_iter++)
			{
				//cout << "+" << m_prava_strana[m_i][y] <<"*" << radix;
				//m_prava_strana[m_i][y]
				citatel+=m_prava_strana[m_i][y]*radix;
				jmenovatel+=m_diagonala[m_i][y]*radix;
				radix*=(*m_iter);

				m_i++;
			}
			//cout << endl;
			prava_strana[y]=citatel.get_d()/jmenovatel.get_d();
			//cout << citatel << "/" << endl << jmenovatel << " = " << prava_strana[y] << endl;
			m_i=0;
			for(list<int>::iterator m_iter=m_prvocisla.begin();m_iter!=m_prvocisla.end();m_iter++)
			{
				mpz_class a=(*m_iter);
				mpz_class m=citatel % a;
				double m_citatel=m.get_d();
				m=jmenovatel % a;
				double m_jmenovatel=m.get_d();
				m_prava_strana[m_i][y]==m_citatel;
				m_diagonala[m_i][y]==m_jmenovatel;
				m_i++;
			}
		}

		//delete[] sieve;
		for(int i=0;i<m_prvocisla.size();i++)
		{
			//delete[] m_prava_strana[m_i];
			//delete[] m_diagonala[m_i];
		}
		//delete[] m_prava_strana;
		//delete[] m_diagonala;
		//*/
	}
	void vypsat()
	{
		printf("\n");
		for(int y=0;y<min(N,6);y++)
		{
			int x;
			for(x=0;x<min(N,5);x++)
			{
				printf("%7.2f ", get_cell(x, y));
			}
			if(x<N-1) printf("...");
			printf("| %7.2f\n", prava_strana[y]);
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