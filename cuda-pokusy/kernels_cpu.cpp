#include "stdafx.h"
#include <iostream>
#include <fstream>
#include "kernels_cpu.h"

using namespace std;

int get_index(int X, int Y, int N);

//template<class TYPE>
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

void gauss_jordan_elim_for(int N, int modul, int* m_matice, int* m_prava_strana, int* m_vys_jmenovatel)
{
	// TODO: posouvat cisla v radcich doleva, kvuli CUDA, aby se pristupovalo stale na ty stejna mista v pameti, 
	//       vysledek bude v prvnim sloupci matice
	for(int ipivot=0;ipivot<N;ipivot++)
	{
		// deleni nulou => nasobeni inverznim prvkem
		if(m_matice[get_index(ipivot, ipivot, N)]==0)
		{
			// v 'ipivot'-tem radku na diagonále je nula => vymena s jinym radkem
			int novy_pivot=ipivot;
			do{
				novy_pivot++;
			}while(m_matice[get_index(ipivot, novy_pivot, N)]==0 && novy_pivot<N);

			if(m_matice[get_index(ipivot, novy_pivot, N)]!=0 && novy_pivot<N)		// nasel jsem radek s nenulovym prvkem ve sloupci ipivot
			{
				// vymena radku ipivot a novy_pivot
				int pom;
				for(int iX=0;iX<N;iX++)
				{
					pom=m_matice[get_index(iX, ipivot, N)];
					m_matice[get_index(iX, ipivot, N)]=m_matice[get_index(iX, novy_pivot, N)];
					m_matice[get_index(iX, novy_pivot, N)]=pom;
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
		int multipl1 = m_matice[get_index(ipivot, ipivot, N)];
		for(int iY=0;iY<N;iY++)	// prochazi jednotlive radky
		{
			if(iY==ipivot) continue;
			int pom;
			int multipl2 = m_matice[get_index(ipivot, iY, N)];
			for(int iX=0;iX<N;iX++)	// prochazi cisla v i1-tem radku
			{
				int m1=m_matice[get_index(iX, iY, N)];
				int m2=m_matice[get_index(iX, ipivot, N)];
				// TODO: jak cuda moduluje hlavne zaporny cisla? potrebuju interval <0;modul)
				pom = multipl1*m1-multipl2*m2;
				pom=pom % modul;
				//if(pom<0) pom+=modul;
				m_matice[get_index(iX, iY, N)]=pom;
			}
			pom = multipl1*m_prava_strana[iY]-multipl2*m_prava_strana[ipivot];
			// TODO: jak cuda moduluje hlavne zaporny cisla? potrebuju interval <0;modul)
			m_prava_strana[iY]=pom % modul;
			//if(m_prava_strana[iY]<0) m_prava_strana[iY]+=modul;
		}
		//cout << "pivot: " << ipivot << endl;
		//vypsat_matlab(N, m_matice, m_prava_strana);
	}
	// ulozit diagonalu do m_vys_jmenovatel
	for(int iX=0;iX<N;iX++)
	{
		m_vys_jmenovatel[iX]=m_matice[get_index(iX, iX, N)];
	}
}

void gauss_jordan_elim_while(int N, int modul, int* m_matice, int* m_prava_strana, int* m_vys_jmenovatel)
{
	// TODO: posouvat cisla v radcich doleva, kvuli CUDA, aby se pristupovalo stale na ty stejna mista v pameti, 
	//       vysledek bude v prvnim sloupci matice
	int tid=0;
	int itid;
	for(int ipivot=0;ipivot<N;ipivot++)
	{
		// deleni nulou => nasobeni inverznim prvkem
		if(m_matice[get_index(ipivot, ipivot, N)]==0)
		{
			// v 'ipivot'-tem radku na diagonále je nula => vymena s jinym radkem
			int novy_pivot=ipivot;
			do{
				novy_pivot++;
			}while(m_matice[get_index(ipivot, novy_pivot, N)]==0 && novy_pivot<N);

			if(m_matice[get_index(ipivot, novy_pivot, N)]!=0 && novy_pivot<N)		// nasel jsem radek s nenulovym prvkem ve sloupci ipivot
			{
				// vymena radku ipivot a novy_pivot
				int pom;
				itid=tid;
				while(itid<=N)
				{
					if(itid==N)
					{
						pom=m_prava_strana[ipivot];
						m_prava_strana[ipivot]=m_prava_strana[novy_pivot];
						m_prava_strana[novy_pivot]=pom;
					}else
					{
						pom=m_matice[get_index(itid, ipivot, N)];
						m_matice[get_index(itid, ipivot, N)]=m_matice[get_index(itid, novy_pivot, N)];
						m_matice[get_index(itid, novy_pivot, N)]=pom;
					}
					itid+=1;
				}
			}else
			{
				// matice nema v 'ipivot'-tem sloupci nenulovy prvek => je singularni
				//cout << "singularni" << endl;
				itid=tid;
				while(itid<=N)	// singularni matice => vysledky jsou nulove (nepouzitelne)
				{
					m_prava_strana[itid]=0;
					m_vys_jmenovatel[itid]=1;
					itid+=1;
				}
				return;
			}
		}
		int multipl1 = m_matice[get_index(ipivot, ipivot, N)];
		//*/
		itid=tid;
		while(itid<N)	// prochazi jednotlive radky
		{
			if(itid==ipivot)
			{
				itid+=1;
				continue;
			}
			int pom;
			int multipl2 = m_matice[get_index(ipivot, itid, N)];
			for(int iX=0;iX<N;iX++)	// prochazi cisla v i1-tem radku
			{
				int m1=m_matice[get_index(iX, itid, N)];
				int m2=m_matice[get_index(iX, ipivot, N)];
				pom = multipl1*m1-multipl2*m2;
				pom=pom % modul;
				m_matice[get_index(iX, itid, N)]=pom;
			}
			pom = multipl1*m_prava_strana[itid]-multipl2*m_prava_strana[ipivot];
			m_prava_strana[itid]=pom % modul;
			itid+=1;
		}
		/*/
		for(int iY=0;iY<N;iY++)	// prochazi jednotlive radky
		{
			if(iY==ipivot) continue;
			int pom;
			int multipl2 = m_matice[get_index(ipivot, iY, N)];
			itid=tid;
			while(itid<N)	// prochazi cisla v i1-tem radku
			{
				int m1=m_matice[get_index(itid, iY, N)];
				int m2=m_matice[get_index(itid, ipivot, N)];
				// TODO: jak cuda moduluje hlavne zaporny cisla? potrebuju interval <0;modul)
				pom = multipl1*m1-multipl2*m2;
				pom=pom % modul;
				//if(pom<0) pom+=modul;
				m_matice[get_index(itid, iY, N)]=pom;
				itid+=1;
			}
			pom = multipl1*m_prava_strana[iY]-multipl2*m_prava_strana[ipivot];
			// TODO: jak cuda moduluje hlavne zaporny cisla? potrebuju interval <0;modul)
			m_prava_strana[iY]=pom % modul;
			//if(m_prava_strana[iY]<0) m_prava_strana[iY]+=modul;
		}//*/
		
		// TODO: _syncthread();
	}
	// ulozit diagonalu do m_vys_jmenovatel
	itid=tid;
	while(itid<N)
	{
		m_vys_jmenovatel[itid]=m_matice[get_index(itid, itid, N)];
		itid+=1;
	}
}

int get_index(int X, int Y, int N)	// SLOUPEC, RADEK
{
	return X*N+Y;
}

void vypsat_mat(int N, TYPE* matice, TYPE* prava_strana)
{
	cout << endl;
	for(int y=0;y<min(N,6);y++)
	{
		int x;
		for(x=0;x<min(N,5);x++)
		{
			cout.precision(7);
			cout << matice[get_index(x, y, N)] << "\t";
		}
		if(x<N-1) cout << "...";
		cout << "| " << prava_strana[y] << endl;
	}
}
void vypsat_matlab(int N, TYPE* matice, TYPE* prava_strana)
{
	cout << endl << "A=[";
	for(int y=0;y<N;y++)
	{
		int x;
		for(x=0;x<N;x++)
		{
			cout << matice[get_index(x, y, N)];
			if(x<N-1) cout << ",";
		}
		if(y<N-1) cout << ";";
	}
	cout << "];" << endl << "b=[";
	for(int y=0;y<N;y++)
	{
		cout << prava_strana[y];
		if(y<N-1) cout << ";";
	}
	cout << "];" << endl;
}

void vypsat_vys(int N, TYPE* citatel, TYPE* jmenovatel)
{
	cout << endl;
	//cout.precision(7);
	int i;
	for(i=0;i<min(N,30);i++)
	{
		cout << citatel[i] << "/" << jmenovatel[i] << "\t";
	}
	if(i<N-1) cout << "...";
}