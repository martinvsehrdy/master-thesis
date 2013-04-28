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
/* 
 * gauss-jordanova eliminace, jednovlaknova, ve for-cyklech, primo na datech ve vstupnim poli, 
 * bez deleni - nasobim oba mergujici radky, po vypoctu kazde bunky se moduluje
 */
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
/* 
 * gauss-jordanova eliminace, jednovlaknova, ve while-cyklech, primo na datech ve vstupnim poli, 
 * bez deleni - nasobim oba mergujici radky, po vypoctu kazde bunky se moduluje, 
 * dva pristupy k matici: ipivot prochazi pres matici pres radky/sloupce
 */
void gauss_jordan_elim_while(int N, int modul, int* m_matice, int* m_prava_strana, int* m_vys_jmenovatel)
{
	// TODO: posouvat cisla v radcich doleva, kvuli CUDA, aby se pristupovalo stale na ty stejna mista v pameti, 
	//       vysledek bude v prvnim sloupci matice
	int tid=0;
	int itid;
	for(int ipivot=0;ipivot<N;ipivot++)
	{
		cout << endl << "pivot=" << ipivot << " ";
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
				cout << novy_pivot;
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
		cout << endl;
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
			// DEBUG
			cout << multipl1 << "," << multipl2 << " | ";

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
		vypsat_mat(N, m_matice, m_prava_strana);
	}
	// ulozit diagonalu do m_vys_jmenovatel
	itid=tid;
	while(itid<N)
	{
		m_vys_jmenovatel[itid]=m_matice[get_index(itid, itid, N)];
		itid+=1;
	}
}

void gauss_jordan_elim_p1(int modul, int nx, int ny, int* s_matice, int* actions, int* diag_pivot, int zpusob_zprac)
/* N, modul - stejne jako v gauss_jordan_elim_..
 * n - velikost s_matice
 * s_matice - pole shared, submatice
 * actions - ny cisel (indexu) radku kvuli vymene radku s nulovym cislem na diagonale, nasleduje pole dvojic cisel, kteryma nasobim radek a pivota
 * diag_pivot - diagonala ze submatice lezici na diagonale matice (sx==sy), predstavuje pivotni radky
 * zpusob_zprac - co s tou submatici delam
 *	1 - na diagonale, generuju actions
 *	2 - upravuju submatici na nulovou matici, generuju actions
 *	3 - tupe upravovani podle action
 */
{
	// TODO: cuda __mul64hi
	int tid=0;
	int itid;
	
	for(int ipivot=0;ipivot<ny;ipivot++)
	{
		if(zpusob_zprac<=2)
		{
			// deleni nulou => nasobeni inverznim prvkem
			if(s_matice[get_index(ipivot, ipivot, nx)]==0)
			{
				// v 'ipivot'-tem radku na diagonále je nula => vymena s jinym radkem
				int novy_pivot=ipivot;
				do{
					novy_pivot++;
				}while(s_matice[get_index(ipivot, novy_pivot, nx)]==0 && novy_pivot<ny);

				if(s_matice[get_index(ipivot, novy_pivot, nx)]!=0 && novy_pivot<ny)		// nasel jsem radek s nenulovym prvkem ve sloupci ipivot
				{
					// vymena radku ipivot a novy_pivot v actions
					actions[ipivot]=novy_pivot;
				}else
				{
					// TODO: matice nema v 'ipivot'-tem sloupci nenulovy prvek => je singularni
					//cout << "singularni" << endl;
					itid=tid;
					// singularni matice => vysledky jsou nulove (nepouzitelne)
				
					return;
					// TODO: musim nacist cely sloupec a hledat nenulovy cislo => vymena s pivotnim radkem
				}
			}else
			{
				actions[ipivot]=ipivot;
			}
		}

		if(actions[ipivot]!=ipivot)
		{
			// vymena radku ipivot a actions[ipivot]
			int pom;
			itid=tid;
			while(itid<nx)
			{
				pom=s_matice[get_index(itid, ipivot, nx)];
				s_matice[get_index(itid, ipivot, nx)]=s_matice[get_index(itid, actions[ipivot], nx)];
				s_matice[get_index(itid, actions[ipivot], nx)]=pom;
				itid+=1;
			}
		}

		int multipl1;
		itid=tid;
		while(itid<ny)	// prochazi jednotlive radky
		{
			int iact=ny+2*(itid+ny*ipivot);
			if(zpusob_zprac==1 && itid==ipivot)
			{
				actions[iact]=1;
				actions[iact+1]=0;
				itid+=1;
				continue;
			}
			int multipl2;
			if(zpusob_zprac==1)	// supmatice je na dianogale
			{
				multipl1=s_matice[get_index(ipivot, ipivot, nx)];
				actions[iact]=multipl1;
				multipl2=s_matice[get_index(ipivot, itid, nx)];
				actions[iact+1]=multipl2;
				cout << multipl1 << "(" << (iact) << ")," << multipl2 << "(" << (iact+1) << ") | ";
			}else
			if(zpusob_zprac==2)
			{
				multipl1=diag_pivot[ipivot];
				actions[iact]=multipl1;
				multipl2=s_matice[get_index(ipivot, itid, nx)];
				actions[iact+1]=multipl2;
				cout << multipl1 << "(" << (iact) << ")," << multipl2 << "(" << (iact+1) << ") | ";
			}else
			{
				multipl1=actions[iact];
				multipl2=actions[iact+1];
			}
			long long pom;
			for(int iX=0;iX<nx;iX++)	// prochazi cisla v i1-tem radku
			{
				// TODO: atomicOperators
				long long m1=(long long)s_matice[get_index(iX, itid, nx)];
				long long m2;
				if(zpusob_zprac==2) m2=(long long)(iX==ipivot ? diag_pivot[iX] : 0);
				else m2=(long long)s_matice[get_index(iX, ipivot, nx)];
				pom = multipl1*m1-multipl2*m2;
				pom=pom % modul;
				s_matice[get_index(iX, itid, nx)]=(int)pom;
			}
			itid+=1;
		}
	}
}

void gauss_jordan_elim_part(int N, int modul, int* m_matice, int* m_prava_strana, int* m_vys_jmenovatel)
{
	// nahraje submatici do shared
	int nx, ny;	// velikost submatice
	nx = ny = 3;
	int sx, sy;	// indexy urcujici submatici ve velke matici
	int px, py;	// indexy urcujici pozici prvku v submatici
	// __shared__
	int* sub_m=(int*)malloc(nx*ny*sizeof(int));
	int* citatele=(int*)malloc((2*ny*ny+ny)*sizeof(int));
	int* diag_pivot=(int*)malloc(nx*sizeof(int));

	for(int sdiag=0;sdiag*nx<N;sdiag++)
	for(int sy1=-1;sy1*ny<N;sy1++)
	{
		if(sy1==sdiag) continue;
		if(sy1<0) sy=sdiag;
		else sy=sy1;
		for(sx=sdiag;sx*nx<=N;sx++)
		{
			int zpusob_zpracovani=0;
			if(sx==sdiag && sy==sdiag) zpusob_zpracovani=1;	// na diagonale, generuju actions
			else if(sx==sdiag)  zpusob_zpracovani=2;	// upravuju submatici na nulovou matici, generuju actions
			else zpusob_zpracovani=3;	// tupe upravovani podle action

			px=0;
			while(px<nx)
			{
				int x1=sx*nx+px;
				if(x1<=N)
				{
					for(py=0;py<ny;py++)
					{
						int y1=sy*ny+py;
						if(y1 < N)
						{
							if(x1==N)
							{
								sub_m[get_index(px, py, nx)] = m_prava_strana[y1];
								// DEBUG
								//m_prava_strana[y1]=zpusob_zpracovani;

							}else
							{
								sub_m[get_index(px, py, nx)] = m_matice[get_index(x1, y1, N)];
								// DEBUG
								//m_matice[get_index(x1, y1, N)]=zpusob_zpracovani;
							}
						}
					}
				}
				if(x1<N) diag_pivot[px] = m_matice[get_index(x1, x1, N)];

				px++;
			}

			// spusti .._p1
			gauss_jordan_elim_p1(modul, nx, ny, sub_m, citatele, diag_pivot, zpusob_zpracovani);
			
			// zpetne nahravani do matice
			px=0;
			while(px<nx)
			{
				int x1=sx*nx+px;
				if(x1<=N)
				{
					for(py=0;py<ny;py++)
					{
						int y1=sy*ny+py;
						if(y1 < N)
						{
							if(x1==N)
							{
								m_prava_strana[y1] = sub_m[get_index(px, py, nx)];

							}else
							{
								 m_matice[get_index(x1, y1, N)] = sub_m[get_index(px, py, nx)];
							}
							if(zpusob_zpracovani==2 && sx>sy)
							{
								// prenasobeni diagonaly v hotovych sloupcich
								int pom=citatele[ny+2*(py+ny*y1)] * m_matice[get_index(y1, y1, N)];
								pom= pom % modul;
								m_matice[get_index(y1, y1, N)] = pom;
							}
						}
					}
				}
				px++;
			}

			// DEBUG
			for(int i=0;i<ny;i++)
			{
				cout << citatele[i] << " ";
			}
			cout << endl;
			for(int i=0;i<2*ny*ny;i++)
			{
				cout << citatele[ny+i];
				if(i&(int)1) cout << " | ";
				else cout << ",";
			}
			cout << endl;
			vypsat_mat(N, m_matice, m_prava_strana);


		}
	}



	free(sub_m);
	free(citatele);
	free(diag_pivot);
}
/* inverse = [1;modul-1], size(inverse)=(modul-1)
 * inverzni k cislu A je inverse[A-1]
 */
void gener_inverse(int modul, int* inverse)
{
	int tid=0;
	int bdim=1;	// blockDim.x;

	int cislo=tid+1;
	while(cislo<modul)
	{
		inverse[cislo-1]=0;
		cislo+=bdim;
	}
	
	cislo=tid+1;
	while(cislo<modul)
	{
		if(inverse[cislo-1]==0)
		{
			int inv=0;
			for(inv=1;inv<modul;inv++)
			{
				if( (cislo*inv)% modul==1 )
				{
					inverse[cislo-1]=inv;
					inverse[inv-1]=cislo;
					break;
				}
			}
		}
		cislo+=bdim;
	}
}


int get_index(int X, int Y, int N)	// SLOUPEC, RADEK
{
	return X*N+Y;
}

void vypsat_mat(int N, TYPE* matice, TYPE* prava_strana)
{
	cout << endl;
	for(int y=0;y<min(N,12);y++)
	{
		int x;
		for(x=0;x<min(N,10);x++)
		{
			cout.precision(5);
			cout << matice[get_index(x, y, N)] << "\t";
		}
		if(x<N-1) cout << "...";
		cout << "| ";
		if(prava_strana!=NULL) cout << prava_strana[y];
		cout << endl;
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