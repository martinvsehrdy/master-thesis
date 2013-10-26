#include "stdafx.h"
#include "kernels_cpu.h"
#include "templates_functions.h"
#include <cstdio>
#include <conio.h>
#include "time_measure.h"
#include "common.h"

using namespace std;


// elementarni uprava s delenim
unsigned int elem_uprava_s_delenim(unsigned int modul, unsigned int a_xy, unsigned int a_xp, unsigned int a_py)
// \STATE $a_{xy} := a_{xy} - a_xp \cdot a_py$
{
	unsigned long long m1;
	unsigned long long pom;

	pom = a_xy;
	m1 = a_xp;
	
	m1 *= a_py;
	if(pom >= m1)
	{
		pom -= m1;
		pom %= modul;
	}else
	{
		m1 -= pom;
		m1 %= modul;
		pom = modul-m1;
	}
	return ((unsigned int)pom);
}
// elementarni uprava bez deleni
unsigned int elem_uprava_bez_deleni(unsigned int modul, unsigned int a_xy, unsigned int a_pp, unsigned int a_xp, unsigned int a_py)
// \STATE $a_{xy} := a_{xy} \cdot a_pp - a_xp \cdot a_py$
{
	unsigned long long m1;
	unsigned long long pom;

	pom = a_xy;
	m1 = a_xp;
	
	pom *= a_pp;
	m1 *= a_py;
	if(pom >= m1)
	{
		pom -= m1;
		pom %= modul;
	}else
	{
		m1 -= pom;
		m1 %= modul;
		pom = modul-m1;
	}
	return ((unsigned int)pom);
}
/* 
 * gauss-jordanova eliminace, jednovlaknova, ve for-cyklech, primo na datech ve vstupnim poli, 
 * bez deleni - nasobim oba mergujici radky, po vypoctu kazde bunky se moduluje
 */
void gauss_jordan_elim_for(int N, int modul, unsigned int* m_matice, unsigned int* m_prava_strana, unsigned int zpusob)
{
	start_measuring();
	for(int ipivot=0;ipivot<N;ipivot++)
	{
		//cout << ipivot << endl;
		// deleni nulou => nasobeni inverznim prvkem
		if(m_matice[get_index(ipivot, ipivot, N)]==0)
		{
			// v 'ipivot'-tem radku na diagon�le je nula => vymena s jinym radkem
			int novy_pivot=ipivot;
			do{
				novy_pivot++;
			}while(m_matice[get_index(ipivot, novy_pivot, N)]==0 && novy_pivot<N);

			if(m_matice[get_index(ipivot, novy_pivot, N)]!=0 && novy_pivot<N)		// nasel jsem radek s nenulovym prvkem ve sloupci ipivot
			{
				// vymena radku ipivot a novy_pivot
				unsigned int pom;
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
				}
				return;
			}
		}
		unsigned int a_pp;
		if( zpusob & ZPUSOB_S_DELENIM )
		{
			unsigned int a_pp_inv = compute_inverse_eukleides(m_matice[get_index(ipivot, ipivot, N)], modul);
			//cout << endl << "vydelit " << a_pp_inv << ": ";
			// vydelit cely ipivot-ty radek cislem a_pp
			unsigned long long pom;
			for(int iX=0;iX<N;iX++)
			{
				pom = m_matice[get_index(iX, ipivot, N)];
				pom *= a_pp_inv;
				pom %= modul;
				m_matice[get_index(iX, ipivot, N)]=(unsigned int)pom;
			}
			pom = m_prava_strana[ipivot];
			pom *= a_pp_inv;
			pom %= modul;
			m_prava_strana[ipivot]=(unsigned int)pom;
		}else
		{
			a_pp = m_matice[get_index(ipivot, ipivot, N)];
		}

		for(int iY=0;iY<N;iY++)	// prochazi jednotlive radky
		{
			if(iY==ipivot) continue;
			unsigned int a_py = m_matice[get_index(ipivot, iY, N)];
			//cout << a_py << ", ";
			for(int iX=0;iX<N;iX++)	// prochazi cisla v i1-tem radku
			{
				unsigned int a_xy = m_matice[get_index(iX, iY, N)];
				unsigned int a_xp = m_matice[get_index(iX, ipivot, N)];
				if( zpusob & ZPUSOB_S_DELENIM )
				{
					m_matice[get_index(iX, iY, N)]=elem_uprava_s_delenim(modul, a_xy, a_xp, a_py);
				}else
				{
					m_matice[get_index(iX, iY, N)]=elem_uprava_bez_deleni(modul, a_xy, a_pp, a_xp, a_py);
				}
			}
			if( zpusob & ZPUSOB_S_DELENIM )
			{
				m_prava_strana[iY]=elem_uprava_s_delenim(modul, m_prava_strana[iY], m_prava_strana[ipivot], a_py);
			}else
			{
				m_prava_strana[iY]=elem_uprava_bez_deleni(modul, m_prava_strana[iY], a_pp, m_prava_strana[ipivot], a_py);
			}
		}
		//cout << "pivot: " << ipivot << endl;
		//vypsat_matlab(N, m_matice, m_prava_strana);
		//vypsat_mat(N, N, m_matice, m_prava_strana);
		//cout << endl;
	}
	/*if( !(zpusob & ZPUSOB_S_DELENIM) )
	{
		unsigned long long pom;
		for(int i=0;i<N;i++)
		{
			pom = m_prava_strana[i];
			pom *= compute_inverse_eukleides(m_matice[get_index(i, i, N)], modul);
			pom %= modul;
			m_prava_strana[i] = (unsigned int)pom;
			m_matice[get_index(i, i, N)]=1;
		}
	}*/
	stop_measuring();
}
/* 
 * gauss-jordanova eliminace, jednovlaknova, ve while-cyklech, primo na datech ve vstupnim poli, 
 * bez deleni - nasobim oba mergujici radky, po vypoctu kazde bunky se moduluje, 
 * dva pristupy k matici: ipivot prochazi pres matici pres radky/sloupce
 */
// TODO: matice bude o vel. (N+1) vpravo bude prava strana a posledni radek bude = 0
void gauss_jordan_elim_while(int Sx, int Sy, unsigned int modul, unsigned int* m_matice)
{
	// TODO: posouvat cisla v radcich doleva, kvuli CUDA, aby se pristupovalo stale na ty stejna mista v pameti, 
	//       vysledek bude v prvnim sloupci matice
	int Smin=min(Sx, Sy);
	int bdim=1;
	int tid=0;
	int itid;
	for(int ipivot=0;ipivot<Smin;ipivot++)
	{
		cout << endl << "pivot=" << ipivot << " ";
		int novy_pivot;	// CUDA: shared
		// CUDA: __syncthreads();
		if(tid==0)
		{
				novy_pivot=ipivot;
			// deleni nulou => nasobeni inverznim prvkem
			if(m_matice[get_index(ipivot, ipivot, Sx)]==0)
			{
				// v 'ipivot'-tem radku na diagon�le je nula => vymena s jinym radkem
				do{
					novy_pivot++;
				}while(m_matice[get_index(ipivot, novy_pivot, Sx)]==0 && novy_pivot<Smin);
			}
		}
		// CUDA: __syncthreads();
		// matice je singularni
		if(novy_pivot>=Smin)
		{
			// matice nema v 'ipivot'-tem sloupci nenulovy prvek => je singularni
			//cout << "singularni" << endl;
			itid=tid;
			// singularni matice => vysledky jsou nulove (nepouzitelne)
			//while(itid<=N)
			{
					
				itid+=1;
			}
			return;
		}
		// musim prehodit pivotni radek s jinym
		if(novy_pivot>ipivot)
		{
			cout << novy_pivot;
			// vymena radku ipivot a novy_pivot
			itid=tid;
			unsigned int pom;
			while(itid<=Sx)
			{
				pom=m_matice[get_index(itid, ipivot, Sx)];
				m_matice[get_index(itid, ipivot, Sx)]=m_matice[get_index(itid, novy_pivot, Sx)];
				m_matice[get_index(itid, novy_pivot, Sx)]=pom;
				itid+=bdim;
			}
		}

		// CUDA: __syncthreads();
#ifdef S_DELENIM
		unsigned int a_pp_inv = compute_inverse_eukleides(m_matice[get_index(ipivot, ipivot, Sx)], modul);
		cout << endl << "vydelit " << a_pp_inv << ": ";
		// vydelit cely ipivot-ty radek cislem a_pp
		itid=tid;
		while(itid<Sx)
		{
			unsigned long long pom = m_matice[get_index(itid, ipivot, Sx)];
			pom *= a_pp_inv;
			pom %= modul;
			m_matice[get_index(itid, ipivot, Sx)]=(unsigned int)pom;

			itid+=bdim;
		}
#else
		unsigned int a_pp = m_matice[get_index(ipivot, ipivot, Sx)];
		cout << endl << a_pp << ": ";
#endif

		 /*
		itid=tid;
		while(itid<Sy)	// prochazi jednotlive radky
		{
			if(itid!=ipivot)
			{
				unsigned int a_py = m_matice[get_index(ipivot, itid, Sx)];
				// DEBUG
				cout << a_py << ", ";

				for(int iX=0;iX<Sx;iX++)	// prochazi cisla v i1-tem radku
				{
					unsigned int a_xy = m_matice[get_index(iX, itid, Sx)];
					unsigned int a_xp = m_matice[get_index(iX, ipivot, Sx)];
#ifdef S_DELENIM
					m_matice[get_index(iX, itid, Sx)] = elem_uprava_s_delenim(modul, a_xy, a_xp, a_py);
#else
					m_matice[get_index(iX, itid, Sx)] = elem_uprava_bez_deleni(modul, a_xy, a_pp, a_xp, a_py);
#endif
				}
			}
			itid+=bdim;
		}
		/*/
		for(int iY=0;iY<Sy;iY++)	// prochazi jednotlive radky
		{
			if(iY!=ipivot)
			{
				unsigned int a_py = m_matice[get_index(ipivot, iY, Sx)];
				// DEBUG
				cout << a_py << ", ";
				itid=tid;
				while(itid<Sx)	// prochazi cisla v i1-tem radku
				{
					unsigned int a_xy = m_matice[get_index(itid, iY, Sx)];
					unsigned int a_xp = m_matice[get_index(itid, ipivot, Sx)];
#ifdef S_DELENIM
					m_matice[get_index(itid, iY, Sx)] = elem_uprava_s_delenim(modul, a_xy, a_xp, a_py);
#else
					m_matice[get_index(itid, iY, Sx)] = elem_uprava_bez_deleni(modul, a_xy, a_pp, a_xp, a_py);
#endif
					itid+=bdim;
				}
			}
			// CUDA: __syncthreads();
		}//*/
		vypsat_mat<unsigned int>(Sx, Sy, m_matice, NULL);
	}
#ifndef S_DELENIM
	unsigned long long pom;
	for(int i=0;i<Smin;i++)
	{
		pom = m_matice[get_index(Sx-1, i, Sx)];
		pom *= compute_inverse_eukleides(m_matice[get_index(i, i, Sx)], modul);
		pom %= modul;
		m_matice[get_index(Sx-1, i, Sx)] = (unsigned int)pom;
	}
#endif
}
/* nacte/ulozi podmatici z globalni p. do sdilene nebo zpet
 * Sx, Sy - velikost podmatice, mela by se vejit do sdilene pameti
 * sx, sy - souradnice zvolene podmatice v matici, sx \in [0; ceil(N/Sx)]
 * mat_A, mat_B - zdrojova nebo cilova adresa
 */
void copy_podmatice(int N, int ipivot, int sx, int sy, int Sx, int Sy, unsigned int* mat_A, unsigned int* mat_B, unsigned int* prava_str, int copy_to)
{
	int tid=0;
	int bdim=1;
	int itid=tid;
	unsigned int a;
	
	while(itid<Sy)
	{
		for(int ix=0;ix<Sx;ix++)
		{
			
			int glob_x=ipivot*min(Sx,Sy)+(sx-ipivot)*Sx+ix;
			int glob_y=sy*Sy+itid;
			if(glob_x<=N && glob_y<N)
			{
				if(glob_x<N)
				{
					switch(copy_to)
					{
					case COPY_MAT_A_SH_TO_B_GLOB:
						a = mat_A[get_index(ix, itid, Sx)];
						mat_B[get_index(glob_x, glob_y, N)] = a;
						break;
					case COPY_MAT_B_GLOB_TO_A_SH:
						a = mat_B[get_index(glob_x, glob_y, N)];
						mat_A[get_index(ix, itid, Sx)] = a;
						break;
					}
				}else
				{
					switch(copy_to)
					{
					case COPY_MAT_A_SH_TO_B_GLOB:
						a = mat_A[get_index(ix, itid, Sx)];
						prava_str[glob_y] = a;
						break;
					case COPY_MAT_B_GLOB_TO_A_SH:
						a = prava_str[glob_y];
						mat_A[get_index(ix, itid, Sx)] = a;
						break;
					}
				}
			}else
			{
				if(copy_to == COPY_MAT_B_GLOB_TO_A_SH)
				{
					//if( sx==sy && ix==itid )
					//mat_A[get_index(ix, itid, Sx)] = 1;
					//else
					mat_A[get_index(ix, itid, Sx)] = 0;
				}
			}
		}
		itid+=bdim;
	}
}
// S DELENIM
void compute_podmatice1(int N, unsigned int modul, int sx, int sy, int Sx, int Sy, unsigned int* s_mat, unsigned int* actions)
{
	cout << "modul = " << modul << endl;
	for(int i=0;i<(Sx*Sy+Sx);i++)
	{
		actions[i]=modul;
	}
	// TODO: ukladat info do actions, kontrola debugovanim
	int tid=0;
	int bdim=1;
	unsigned int m1;
	unsigned long long pom;
	int minS=min( min(Sx, Sy), min(N-Sx*sx, N-Sy*sy) );
	// \FOR{$p$ := $1$ do $N$}
	for(int ipivot=0;ipivot<minS;ipivot++)
	{
		// todo: jak se to chova pri S=4, ipivot=3
		cout << endl << "ipivot = " << ipivot << endl;
		vypsat_mat<unsigned int>(Sx, Sy, s_mat, NULL);
		int novy_pivot;
		if(tid==0)
		{
			novy_pivot=ipivot;
			while(s_mat[get_index(novy_pivot, novy_pivot, Sx)]==0 && novy_pivot<minS)
			{
				novy_pivot++;
			}
			if(novy_pivot==minS) novy_pivot=ipivot;
			actions[ipivot] = novy_pivot;
			cout << " radek pivota=" << novy_pivot;
		}
		// cuda: synchronize
		novy_pivot=actions[ipivot];
		if(novy_pivot>=minS) return;
		if(novy_pivot != ipivot)
		{
			unsigned int pom1;
			int ix=ipivot+tid;
			while( (ix<Sx) && (Sx*sx+ix<=N) )
			{
				pom1=s_mat[get_index(ix, novy_pivot, Sx)];
				s_mat[get_index(ix, novy_pivot, Sx)] = s_mat[get_index(ix, ipivot, Sx)];
				s_mat[get_index(ix, ipivot, Sx)] = pom1;
				ix+=bdim;
			}
		}
		// \STATE \COMMENT {vydelit cel� $p$-t� r�dek c�slem $a_{pp}$, v $p$-t�m sloupci na diagon�le bude c�slo 1}
		// \STATE ulo�it "jedna lomeno  $a_{pp}$" do $actions$
		m1=compute_inverse_eukleides(s_mat[get_index(ipivot, ipivot, Sx)], modul);
		actions[minS+ipivot]=m1;
		cout << " | vydelit " << m1;
		// \FOR{$x$ := $p$ do $N$}
		int ix=ipivot+tid;
		while( (ix<Sx) && (Sx*sx+ix<=N) )
		{
		  // \STATE $a_{xp} := \frac{a_{xp}}{a_pp}$
			pom = s_mat[get_index(ix, ipivot, Sx)];
			pom *= m1;
			pom %= modul;
			s_mat[get_index(ix, ipivot, Sx)]=(unsigned int)pom;
			
			ix+=bdim;
		// \ENDFOR
		}
		vypsat_mat<unsigned int>(Sx, Sy, s_mat, NULL);
		// \FOR{$y$ := $1$ do $N$}
			int iy=tid;
		while( (iy<Sy) && (Sy*sy+iy<N) )
		{
		  // \IF {$p$ != $y$}
			if(ipivot != iy)
			{
			// \STATE ulo�it $a_py$ do $actions$
				unsigned int m_py=s_mat[get_index(ipivot, iy, Sx)];
				int index_actions=2*minS+(Sy-1)*ipivot+iy;
				if(ipivot<iy) index_actions--;
				actions[index_actions]=m_py;
				cout << " | minus " << m_py << " krat pivot";
			// \FOR{$x$ := $p$ do $N$}
				for(ix=ipivot;(ix<Sx)&&(Sx*sx+ix<=N);ix++)
				{
			  // \STATE $a_{xy} := a_{xy} - a_xp \cdot a_py$
					s_mat[get_index(ix, iy, Sx)]=elem_uprava_s_delenim(modul, s_mat[get_index(ix, iy, Sx)], s_mat[get_index(ix, ipivot, Sx)], m_py);
			// \ENDFOR
				}
		  // \ENDIF
			}
			iy+=bdim;
		// \ENDFOR
		}

		// DEBUG
		cout << endl;
		for(int i=0;i<(Sx*Sy+Sx);i++)
		{
			printf("%5u", i);
		}
		cout << endl;
		for(int i=0;i<(Sx*Sy+Sx);i++)
		{
			if( actions[i]<modul ) printf("%5u", actions[i]);
			else printf("    _");
		}
		vypsat_mat<unsigned int>(Sx, Sy, s_mat, NULL);
		cout << endl << "=============================" << endl;
		// ---------------------------------
	// \ENDFOR
	}
}
// S DELENIM
void compute_podmatice24(int N, unsigned int modul, int sx, int sy, int Sx, int Sy, unsigned int* s_mat, unsigned int* actions, unsigned int zpusob)
{
	unsigned int* p_mat=&(s_mat[Sx*Sy]);
	unsigned int* actions1=&(actions[1]);	// indexy pivotnich radku, permutace radku, 'Sy' cisel
	unsigned int* actions2=&(actions1[Sy]);	// cim vynasobit pivotni radek, 'minS' cisel
	unsigned int* actions3=&(actions2[min(Sx,Sy)]);	// multiplikatory, 'Sx*Sy' cisel
	//cout << "modul = " << modul << endl;
	int tid=0;
	int bdim=1;
	int minS=min( min(Sx, Sy), min(N-Sx*sx, N-Sy*sy) );
	int pod_diag_x=Sy*sy+Sy-Sx*sx;
	// p_mat - pomocn� podmatice, max velikost Sx*Sy
	cout << "actions1: ";
	vypsat_vys<unsigned int>(Sy, actions1, NULL);
	cout << endl << "actions2: ";
	vypsat_vys<unsigned int>(min(Sx,Sy), actions2, NULL);
	cout << endl << "actions3: ";
	vypsat_vys<unsigned int>(min(Sx,Sy)*Sy, actions3, NULL);
	cout << endl << "|||||||||||||||||||||" << endl;

	vypsat_mat<unsigned int>(Sx, Sy, s_mat, NULL);
	cout << "-------------";
	if( !(zpusob & PODMATICE_12) )
		vypsat_mat<unsigned int>(Sx, Sy, p_mat, NULL);
	cout << "=============";
	
	for(int isloupec=0;(isloupec<min(Sx,Sy));isloupec++)
	{
		int gdiag=Sx*sx+isloupec;
		int sdiagy;
		unsigned int* pom_mat;
		bool is_podm3;
		if( !(zpusob & PODMATICE_12) )
		{
			// podmatice4
			cout << "PODMATICE 4 - ";
			sdiagy=isloupec;
			pom_mat=p_mat;
			is_podm3 = true;
		}else
		{
			// podmatice2
			cout << "PODMATICE 2 - ";
			sdiagy=gdiag-sy*Sy;	// index pivotniho radku
			pom_mat=s_mat;
			is_podm3 = false;
			// TODO: deleni: radek sdiag na '1'
			if( zpusob & ZPUSOB_S_DELENIM ) { }
		}
		cout << "isloupec=" << isloupec << endl;
		// -------------------
		unsigned int a_pp = actions2[isloupec];
		for(int iY=0;iY<Sy;iY++)
		{
			if( is_podm3 || iY!=isloupec )	// neupravuji pivotni radek pokud je podmatice1
			{
				unsigned int a_py = actions3[isloupec*Sy+iY];
				cout << "SAVE(" << a_pp << ", " << a_py << ")" << endl;
				for(int iX=0;iX<Sx;iX++)
				{
					unsigned int a_xy = s_mat[get_index(iX, iY, Sx)];
					unsigned int a_xp = pom_mat[get_index(iX, isloupec, Sx)];
					//cout << "  " << a_xy << " * " << a_pp << " - " << a_xp << " * " << a_py << endl;
					if(zpusob & ZPUSOB_S_DELENIM)
					{
						s_mat[get_index(iX, iY, Sx)] = elem_uprava_s_delenim(modul, a_xy, a_xp, a_py);
					}else
					{
						s_mat[get_index(iX, iY, Sx)] = elem_uprava_bez_deleni(modul, a_xy, a_pp, a_xp, a_py);
					}
				}
			}else
			{
			}
		}

	}
	
	vypsat_mat<unsigned int>(Sx, Sy, s_mat, NULL);
	cout << "-------------";
}
// S DELENIM
void compute_podmatice13(int N, unsigned int modul, int sx, int sy, int Sx, int Sy, unsigned int* s_mat, unsigned int* actions, unsigned int zpusob)
{
	// podmatice s_mat: |-- podmatice, kterou pocitam (Sx*Sy cisel) --|
	// podmatice p_mat: |-- podmatice, kterou potrebuji (az Sx^2 cisel) --|
	unsigned int* p_mat=&(s_mat[Sx*Sy]);
	unsigned int* actions1=&(actions[1]);	// indexy pivotnich radku, permutace radku, 'Sy' cisel
	unsigned int* actions2=&(actions1[Sy]);	// cim vynasobit pivotni radek, 'minS' cisel
	unsigned int* actions3=&(actions2[min(Sx,Sy)]);	// multiplikatory, 'Sx*Sy' cisel
	//cout << "modul = " << modul << endl;
	for(int i=0;i<(Sx*Sy+Sx);i++)
	{
		actions[i]=modul;
	}
	int tid=0;
	int bdim=1;
	int minS=min( min(Sx, Sy), min(N-Sx*sx, N-Sy*sy) );
	int pod_diag_x=Sy*sy+Sy-Sx*sx;
	// p_mat - pomocn� podmatice, max velikost Sx*Sy
	
	vypsat_mat<unsigned int>(Sx, Sy, s_mat, NULL);
	cout << "-------------";
	if( !(zpusob & PODMATICE_12) )
		vypsat_mat<unsigned int>(Sx, Sy, p_mat, NULL);
	cout << "=============";
// \FOR{$p$ := $1$ do $Sx$}
	//for(int isloupec=0;(isloupec<Sx)&&(isloupec<pod_diag_x);isloupec++)
	for(int isloupec=0;(isloupec<min(Sx,Sy));isloupec++)
	{
		// najit g_diagonalu ve sloupci 'isloupec'
		int gdiag=min(Sx,Sy)*sx+isloupec;
		if(gdiag>=N) continue;
		int sdiagy;
		unsigned int* pom_mat;
		bool is_podm3;
		for(int i=0;i<Sy;i++) actions1[i]=i;
		
		
		if( (sy*Sy<=gdiag) && (gdiag<(sy+1)*Sy) )	// diagonalni prvek je v 'isloupec'-tem sloupci v aktualni podmatici
		{
			// podmatice1
			cout << "PODMATICE 1 - ";
			sdiagy=gdiag-sy*Sy;	// index pivotniho radku
			pom_mat=s_mat;
			is_podm3 = false;
			// TODO: deleni: radek sdiag na '1'
			if( zpusob & ZPUSOB_S_DELENIM ) { }
		}else	// 'isloupec'-ty sloupec v podmatici je pod nebo nad diagonalnim prvkem
		{
			// podmatice3
			cout << "PODMATICE 3 - ";
			sdiagy=isloupec;
			pom_mat=p_mat;
			is_podm3 = true;
		}
		cout << "isloupec=" << isloupec << endl;
		// -------------------
		unsigned int a_pp = pom_mat[get_index(isloupec, sdiagy, Sx)];
		actions2[isloupec]=a_pp;
		cout << "a_pp actions2[" << isloupec << "]=" << a_pp << endl;
		for(int iY=0;iY<Sy;iY++)
		{
			if( is_podm3 || iY!=isloupec )	// neupravuji pivotni radek pokud je podmatice1
			{
				unsigned int a_py = s_mat[get_index(sdiagy, iY, Sx)];
				// TODO: ulozit a_pp, a_py
				actions3[isloupec*Sy+iY]=a_py;
				cout << "a_py actions3[" << isloupec*Sy+iY << "]=" << a_py << endl;
				//cout << "SAVE(" << a_pp << ", " << a_py << ")" << endl;
				for(int iX=0;iX<Sx;iX++)
				{
					unsigned int a_xy = s_mat[get_index(iX, iY, Sx)];
					unsigned int a_xp = pom_mat[get_index(iX, sdiagy, Sx)];
					//cout << "  " << a_xy << " * " << a_pp << " - " << a_xp << " * " << a_py << endl;
					if(zpusob & ZPUSOB_S_DELENIM)
					{
						s_mat[get_index(iX, iY, Sx)] = elem_uprava_s_delenim(modul, a_xy, a_xp, a_py);
					}else
					{
						s_mat[get_index(iX, iY, Sx)] = elem_uprava_bez_deleni(modul, a_xy, a_pp, a_xp, a_py);
					}
				}
			}else
			{
				actions3[isloupec*Sy+iY]=0;
				cout << "a_py actions3[" << isloupec*Sy+iY << "]=0" << endl;
			}
		}
		
		
		
	}
	vypsat_mat<unsigned int>(Sx, Sy, s_mat, NULL);
	cout << "-------------";
	cout << "actions1: ";
	vypsat_vys<unsigned int>(Sy, actions1, NULL);
	cout << endl << "actions2: ";
	vypsat_vys<unsigned int>(min(Sx,Sy), actions2, NULL);
	cout << endl << "actions3: ";
	vypsat_vys<unsigned int>(min(Sx,Sy)*Sy, actions3, NULL);
	cout << endl << "|||||||||||||||||||||" << endl;
	//*/
	
}
/* celou matici rozdelim do obdelnikovych "podmatic", ktere budu postupne nahravat do sdilene pameti a pocitat
 * podmatice nemusi byt nutne ctvercova
 * zpusob zpracovani: 1, 2, 3, 4
 */
void GJE_podmatice(int N, unsigned int modul, unsigned int* m_matice, unsigned int* m_prava_strana, unsigned int* m_vys_jmenovatel, unsigned int zpusob)
{
	int Sx=3;
	int Sy=3;
	int Smin=min(Sx, Sy);
	unsigned int* s_matice=(unsigned int*)malloc(Sx*(Sy+Sx)*sizeof(unsigned int));
	unsigned int* actions=(unsigned int*)malloc((Sx*Sy+Smin+1)*sizeof(unsigned int));

// \FOR{$p$ := $1$ do $\lceil\frac{N}{\min(S_x, S_y)}\rceil$}
	for(int ipivot=0;ipivot<ceil((double)N/min(Sx,Sy));ipivot++)
	{
		// DEBUG
		cout << endl << ipivot << endl;
	// \STATE \COMMENT{zpracovani radku, kde je Z=1}
	// \STATE nacist a spocitat $podmatice_{pp}$ \COMMENT{Z=1}
		int Py=ipivot*min(Sx,Sy)/Sy;

		// TODO: tady kdyz ipivot==1, jak nacita??
		copy_podmatice(N, ipivot, ipivot, Py, Sx, Sy, s_matice, m_matice, m_prava_strana, COPY_MAT_B_GLOB_TO_A_SH);
		system("cls");
		vypsat_mat<unsigned int>(N, N, m_matice, m_prava_strana);
		// todo: compute_podmatice1
		compute_podmatice13(N, modul, ipivot, Py, Sx, Sy, s_matice, actions, zpusob | PODMATICE_12);
		//vypsat_mat<unsigned int>(Sx, Sy, s_matice, NULL);
		copy_podmatice(N, ipivot, ipivot, Py, Sx, Sy, s_matice, m_matice, m_prava_strana, COPY_MAT_A_SH_TO_B_GLOB);
		vypsat_mat<unsigned int>(N, N, m_matice, m_prava_strana);
	// \FOR{$x$ := $p+1$ do $\lceil\frac{N+1}{S_x}\rceil$}
		for(int x=ipivot+1;x<ceil((double)(N+1)/Sx);x++)
		{
		// \STATE nacist a aplikovat operace v $actions$ na $podmatice_{xp}$ \COMMENT{Z=2}
			copy_podmatice(N, ipivot, x, Py, Sx, Sy, s_matice, m_matice, m_prava_strana, COPY_MAT_B_GLOB_TO_A_SH);
			system("cls");
			vypsat_mat<unsigned int>(N, N, m_matice, m_prava_strana);
			// todo: compute_podmatice2
			//for(int i=0;i<Sx*Sy;i++) s_matice[i]=2;
			compute_podmatice24(N, modul, x, Py, Sx, Sy, s_matice, actions, zpusob | PODMATICE_12);
			copy_podmatice(N, ipivot, x, Py, Sx, Sy, s_matice, m_matice, m_prava_strana, COPY_MAT_A_SH_TO_B_GLOB);
		vypsat_mat<unsigned int>(N, N, m_matice, m_prava_strana);
		}
	//\ENDFOR
	// \STATE \COMMENT{zpracovani ostatnich radku}
	// \FOR{$y$ := $1$ do $\lceil\frac{N}{S_y}\rceil$}
		for(int y=0;y<ceil((double)N/Sy);y++)
		{
		// \IF{$y$ != $p$}
			if(y!=Py)
			{
				// TODO: nacitani p_matice: 1) nacitat v jedne funkci spolu s s_matice (problem: rozlisovani kdy nacitat a kdy ne)
				//                          2) nacitat ve fci copy_podmatice (nebo nejake spec. fci), problem s umistenim ve velke matici
				// int Py=max(0, min(Sx, Sy*y-Sx*ipivot));


			// \STATE nacist a vynulovat $podmatice_{py}$; \COMMENT{Z=3}
				copy_podmatice(N, ipivot, ipivot, y, Sx, Sy, s_matice, m_matice, m_prava_strana, COPY_MAT_B_GLOB_TO_A_SH);
				copy_podmatice(N, ipivot, ipivot, Py, Sx, Sy, &(s_matice[Sx*Sy]), m_matice, m_prava_strana, COPY_MAT_B_GLOB_TO_A_SH);	// todo: nenacitat prvky, ktere uz jsou v s_matice
				system("cls");
				vypsat_mat<unsigned int>(N, N, m_matice, m_prava_strana);
				// todo: compute_podmatice3
				//for(int i=0;i<Sx*Sy;i++) s_matice[i]=3;
				compute_podmatice13(N, modul, ipivot, y, Sx, Sy, s_matice, actions, zpusob);
				copy_podmatice(N, ipivot, ipivot, y, Sx, Sy, s_matice, m_matice, m_prava_strana, COPY_MAT_A_SH_TO_B_GLOB);
		vypsat_mat<unsigned int>(N, N, m_matice, m_prava_strana);
			// \FOR{$x$ := $p+1$ do $\lceil\frac{N+1}{S_x}\rceil$}
				for(int x=ipivot+1;x<ceil((double)(N+1)/Sx);x++)
				{
				// \STATE nacist a aplikovat operace v $actions$ na $podmatice_{xy}$; \COMMENT{Z=4}
					copy_podmatice(N, ipivot, x, y, Sx, Sy, s_matice, m_matice, m_prava_strana, COPY_MAT_B_GLOB_TO_A_SH);
					copy_podmatice(N, ipivot, x, Py, Sx, Sy, &(s_matice[Sx*Sy]), m_matice, m_prava_strana, COPY_MAT_B_GLOB_TO_A_SH);
					system("cls");
					vypsat_mat<unsigned int>(N, N, m_matice, m_prava_strana);
					// todo: compute_podmatice4
					compute_podmatice24(N, modul, x, Py, Sx, Sy, s_matice, actions, zpusob);
					//for(int i=0;i<Sx*Sy;i++) s_matice[i]=4;
					copy_podmatice(N, ipivot, x, y, Sx, Sy, s_matice, m_matice, m_prava_strana, COPY_MAT_A_SH_TO_B_GLOB);
		vypsat_mat<unsigned int>(N, N, m_matice, m_prava_strana);
				}
			// \ENDFOR
			}
		// \ENDIF
		}
		// DEBUG
		vypsat_mat<unsigned int>(N, N, m_matice, m_prava_strana);
	//\ENDFOR
	}
	/*if( !(zpusob & ZPUSOB_S_DELENIM) )
	{
		unsigned long long pom;
		for(int i=0;i<N;i++)
		{
			pom = m_prava_strana[i];
			pom *= compute_inverse_eukleides(m_matice[get_index(i, i, N)], modul);
			pom %= modul;
			m_prava_strana[i] = (unsigned int)pom;
			m_matice[get_index(i, i, N)]=1;
		}
	}*/
//\ENDFOR
}

unsigned int compute_inverse(unsigned int cislo, unsigned int modul)
{
	// TODO: pouzit eukliduv alg.
	unsigned long long inv=cislo;
	unsigned int i=1;
	while(i<modul)
	{
		if( inv==1 )
		{
			return i;
		}
		inv+=cislo;
		inv%=modul;
		i++;
	}
	return (unsigned int)0;
}

unsigned int compute_inverse_eukleides(unsigned int cislo, unsigned int modul)
{
	unsigned int a, b, a1, a2, q, r;
	a = cislo;
	b = modul;
	a1 = 0;
	a2 = 1;
	int plus = 1;

	while( b!=0 )
	{
		q = a / b;
		r = a % b;
		a = b;
		b = r;
		r = a1;
		a1 = a2 + r*q;
		a2 = r;
		plus=-plus;
	}
	if( a==1 )
	{
		if( 0<plus )
		{
			return (unsigned int)a2;
		}else
		{
			return (unsigned int)(modul-a2);
		}
	}
	return (unsigned int)0;
}

int get_index(int X, int Y, int N)	// SLOUPEC, RADEK
{
	return Y*N+X;
}
