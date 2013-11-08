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
		m1--;		// odecist a pak pricist 1, aby m1%modul=0 => m1=0 a modul-m1=modul-0=modul
		m1 %= modul;
		m1++;
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
		m1--;		// odecist a pak pricist 1, aby m1%modul=0 => m1=0 a modul-m1=modul-0=modul
		m1 %= modul;
		m1++;
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
			// v 'ipivot'-tem radku na diagonále je nula => vymena s jinym radkem
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
				// v 'ipivot'-tem radku na diagonále je nula => vymena s jinym radkem
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
 * gx, gy - globalni souradnice prvku, ktery je v podmatici na souradnicich [0;0], \in [0; N)
 * mat_A, mat_B - zdrojova nebo cilova adresa
 */
void copy_podmatice(int N, int gx, int gy, int Sx, int Sy, unsigned int* mat_A, unsigned int* mat_B, unsigned int* prava_str, int copy_to)
{
	int tid=0;
	int bdim=1;
	int itid=tid;
	unsigned int a;
	bool bez_deleni = !!(copy_to & COPY_MAT_BEZ_DELENI);
	copy_to &= ~COPY_MAT_BEZ_DELENI;
	
	while(itid<Sy)
	{
		int glob_y=gy+itid;
		for(int glob_x=gx;glob_x<gx+Sx;glob_x++)
		{
			int shared_x=glob_x-gx;
			int shared_y=glob_y-gy;
	
			if(glob_x<=(N+1) && glob_y<N)
			{
				if(glob_x<N)
				{
					switch(copy_to)
					{
					case COPY_MAT_A_SH_TO_B_GLOB:
						a = mat_A[get_index(shared_x, shared_y, Sx)];
						mat_B[get_index(glob_x, glob_y, N)] = a;
						break;
					case COPY_MAT_B_GLOB_TO_A_SH:
						a = mat_B[get_index(glob_x, glob_y, N)];
						mat_A[get_index(shared_x, shared_y, Sx)] = a;
						break;
					}
				}
				if(glob_x==N)
				{
					switch(copy_to)
					{
					case COPY_MAT_A_SH_TO_B_GLOB:
						a = mat_A[get_index(shared_x, shared_y, Sx)];
						prava_str[glob_y] = a;
						break;
					case COPY_MAT_B_GLOB_TO_A_SH:
						a = prava_str[glob_y];
						mat_A[get_index(shared_x, shared_y, Sx)] = a;
						break;
					}
				}
				if(bez_deleni && (glob_x==(N+1)))
				{
					switch(copy_to)
					{
					case COPY_MAT_A_SH_TO_B_GLOB:
						a = mat_A[get_index(shared_x, shared_y, Sx)];
						mat_B[get_index(glob_y, glob_y, N)] = a;
						break;
					case COPY_MAT_B_GLOB_TO_A_SH:
						a = mat_B[get_index(glob_y, glob_y, N)];
						mat_A[get_index(shared_x, shared_y, Sx)] = a;
						break;
					}
				}
			}else
			{
				if(copy_to == COPY_MAT_B_GLOB_TO_A_SH)
				{
					//if( sx==sy && ix==itid )
					//mat_A[get_index(shared_x, shared_y, Sx)] = 1;
					//else
					mat_A[get_index(shared_x, shared_y, Sx)] = 0;
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
		// \STATE \COMMENT {vydelit celý $p$-tý rádek císlem $a_{pp}$, v $p$-tém sloupci na diagonále bude císlo 1}
		// \STATE uložit "jedna lomeno  $a_{pp}$" do $actions$
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
			// \STATE uložit $a_py$ do $actions$
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
void compute_podmatice24(int N, unsigned int modul, int pivot_x, int Sx, int Sy, unsigned int* s_mat, unsigned int* actions, unsigned int zpusob)
{
	// podmatice s_mat: |-- podmatice, kterou pocitam (Sx*Sy cisel) --|
	// podmatice p_mat: |-- podmatice, kterou potrebuji (az Sx^2 cisel) --|
	int minS=min(Sx,Sy);
	unsigned int* p_mat=&(s_mat[Sx*Sy]);		// ctvercova podmatice s pivotnimi radky, Sx*Smin cisel
	unsigned int* actions1=actions;				// indexy pivotnich radku, permutace radku, 'minS' cisel
	unsigned int* actions2=&(actions1[minS]);	// cim vynasobit nebo vydelit pivotni radek; 'minS' cisel
	unsigned int* actions3=&(actions2[minS]);	// multiplikatory, 'Smin*Sy' cisel
	//cout << "modul = " << modul << endl;
	int tid=0;
	int bdim=1;
	// p_mat - pomocná podmatice, max velikost Sx*Sy
	cout << "actions1: ";
	vypsat_vys<unsigned int>(minS, actions1, NULL);
	cout << endl << "actions2: ";
	vypsat_vys<unsigned int>(minS, actions2, NULL);
	cout << endl << "actions3: ";
	vypsat_vys<unsigned int>(minS*Sy, actions3, NULL);
	cout << endl << "|||||||||||||||||||||" << endl;

	
	for(int isloupec=0;(isloupec<minS);isloupec++)
	{
		vypsat_mat<unsigned int>(Sx, Sy, s_mat, NULL);
		cout << "-------------";
		if( !(zpusob & PODMATICE_12) )
			vypsat_mat<unsigned int>(Sx, Sy, p_mat, NULL);
		cout << "=============";

		int gdiag=pivot_x+isloupec;
		unsigned int* pom_mat;
		bool is_podm3;
		if( (zpusob & PODMATICE_12) )	// diagonalni prvek je v 'isloupec'-tem sloupci v aktualni podmatici
		{
			// podmatice2
			cout << "PODMATICE 2 - ";
			pom_mat=s_mat;
			is_podm3 = false;
			// deleni: radek sdiag na '1'
			if( zpusob & ZPUSOB_S_DELENIM )
			{
				unsigned int a_pp_inv=actions2[isloupec];
				int x=tid;
				while(x<Sx)
				{
					unsigned long long pom = s_mat[get_index(x, isloupec, Sx)];
					pom *= a_pp_inv;
					pom %= modul;
					s_mat[get_index(x, isloupec, Sx)]=(unsigned int)pom;
					x+=bdim;
				}
			}
		}else	// 'isloupec'-ty sloupec v podmatici je pod nebo nad diagonalnim prvkem
		{
			// podmatice4
			cout << "PODMATICE 4 - ";
			pom_mat=p_mat;
			is_podm3 = true;
		}
		cout << "isloupec=" << isloupec << endl;
		// -------------------
		unsigned int a_pp = actions2[isloupec];
		for(int iY=0;iY<Sy;iY++)
		{
			if( is_podm3 || iY!=isloupec )	// neupravuji pivotni radek pokud je podmatice1
			{
				unsigned int a_py = actions3[isloupec*Sy+iY];
				for(int iX=0;iX<Sx;iX++)
				{
					unsigned int a_xy = s_mat[get_index(iX, iY, Sx)];
					unsigned int a_xp = pom_mat[get_index(iX, isloupec, Sx)];
					//cout << "  a_xy := " << a_xy << " * " << a_pp << " - " << a_xp << " * " << a_py << endl;
					if(zpusob & ZPUSOB_S_DELENIM)
					{
						s_mat[get_index(iX, iY, Sx)] = elem_uprava_s_delenim(modul, a_xy, a_xp, a_py);
					}else
					{
						s_mat[get_index(iX, iY, Sx)] = elem_uprava_bez_deleni(modul, a_xy, a_pp, a_xp, a_py);
					}
				}
			}
		}
	}
	
	vypsat_mat<unsigned int>(Sx, Sy, s_mat, NULL);
	cout << "-------------";
}
// S DELENIM
void compute_podmatice13(int N, unsigned int modul, int pivot_x, int Sx, int Sy, unsigned int* s_mat, unsigned int* actions, unsigned int zpusob)
{
	// podmatice s_mat: |-- podmatice, kterou pocitam (Sx*Sy cisel) --|
	// podmatice p_mat: |-- podmatice, kterou potrebuji (az Sx^2 cisel) --|
	int minS=min(Sx,Sy);
	unsigned int* p_mat=&(s_mat[Sx*Sy]);		// ctvercova podmatice s pivotnimi radky, Sx*Sx cisel
	unsigned int* actions1=actions;				// indexy pivotnich radku, permutace radku, 'minS' cisel
	unsigned int* actions2=&(actions1[minS]);	// cim vynasobit nebo vydelit pivotni radek; 'minS' cisel
	unsigned int* actions3=&(actions2[minS]);	// multiplikatory, 'Sx*Sy' cisel
	//cout << "modul = " << modul << endl;
	int tid=0;
	int bdim=1;
	// p_mat - pomocná podmatice, max velikost Sx*Sy
	
	vypsat_mat<unsigned int>(Sx, Sy, s_mat, NULL);
	cout << "-------------";
	if( !(zpusob & PODMATICE_12) )
		vypsat_mat<unsigned int>(Sx, Sy, p_mat, NULL);
	cout << "=============";
// \FOR{$p$ := $1$ do $Sx$}
	for(int isloupec=0;(isloupec<minS);isloupec++)
	{
		// najit g_diagonalu ve sloupci 'isloupec'
		unsigned int* pom_mat;
		bool is_podm3;
		// TODO: permutace radku, aby na diagonale nebyla nula
		actions1[isloupec]=isloupec;
		if( zpusob & PODMATICE_12 )	// diagonalni prvek je v 'isloupec'-tem sloupci v aktualni podmatici
		{
			// podmatice1
			cout << "PODMATICE 1 - ";
			pom_mat=s_mat;
			is_podm3 = false;
			// deleni: radek sdiag na '1'
			if( zpusob & ZPUSOB_S_DELENIM )
			{
				unsigned int a_pp_inv=compute_inverse_eukleides(s_mat[get_index(isloupec, isloupec, Sx)], modul);
				actions2[isloupec]=a_pp_inv;
				int x=tid;
				while(x<Sx)
				{
					unsigned long long pom = s_mat[get_index(x, isloupec, Sx)];
					pom *= a_pp_inv;
					pom %= modul;
					s_mat[get_index(x, isloupec, Sx)]=(unsigned int)pom;
					x+=bdim;
				}
			}
		}else	// 'isloupec'-ty sloupec v podmatici je pod nebo nad diagonalnim prvkem
		{
			// podmatice3
			cout << "PODMATICE 3 - ";
			pom_mat=p_mat;
			is_podm3 = true;
		}

		//vypsat_mat<unsigned int>(Sx, Sy, s_mat, NULL);
		// -------------------
		unsigned int a_pp=1;
		if( !(zpusob & ZPUSOB_S_DELENIM) )
		{
			a_pp = pom_mat[get_index(isloupec, isloupec, Sx)];
			actions2[isloupec]=a_pp;
		}
		for(int iY=0;iY<Sy;iY++)
		{
			unsigned int a_py=0;
			if( is_podm3 || iY!=isloupec )	// neupravuji pivotni radek pokud je podmatice1
			{
				a_py = s_mat[get_index(isloupec, iY, Sx)];
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
			}
			actions3[isloupec*Sy+iY]=a_py;
			cout << "actions3[" << (isloupec*Sy+iY) << "] = " << a_py << endl;
		}
		
		vypsat_mat<unsigned int>(Sx, Sy, s_mat, NULL);
		
	}
	cout << "-------------";
	cout << "actions1: ";
	vypsat_vys<unsigned int>(minS, actions1, NULL);
	cout << endl << "actions2: ";
	vypsat_vys<unsigned int>(minS, actions2, NULL);
	cout << endl << "actions3: ";
	vypsat_vys<unsigned int>(minS*Sy, actions3, NULL);
	cout << endl << "|||||||||||||||||||||" << endl;
	//*/
	
}
/* celou matici rozdelim do obdelnikovych "podmatic", ktere budu postupne nahravat do sdilene pameti a pocitat
 * podmatice nemusi byt nutne ctvercova
 * zpusob zpracovani: 1, 2, 3, 4
 */
void GJE_podmatice(int N, unsigned int modul, unsigned int* m_matice, unsigned int* m_prava_strana, unsigned int zpusob)
{
	int Sx=3;	// TODO: s nastavenim Sx=3, Sy=4 to vyhazuje chybu
	int Sy=5;
	int Smin=min(Sx, Sy);
	int size_s_matice=Sx*Sy+Sx*Smin;
	int size_actions=(Smin)*Sy+2*Smin;
	int size_actions1=3*size_actions;
	unsigned int mask_copy = 0x0000;
	if( !(zpusob & ZPUSOB_S_DELENIM) ) mask_copy |= COPY_MAT_BEZ_DELENI; 
	unsigned int* s_matice=(unsigned int*)malloc(size_s_matice*sizeof(unsigned int));
	unsigned int* actions=(unsigned int*)malloc(size_actions1*sizeof(unsigned int));
	int konst=1000000;
	for(int i=0;i<size_actions1;i++) actions[i]=konst;

	unsigned int* p_matice=&(s_matice[Sx*Sy]);

// \FOR{$p$ := $1$ do $\lceil\frac{N}{\min(S_x, S_y)}\rceil$}
	for(int ipivot=0;ipivot<N;ipivot+=Smin)
	{
	// \STATE \COMMENT{zpracovani radku, kde je Z=1}
	// \STATE nacist a spocitat $podmatice_{pp}$ \COMMENT{Z=1}
		int Py=ipivot;
	for(int i=0;i<size_actions1;i++) cout << actions[i] << " ";

		copy_podmatice(N, ipivot, Py, Sx, Sy, s_matice, m_matice, m_prava_strana, COPY_MAT_B_GLOB_TO_A_SH);
		// todo: compute_podmatice1
		compute_podmatice13(N, modul, ipivot, Sx, Sy, s_matice, actions, zpusob | PODMATICE_12);
		copy_podmatice(N, ipivot, Py, Sx, Sy, s_matice, m_matice, m_prava_strana, COPY_MAT_A_SH_TO_B_GLOB);
		vypsat_mat<unsigned int>(N, N, m_matice, m_prava_strana);
	// \FOR{$x$ := $p+1$ do $\lceil\frac{N+1}{S_x}\rceil$}
		for(int x=ipivot+Sx;x<N+1;x+=Sx)
		{
		// \STATE nacist a aplikovat operace v $actions$ na $podmatice_{xp}$ \COMMENT{Z=2}
			copy_podmatice(N, x, Py, Sx, Sy, s_matice, m_matice, m_prava_strana, COPY_MAT_B_GLOB_TO_A_SH);
			// todo: compute_podmatice2
			compute_podmatice24(N, modul, x, Sx, Sy, s_matice, actions, zpusob | PODMATICE_12);
			copy_podmatice(N, x, Py, Sx, Sy, s_matice, m_matice, m_prava_strana, COPY_MAT_A_SH_TO_B_GLOB);
		vypsat_mat<unsigned int>(N, N, m_matice, m_prava_strana);
		}
	//\ENDFOR
	// \STATE \COMMENT{zpracovani ostatnich radku}
	// \FOR{$y$ := $1$ do $\lceil\frac{N}{S_y}\rceil$}
		for(int y=0;y<N;y+=Sy)
		{
		// \IF{$y$ != $p$}
			//if(y!=Py)
			{
				// TODO: nacitani p_matice: 1) nacitat v jedne funkci spolu s s_matice (problem: rozlisovani kdy nacitat a kdy ne)
				//                          2) nacitat ve fci copy_podmatice (nebo nejake spec. fci), problem s umistenim ve velke matici
				// int Py=max(0, min(Sx, Sy*y-Sx*ipivot));
				int Py1, Sy1=0;
				int Py2, Sy2=0;
				if(y+Sy<=ipivot)
				{
					// cela podmatice je nad diagonalou ve velke matici
					Sy1=Sy;
					Py1=y;
				}else if(y<ipivot)
				{
					// cast bude nad a cast pod diagonalou
					Sy1=ipivot-y;
					Py1=y;
					Sy2=Sy-Sy1;
					Py2=y+Sy1+Sy;
				}else	// y>=ipivot
				{
					// cela podmatice bude pod diagonalou
					Sy1=Sy;
					Py1=y+Sy;
				}
				if(Py1>=N) break;
			// \STATE nacist a vynulovat $podmatice_{py}$; \COMMENT{Z=3}
				copy_podmatice(N, ipivot, Py1, Sx, Sy1, s_matice, m_matice, m_prava_strana, (COPY_MAT_B_GLOB_TO_A_SH | mask_copy) );
				if(Sy2>0)
					copy_podmatice(N, ipivot, Py2, Sx, Sy2, &(s_matice[Sx*Sy1]), m_matice, m_prava_strana, (COPY_MAT_B_GLOB_TO_A_SH | mask_copy) );
				// todo: nenacitat prvky, ktere uz jsou v s_matice
				copy_podmatice(N, ipivot, Py, Sx, Smin, p_matice, m_matice, m_prava_strana, (COPY_MAT_B_GLOB_TO_A_SH | mask_copy) );
				// todo: compute_podmatice3
				compute_podmatice13(N, modul, ipivot, Sx, Sy, s_matice, actions, zpusob);

				copy_podmatice(N, ipivot, Py1, Sx, Sy1, s_matice, m_matice, m_prava_strana, (COPY_MAT_A_SH_TO_B_GLOB | mask_copy) );
				if(Sy2>0) copy_podmatice(N, ipivot, Py2, Sx, Sy2, &(s_matice[Sx*Sy1]), m_matice, m_prava_strana, (COPY_MAT_A_SH_TO_B_GLOB | mask_copy) );
				vypsat_mat<unsigned int>(N, N, m_matice, m_prava_strana);
			// \FOR{$x$ := $p+1$ do $\lceil\frac{N+1}{S_x}\rceil$}
				int xUp=N;
				if( !(zpusob & ZPUSOB_S_DELENIM) ) xUp++;
				for(int x=ipivot+Sx;x<xUp;x+=Sx)
				{
				// \STATE nacist a aplikovat operace v $actions$ na $podmatice_{xy}$; \COMMENT{Z=4}
					copy_podmatice(N, x, Py1, Sx, Sy1, s_matice, m_matice, m_prava_strana, (COPY_MAT_B_GLOB_TO_A_SH | mask_copy) );
					if(Sy2>0) copy_podmatice(N, x, Py2, Sx, Sy2, &(s_matice[Sx*Sy1]), m_matice, m_prava_strana, (COPY_MAT_B_GLOB_TO_A_SH | mask_copy) );
					copy_podmatice(N, x, Py, Sx, Smin, p_matice, m_matice, m_prava_strana, (COPY_MAT_B_GLOB_TO_A_SH | mask_copy) );
					// todo: compute_podmatice4
					compute_podmatice24(N, modul, x, Sx, Sy, s_matice, actions, zpusob);
					copy_podmatice(N, x, Py1, Sx, Sy1, s_matice, m_matice, m_prava_strana, (COPY_MAT_A_SH_TO_B_GLOB | mask_copy) );
					if(Sy2>0) copy_podmatice(N, x, Py2, Sx, Sy2, &(s_matice[Sx*Sy1]), m_matice, m_prava_strana, (COPY_MAT_A_SH_TO_B_GLOB | mask_copy) );
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
	cout << "size_actions = " << size_actions << endl;

	for(int i=0;i<size_actions1;i++)
	{
		if(actions[i]==konst)
		{
			cout << i << endl;
			break;
		}
	}
	if( !(zpusob & ZPUSOB_S_DELENIM) )
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
	}
		vypsat_mat<unsigned int>(N, N, m_matice, m_prava_strana);
//\ENDFOR
#ifndef _DEBUG
	//free(s_matice);
	//free(actions);
#endif
}

void GJE_radky(int N, unsigned int modul, unsigned int* m_matice, unsigned int* m_prava_strana, unsigned int zpusob)
{
	int size_sh_mem=N+1;
	unsigned int* sh_mem=(unsigned int*)malloc(size_sh_mem*sizeof(unsigned int));
// \FOR{$p$ := $1$ do $N$}
	for(int ipivot=0;ipivot<N;ipivot++)
	{
	// \STATE \COMMENT{nalezeni radku s nenulovou hodnotou prvku $[p;q]$, kde $p<=q$}
		int q;	// CUDA: 'q' sdilene, pak si musi kazde vlakno vzit svou kopii
		for(int i=ipivot;i<N;i++)
		{
			if(m_matice[get_index(ipivot, i, N)]!=0)
			{
				q=i;
				break;
			}
		}
	// \STATE \COMMENT{priprava pivotniho radku}
	// \STATE nacist prvek $[p;q]$ do sdilene pameti
		// CUDA: shared, tid==0
		unsigned a_pq = m_matice[get_index(ipivot, q, N)];
		sh_mem[ipivot] = a_pq;
		unsigned int a_pq_inv=compute_inverse_eukleides(a_pq, modul);
	// \FOR{$x$ := $p+1$ do $N$}
		for(int iX=ipivot;iX<=N;iX++)	// CUDA: pres tid
		{
		// \STATE nacist, vydelit a ulozit do sdilene pameti
			unsigned long long a;
			if(iX==N) a = m_prava_strana[q];
			else a = m_matice[get_index(iX, q, N)];
			a *= a_pq_inv;
			a %= modul;
			sh_mem[iX] = (unsigned int)a;
		}
	// \ENDFOR
	// \FOR{$y$ := $1$ do $N$}
		for(int iY=0;iY<N;iY++)
		{
			unsigned int a_py = m_matice[get_index(ipivot, iY, N)];
		// \FOR{$x$ := $p+1$ do $N$}
			for(int iX=ipivot;iX<=N;iX++)
			{
			// \IF{$y$ == $q$}
				if(iY == q)
				{
				// \STATE ulozit do globalni pameti prvek $[x;y]=[x;q]$
					if(iX==N) m_prava_strana[iY] = sh_mem[iX];
					else m_matice[get_index(iX, iY, N)] = sh_mem[iX];
			// \ELSE
				}else
				{
				// \STATE upravit prvek $[x;y]$ stejne jako pri nulovani prvku $[p;y]$
					unsigned int a_xy;
					if(iX==N) a_xy = m_prava_strana[iY];
					else a_xy = m_matice[get_index(iX, iY, N)];
					unsigned int a_xp = sh_mem[iX];
					//cout << "  " << a_xy << " * " << a_pp << " - " << a_xp << " * " << a_py << endl;
					if(zpusob & ZPUSOB_S_DELENIM)
					{
						a_xy = elem_uprava_s_delenim(modul, a_xy, a_xp, a_py);
					}else
					{
						a_xy = elem_uprava_bez_deleni(modul, a_xy, a_pq, a_xp, a_py);
					}
					if(iX==N) m_prava_strana[iY] = a_xy;
					else m_matice[get_index(iX, iY, N)] = a_xy;
				}
			// \ENDIF
		// \ENDFOR
			}
		}
	// \ENDFOR
	}
// \ENDFOR
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
