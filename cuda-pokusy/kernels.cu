#include <stdio.h>
#include <cstdio>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include "kernels.h"
#include "time_measure.h"
#include "common.h"



__device__ int cuda_get_index(int X, int Y, int N)	// SLOUPEC, RADEK
{
	return Y*N+X;
}
__device__ unsigned int cuda_compute_inverse_eukleides(unsigned int cislo, unsigned int modul)
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

// elementarni uprava s delenim
__device__ unsigned int cuda_elem_uprava_s_delenim(unsigned int modul, unsigned int a_xy, unsigned int a_xp, unsigned int a_py)
// \STATE $a_{xy} := a_{xy} - a_xp \cdot a_py$
// TODO: merit rychlosti modulovani % a __umulhi
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
__device__ unsigned int cuda_elem_uprava_bez_deleni(unsigned int modul, unsigned int a_xy, unsigned int a_pp, unsigned int a_xp, unsigned int a_py)
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
// elementarni uprava s delenim
__device__ unsigned int cuda_elem_uprava_s_delenim1(unsigned int modul, unsigned int a_xy, unsigned int a_xp, unsigned int a_py)
// \STATE $a_{xy} := a_{xy} - a_xp \cdot a_py$
// TODO: merit rychlosti modulovani % a __umulhi
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
__device__ unsigned int cuda_elem_uprava_bez_deleni1(unsigned int modul, unsigned int a_xy, unsigned int a_pp, unsigned int a_xp, unsigned int a_py)
// \STATE $a_{xy} := a_{xy} \cdot a_pp - a_xp \cdot a_py$
{
	unsigned int x1=a_xy;
	unsigned int y1=a_pp;
	unsigned int x2=a_xp;
	unsigned int y2=a_py;
	/*
 * computes (x1y1 ? x2y2) mod m
13: h1 = uint2float_rz(umul24hi(x1, y1)) * two inlined mul_mod’s
14: h2 = uint2float_rz(umul24hi(x2, y2))
15: l1 = float2uint_rz(fmul_rn(h1, inv1)) * inv1 = 65536.0f/m
16: l2 = float2uint_rz(fmul_rn(h2, inv1)) * mul. and truncate
17: r = mc + umul24(x1, y1) ? umul24(l1,m) * mc = m * 100
18: r = r ? umul24(x2, y2) + umul24(l2,m) * diff. of mul_mod’s
19: * inv2 = 1.0f/m, e23 = (float)(1 << 23)
20: rf = uint2float_rn(r) ? inv2 + e23 * rf = ?r/m
21: r = r ? umul24(float_as_int(rf),m)
22: return (r < 0 ? r + m : r)
*/
	/*float invm=
	h1=__uint2float_rz(__umulhi(x1,y1));
	h2=__uint2float_rz(__umulhi(x2,y2));
	l1=__float2uint_rz(__fmul_rn(h1, invm));
	*/
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
/* nacte/ulozi podmatici z globalni p. do sdilene nebo zpet
 * Sx, Sy - velikost podmatice, mela by se vejit do sdilene pameti
 * sx, sy - souradnice zvolene podmatice v matici, sx \in [0; ceil(N/Sx)]
 * mat_A, mat_B - zdrojova nebo cilova adresa
 */
//#define COPY_MAT_B_GLOB_TO_A_SH	1
//#define COPY_MAT_A_SH_TO_B_GLOB	2
//#define COPY_MAT_A_SH_TO_B_SH 	3
__device__ void cuda_copy_podmatice(int N, int gx, int gy, int Sx, int Sy, unsigned int* mat_A, unsigned int* mat_B, unsigned int* prava_str, int copy_to)
{
	int tid=0;
	int bdim=1;
	int itid=tid;
	unsigned int a;
	
	while(itid<Sy)
	{
		int glob_y=gy+itid;
		for(int glob_x=gx;glob_x<gx+Sx;glob_x++)
		{
			int shared_x=glob_x-gx;
			int shared_y=glob_y-gy;
	
			if(glob_x<=N && glob_y<N)
			{
				if(glob_x<N)
				{
					switch(copy_to)
					{
					case COPY_MAT_A_SH_TO_B_GLOB:
						a = mat_A[cuda_get_index(shared_x, shared_y, Sx)];
						mat_B[cuda_get_index(glob_x, glob_y, N)] = a;
						break;
					case COPY_MAT_B_GLOB_TO_A_SH:
						a = mat_B[cuda_get_index(glob_x, glob_y, N)];
						mat_A[cuda_get_index(shared_x, shared_y, Sx)] = a;
						break;
					}
				}else
				{
					switch(copy_to)
					{
					case COPY_MAT_A_SH_TO_B_GLOB:
						a = mat_A[cuda_get_index(shared_x, shared_y, Sx)];
						prava_str[glob_y] = a;
						break;
					case COPY_MAT_B_GLOB_TO_A_SH:
						a = prava_str[glob_y];
						mat_A[cuda_get_index(shared_x, shared_y, Sx)] = a;
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
					mat_A[cuda_get_index(shared_x, shared_y, Sx)] = 0;
				}
			}
		}
		itid+=bdim;
	}
}
/* 
 * gauss-jordanova eliminace, jednovlaknova, ve while-cyklech, primo na datech ve vstupnim poli, 
 * bez deleni - nasobim oba mergujici radky, po vypoctu kazde bunky se moduluje, 
 * dva pristupy k matici: ipivot prochazi pres matici pres radky/sloupce
 * void gauss_jordan_elim_while(int Sx, int Sy, unsigned int modul, unsigned int* m_matice)
 */
__device__ void gauss_jordan_elim_while_kernel(int Sx, int Sy, unsigned int modul, unsigned int* m_matice, unsigned int zpusob)
{
	int Smin=min(Sx, Sy);
	int tid=threadIdx.x;
	int bdim=blockDim.x;
	int itid;
	for(int ipivot=0;ipivot<Smin;ipivot++)
	{
		__shared__ int novy_pivot;
		__syncthreads();
		if(tid==0)
		{
			novy_pivot=ipivot;
			// deleni nulou => nasobeni inverznim prvkem
			if(m_matice[cuda_get_index(ipivot, ipivot, Sx)]==0)
			{
				// v 'ipivot'-tem radku na diagonále je nula => vymena s jinym radkem
				do{
					novy_pivot++;
				}while(m_matice[cuda_get_index(ipivot, novy_pivot, Sx)]==0 && novy_pivot<Smin);
			}
		}
		__syncthreads();
		// matice je singularni
		if(novy_pivot>=Smin)
		{
			// matice nema v 'ipivot'-tem sloupci nenulovy prvek => je singularni
			//cout << "singularni" << endl;
			itid=tid;
			// singularni matice => vysledky jsou nulove (nepouzitelne)
			//while(itid<=N)
			{
					
				itid+=bdim;
			}
			return;
		}
		// musim prehodit pivotni radek s jinym
		if(novy_pivot>ipivot)
		{
			// vymena radku ipivot a novy_pivot
			itid=tid;
			unsigned int pom;
			while(itid<=Sx)
			{
				pom=m_matice[cuda_get_index(itid, ipivot, Sx)];
				m_matice[cuda_get_index(itid, ipivot, Sx)]=m_matice[cuda_get_index(itid, novy_pivot, Sx)];
				m_matice[cuda_get_index(itid, novy_pivot, Sx)]=pom;
				itid+=bdim;
			}
		}

		__syncthreads();
		unsigned int a_pp;
		if( zpusob & ZPUSOB_S_DELENIM )
		{
			unsigned int a_pp_inv = cuda_compute_inverse_eukleides(m_matice[cuda_get_index(ipivot, ipivot, Sx)], modul);
			// vydelit cely ipivot-ty radek cislem a_pp
			itid=tid;
			while(itid<Sx)
			{
				unsigned long long pom = m_matice[cuda_get_index(itid, ipivot, Sx)];
				pom *= a_pp_inv;
				pom %= modul;
				m_matice[cuda_get_index(itid, ipivot, Sx)]=(unsigned int)pom;

				itid+=bdim;
			}
		}else
		{
			a_pp = m_matice[cuda_get_index(ipivot, ipivot, Sx)];
		}

		//*
		if(zpusob & ZPUSOB_WF)
		{
			itid=tid;
			while(itid<Sy)	// prochazi jednotlive radky
			{
				if(itid!=ipivot)
				{
					unsigned int a_py = m_matice[cuda_get_index(ipivot, itid, Sx)];

					for(int iX=0;iX<Sx;iX++)	// prochazi cisla v i1-tem radku
					{
						unsigned int a_xy = m_matice[cuda_get_index(iX, itid, Sx)];
						unsigned int a_xp = m_matice[cuda_get_index(iX, ipivot, Sx)];
						if( zpusob & ZPUSOB_S_DELENIM )
						{
							m_matice[cuda_get_index(iX, itid, Sx)] = cuda_elem_uprava_s_delenim(modul, a_xy, a_xp, a_py);
						}else
						{
							m_matice[cuda_get_index(iX, itid, Sx)] = cuda_elem_uprava_bez_deleni(modul, a_xy, a_pp, a_xp, a_py);
						}
					}
				}
				itid+=bdim;
			}
		}else
		{
			for(int iY=0;iY<Sy;iY++)	// prochazi jednotlive radky
			{
				if(iY!=ipivot)
				{
					unsigned int a_py = m_matice[cuda_get_index(ipivot, iY, Sx)];
					// DEBUG
					itid=tid;
					while(itid<Sx)	// prochazi cisla v i1-tem radku
					{
						unsigned int a_xy = m_matice[cuda_get_index(itid, iY, Sx)];
						unsigned int a_xp = m_matice[cuda_get_index(itid, ipivot, Sx)];
						if( zpusob & ZPUSOB_S_DELENIM )
						{
							m_matice[cuda_get_index(itid, iY, Sx)] = cuda_elem_uprava_s_delenim(modul, a_xy, a_xp, a_py);
						}else
						{
							m_matice[cuda_get_index(itid, iY, Sx)] = cuda_elem_uprava_bez_deleni(modul, a_xy, a_pp, a_xp, a_py);
						}
						itid+=bdim;
					}
				}
				__syncthreads();
			}
		}
	}
	if( zpusob & ZPUSOB_S_DELENIM )
	{
		unsigned long long pom;
		itid=tid;
		while(itid<Smin)
		{
			pom = m_matice[cuda_get_index(Sx-1, itid, Sx)];
			pom *= cuda_compute_inverse_eukleides(m_matice[cuda_get_index(itid, itid, Sx)], modul);
			pom %= modul;
			m_matice[cuda_get_index(Sx-1, itid, Sx)] = (unsigned int)pom;
			itid+=bdim;
		}
	}
}

__global__ void kernel(int N, int* pole, int cislo)
{
	int tid=threadIdx.x;
	while(tid<N)
	{
		pole[tid]=5;
		tid+=blockDim.x;
	}
}
__global__ void cuda_GJE_while_kernel(int N, unsigned int modul, unsigned int* g_matice, unsigned int* g_prava_strana, unsigned int zpusob)
{
	int Sx=N+1;
	int Sy=N;
	extern __shared__ unsigned int shared_memory[];
	unsigned int* s_mat=&(shared_memory[0]);
	cuda_copy_podmatice(N, 0, 0, Sx, Sy, s_mat, g_matice, g_prava_strana, COPY_MAT_B_GLOB_TO_A_SH);
	gauss_jordan_elim_while_kernel(Sx, Sy, modul, s_mat, zpusob);
	cuda_copy_podmatice(N, 0, 0, Sx, Sy, s_mat, g_matice, g_prava_strana, COPY_MAT_A_SH_TO_B_GLOB);
}

__device__ void cuda_compute_podmatice24(int N, unsigned int modul, int pivot_x, int Sx, int Sy, unsigned int* s_mat, unsigned int* actions, unsigned int zpusob)
{
	// podmatice s_mat: |-- podmatice, kterou pocitam (Sx*Sy cisel) --|
	// podmatice p_mat: |-- podmatice, kterou potrebuji (az Sx^2 cisel) --|
	int minS=min(Sx,Sy);
	unsigned int* p_mat=&(s_mat[Sx*Sy]);
	unsigned int* actions1=actions;				// indexy pivotnich radku, permutace radku, 'minS' cisel
	unsigned int* actions2=&(actions1[minS]);	// cim vynasobit nebo vydelit pivotni radek; 'minS' cisel
	unsigned int* actions3=&(actions2[minS]);	// multiplikatory, 'Sx*Sy' cisel
	//cout << "modul = " << modul << endl;
	int tid=threadIdx.x;
	int bdim=blockDim.x;
	// p_mat - pomocná podmatice, max velikost Sx*Sy
	
	for(int isloupec=0;(isloupec<minS);isloupec++)
	{
		unsigned int* pom_mat;
		bool is_podm3;
		__syncthreads();
		if( (zpusob & PODMATICE_12) )
		{
			// podmatice2
			pom_mat=s_mat;
			is_podm3 = false;
			// deleni: radek sdiag na '1'
			if( zpusob & ZPUSOB_S_DELENIM )
			{
				unsigned int a_pp_inv=actions2[isloupec];
				int x=tid;
				while(x<Sx)
				{
					unsigned long long pom = s_mat[cuda_get_index(x, isloupec, Sx)];
					pom *= a_pp_inv;
					pom %= modul;
					s_mat[cuda_get_index(x, isloupec, Sx)]=(unsigned int)pom;
					x+=bdim;
				}
			}
		}else
		{
			// podmatice4
			pom_mat=p_mat;
			is_podm3 = true;
		}
		// -------------------
		unsigned int a_pp = actions2[isloupec];
		__syncthreads();
		int iY=tid;
		while(iY<Sy)
		{
			if( is_podm3 || iY!=isloupec )	// neupravuji pivotni radek pokud je podmatice1
			{
				unsigned int a_py = actions3[isloupec*Sy+iY];
				for(int iX=0;iX<Sx;iX++)
				{
					unsigned int a_xy = s_mat[cuda_get_index(iX, iY, Sx)];
					unsigned int a_xp = pom_mat[cuda_get_index(iX, isloupec, Sx)];
					//cout << "  " << a_xy << " * " << a_pp << " - " << a_xp << " * " << a_py << endl;
					if(zpusob & ZPUSOB_S_DELENIM)
					{
						s_mat[cuda_get_index(iX, iY, Sx)] = cuda_elem_uprava_s_delenim(modul, a_xy, a_xp, a_py);
					}else
					{
						s_mat[cuda_get_index(iX, iY, Sx)] = cuda_elem_uprava_bez_deleni(modul, a_xy, a_pp, a_xp, a_py);
					}
				}
			}else
			{
			}
			iY+=bdim;
		}
	}
	
}

__device__ void cuda_compute_podmatice13(int N, unsigned int modul, int pivot_x, int Sx, int Sy, unsigned int* s_mat, unsigned int* actions, unsigned int zpusob)
{
	// podmatice s_mat: |-- podmatice, kterou pocitam (Sx*Sy cisel) --|
	// podmatice p_mat: |-- podmatice, kterou potrebuji (az Sx^2 cisel) --|
	int minS=min(Sx,Sy);
	unsigned int* p_mat=&(s_mat[Sx*Sy]);
	unsigned int* actions1=&(actions[0]);				// indexy pivotnich radku, permutace radku, 'minS' cisel
	unsigned int* actions2=&(actions1[minS]);	// cim vynasobit nebo vydelit pivotni radek; 'minS' cisel
	unsigned int* actions3=&(actions2[minS]);	// multiplikatory, 'Sx*Sy' cisel

	int tid=threadIdx.x;
	int bdim=blockDim.x;
	// p_mat - pomocná podmatice, max velikost Sx*Sy
	
// \FOR{$p$ := $1$ do $Sx$}
	for(int isloupec=0;(isloupec<minS);isloupec++)
	{
		// najit g_diagonalu ve sloupci 'isloupec'
		int gdiag=pivot_x+isloupec;
		//if(gdiag>=N) continue;
		int sdiagy=isloupec;	// index pivotniho radku
		unsigned int* pom_mat;
		bool is_podm3;
		// TODO: permutace radku, aby na diagonale nebyla nula
		actions1[isloupec]=isloupec;
	
		__syncthreads();
		if( zpusob & PODMATICE_12 )	// diagonalni prvek je v 'isloupec'-tem sloupci v aktualni podmatici
		{
			// podmatice1
			pom_mat=s_mat;
			is_podm3 = false;
			// deleni: radek sdiag na '1'
			if( (tid==0) && (zpusob & ZPUSOB_S_DELENIM) )
			{
				actions2[isloupec]=cuda_compute_inverse_eukleides(s_mat[cuda_get_index(isloupec, sdiagy, Sx)], modul);
			}
			__syncthreads();
			if( zpusob & ZPUSOB_S_DELENIM )
			{
				unsigned int a_pp_inv=actions2[isloupec];
				int x=tid;
				while(x<Sx)
				{
					unsigned long long pom = s_mat[cuda_get_index(x, sdiagy, Sx)];
					pom *= a_pp_inv;
					pom %= modul;
					s_mat[cuda_get_index(x, sdiagy, Sx)]=(unsigned int)pom;
					x+=bdim;
				}
				
			}
		}else	// 'isloupec'-ty sloupec v podmatici je pod nebo nad diagonalnim prvkem
		{
			// podmatice3
			pom_mat=p_mat;
			is_podm3 = true;
		}
		

		//vypsat_mat<unsigned int>(Sx, Sy, s_mat, NULL);
		// -------------------
		if( tid==0 && !(zpusob & ZPUSOB_S_DELENIM) )
		{
			actions2[isloupec] = pom_mat[cuda_get_index(isloupec, sdiagy, Sx)];
		}
		unsigned int a_pp=1;
		if( !(zpusob & ZPUSOB_S_DELENIM) )
		{
			__syncthreads();
			a_pp=actions2[isloupec];
		}
		__syncthreads();
		int iY=tid;
		while(iY<Sy)
		{
			unsigned int a_py;
			if( is_podm3 || iY!=sdiagy )	// neupravuji pivotni radek pokud je podmatice1
			{
				a_py = s_mat[cuda_get_index(isloupec, iY, Sx)];
				// TODO: ulozit a_pp, a_py
				actions3[isloupec*Sy+iY]=a_py;
				//cout << "SAVE(" << a_pp << ", " << a_py << ")" << endl;
				if(a_py!=0)	// TODO: tuhle podminku dat do podm24
				{
					for(int iX=0;iX<Sx;iX++)
					{
						unsigned int a_xy = s_mat[cuda_get_index(iX, iY, Sx)];
						unsigned int a_xp = pom_mat[cuda_get_index(iX, sdiagy, Sx)];
						//cout << "  " << a_xy << " * " << a_pp << " - " << a_xp << " * " << a_py << endl;
						if(a_xp!=0)
						{
							if(zpusob & ZPUSOB_S_DELENIM)
							{
								s_mat[cuda_get_index(iX, iY, Sx)] = cuda_elem_uprava_s_delenim(modul, a_xy, a_xp, a_py);
							}else
							{
								s_mat[cuda_get_index(iX, iY, Sx)] = cuda_elem_uprava_bez_deleni(modul, a_xy, a_pp, a_xp, a_py);
							}
						}
					}
				}
			}else
			{
				actions3[isloupec*Sy+iY]=0;
			}
			iY+=bdim;
		}
		
	}
	//*/
	
}

__global__ void cuda_GJE_podmatice_kernel(int N, int Sx, int Sy, unsigned int modul, unsigned int* m_matice, unsigned int* m_prava_strana, unsigned int zpusob)
{
	extern __shared__ unsigned int shared_memory[];
	int Smin=min(Sx, Sy);
	unsigned int* s_matice=&(shared_memory[0]);	// velikost Sx*(Sy+Sx)
	unsigned int* actions=&(shared_memory[Sx*(Sy+Sx)]);	// velikost (Sx*Sy+2*Smin)

	
// \FOR{$p$ := $1$ do $\lceil\frac{N}{\min(S_x, S_y)}\rceil$}
	for(int ipivot=0;ipivot<N;ipivot+=Smin)
	{
	// \STATE \COMMENT{zpracovani radku, kde je Z=1}
	// \STATE nacist a spocitat $podmatice_{pp}$ \COMMENT{Z=1}
		int Py=ipivot;

		cuda_copy_podmatice(N, ipivot, Py, Sx, Sy, s_matice, m_matice, m_prava_strana, COPY_MAT_B_GLOB_TO_A_SH);
		//__syncthreads();
		// todo: compute_podmatice1
		cuda_compute_podmatice13(N, modul, ipivot, Sx, Sy, s_matice, actions, zpusob | PODMATICE_12);
		//__syncthreads();
		cuda_copy_podmatice(N, ipivot, Py, Sx, Sy, s_matice, m_matice, m_prava_strana, COPY_MAT_A_SH_TO_B_GLOB);
		//__syncthreads();
	// \FOR{$x$ := $p+1$ do $\lceil\frac{N+1}{S_x}\rceil$}
		for(int x=ipivot+Sx;x<N+1;x+=Sx)
		{
		// \STATE nacist a aplikovat operace v $actions$ na $podmatice_{xp}$ \COMMENT{Z=2}
			cuda_copy_podmatice(N, x, Py, Sx, Sy, s_matice, m_matice, m_prava_strana, COPY_MAT_B_GLOB_TO_A_SH);
			//__syncthreads();
			// todo: compute_podmatice2
			cuda_compute_podmatice24(N, modul, x, Sx, Sy, s_matice, actions, zpusob | PODMATICE_12);
			//__syncthreads();
			cuda_copy_podmatice(N, x, Py, Sx, Sy, s_matice, m_matice, m_prava_strana, COPY_MAT_A_SH_TO_B_GLOB);
			//__syncthreads();
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
				__syncthreads();
			// \STATE nacist a vynulovat $podmatice_{py}$; \COMMENT{Z=3}
				cuda_copy_podmatice(N, ipivot, Py1, Sx, Sy1, s_matice, m_matice, m_prava_strana, COPY_MAT_B_GLOB_TO_A_SH);
				if(Sy2>0) cuda_copy_podmatice(N, ipivot, Py2, Sx, Sy2, &(s_matice[Sx*Sy1]), m_matice, m_prava_strana, COPY_MAT_B_GLOB_TO_A_SH);
				//__syncthreads();
				// todo: nenacitat prvky, ktere uz jsou v s_matice
				cuda_copy_podmatice(N, ipivot, Py, Sx, Sy, &(s_matice[Sx*Sy]), m_matice, m_prava_strana, COPY_MAT_B_GLOB_TO_A_SH);
				//__syncthreads();
				// todo: compute_podmatice3
				cuda_compute_podmatice13(N, modul, ipivot, Sx, Sy, s_matice, actions, zpusob);
				//__syncthreads();

				cuda_copy_podmatice(N, ipivot, Py1, Sx, Sy1, s_matice, m_matice, m_prava_strana, COPY_MAT_A_SH_TO_B_GLOB);
				if(Sy2>0) cuda_copy_podmatice(N, ipivot, Py2, Sx, Sy2, &(s_matice[Sx*Sy1]), m_matice, m_prava_strana, COPY_MAT_A_SH_TO_B_GLOB);
				//__syncthreads();
			// \FOR{$x$ := $p+1$ do $\lceil\frac{N+1}{S_x}\rceil$}
				for(int x=ipivot+Sx;x<N+1;x+=Sx)
				{
				// \STATE nacist a aplikovat operace v $actions$ na $podmatice_{xy}$; \COMMENT{Z=4}
					cuda_copy_podmatice(N, x, Py1, Sx, Sy1, s_matice, m_matice, m_prava_strana, COPY_MAT_B_GLOB_TO_A_SH);
					if(Sy2>0) cuda_copy_podmatice(N, x, Py2, Sx, Sy2, &(s_matice[Sx*Sy1]), m_matice, m_prava_strana, COPY_MAT_B_GLOB_TO_A_SH);
					cuda_copy_podmatice(N, x, Py, Sx, Sy, &(s_matice[Sx*Sy]), m_matice, m_prava_strana, COPY_MAT_B_GLOB_TO_A_SH);
					//__syncthreads();
					// todo: compute_podmatice4
					cuda_compute_podmatice24(N, modul, x, Sx, Sy, s_matice, actions, zpusob);
					//__syncthreads();
					cuda_copy_podmatice(N, x, Py1, Sx, Sy1, s_matice, m_matice, m_prava_strana, COPY_MAT_A_SH_TO_B_GLOB);
					if(Sy2>0) cuda_copy_podmatice(N, x, Py2, Sx, Sy2, &(s_matice[Sx*Sy1]), m_matice, m_prava_strana, COPY_MAT_A_SH_TO_B_GLOB);
					//__syncthreads();
				}
			// \ENDFOR
			}
		// \ENDIF
		}
	//\ENDFOR
	}
}
	
void cuda_GJE_podmatice(int N, unsigned int modul, unsigned int* m_matice, unsigned int* m_prava_strana, unsigned int zpusob)
{
	if(num_of_gpu<=0) return;
	int T=(gpu_property.sharedMemPerBlock / sizeof(unsigned int));
	int Nt=(int)floor((sqrt(1.0+4*(double)T)-1.0)/2.0);
	unsigned int *g_matice, *g_prava_strana;
	cudaProfilerStart();
	cudaMalloc((void**)&g_matice, (N*N)*sizeof(unsigned int));
	cudaMalloc((void**)&g_prava_strana, N*sizeof(unsigned int));
	cudaMemcpy(g_matice, m_matice, (N*N)*sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy(g_prava_strana, m_prava_strana, N*sizeof(unsigned int), cudaMemcpyHostToDevice);
	cuda_start_measuring();
	int num_of_threads;
	switch( ((zpusob & ZPUSOB_VLAKNA) >> 2) )
	{
	case 0:
		num_of_threads=1;
		break;
	case 1:
		num_of_threads=32;
		break;
	case 2:
		num_of_threads=128;
		break;
	case 3:
		num_of_threads = min( 32*((int)ceil((float)(N+1)/32.0)), gpu_property.maxThreadsPerBlock );
		break;
	}
	if(N<=Nt)
	{
		cuda_GJE_while_kernel<<<1,num_of_threads,(N*(N+1))*sizeof(unsigned int)>>>(N, modul, g_matice, g_prava_strana, zpusob);
	}else
	{
		int Sx=4;
		int Sy=4;
		int size_of_shared=Sx*(Sx+2*Sy)+2*min(Sx,Sy);
		// 
		cuda_GJE_podmatice_kernel<<<1,num_of_threads,size_of_shared*sizeof(unsigned int)>>>(N, Sx, Sy, modul, g_matice, g_prava_strana, zpusob);
	}
	cudaThreadSynchronize();
	cuda_stop_measuring();
	
	cudaMemcpy(m_matice, g_matice, (N*N)*sizeof(unsigned int), cudaMemcpyDeviceToHost);
	cudaMemcpy(m_prava_strana, g_prava_strana, N*sizeof(unsigned int), cudaMemcpyDeviceToHost);
	cudaFree(g_matice);
	cudaFree(g_prava_strana);

	cudaProfilerStop();
}

void init_gpu_compute(void)
{
	extern cudaEvent_t cuda_start;
	extern cudaEvent_t cuda_stop;
	num_of_gpu=0;
    cudaGetDeviceCount( &num_of_gpu);
	if (0<num_of_gpu) cudaGetDeviceProperties( &gpu_property, 0);
	cudaEventCreate(&cuda_start);
	cudaEventCreate(&cuda_stop);
	//cudaProfilerInitialize
}
void print_gpus_info(void)
{
	cudaDeviceProp prop;
    int count=0;
 
    cudaGetDeviceCount( &count);
	printf("Pocet CUDA zarizeni: %d\n", count);
	for (int i=0; i< count; i++)
	{
		cudaGetDeviceProperties( &prop, i);

		printf( " --- General Information for device %d ---\n", i );
		printf( "Name: %s\n", prop.name );
		printf( "Compute capability: %d.%d\n", prop.major, prop.minor );
		printf( "Clock rate: %d\n", prop.clockRate );
		printf( "Device copy overlap: " );
       
		if (prop.deviceOverlap)
			printf( "Enabled\n" );
		else
			printf( "Disabled\n" );
		printf( "Kernel execition timeout : " );


		if (prop.kernelExecTimeoutEnabled)
			printf( "Enabled\n" );
		else
			printf( "Disabled\n" );
    
		printf( " --- Memory Information for device %d ---\n", i );
		printf( "Total global mem: %ld\n", prop.totalGlobalMem );
		printf( "Total constant Mem: %ld\n", prop.totalConstMem );
		printf( "Max mem pitch: %ld\n", prop.memPitch );
		printf( "Texture Alignment: %ld\n", prop.textureAlignment );
		printf( " --- MP Information for device %d ---\n", i );
		printf( "Multiprocessor count: %d\n",
		prop.multiProcessorCount );
		printf( "Shared mem per mp: %ld\n", prop.sharedMemPerBlock );
		printf( "Registers per mp: %d\n", prop.regsPerBlock );
		printf( "Threads in warp: %d\n", prop.warpSize );
		printf( "Max threads per block: %d\n",
		prop.maxThreadsPerBlock );
		printf( "Max thread dimensions: (%d, %d, %d)\n",
		prop.maxThreadsDim[0], prop.maxThreadsDim[1],
		prop.maxThreadsDim[2] );
		printf( "Max grid dimensions: (%d, %d, %d)\n",
		prop.maxGridSize[0], prop.maxGridSize[1],
		prop.maxGridSize[2] );
		printf( "\n" );
	}
}

void print_cuda_err(cudaError_t cudaErr)
{
	switch(cudaErr)
	{
	case cudaSuccess: printf("cudaSuccess");
		break;
	case cudaErrorInvalidValue: printf("cudaErrorInvalidValue");
		break;
	case cudaErrorInvalidDevicePointer: printf("cudaErrorInvalidDevicePointer");
		break;
	case cudaErrorInvalidMemcpyDirection: printf("cudaErrorInvalidMemcpyDirection");
		break;
	}
}

__global__ void cuda_GJE_global1(int N, unsigned int modul, unsigned int* g, unsigned int zpusob)
{
	gauss_jordan_elim_while_kernel(N+1, N, modul, g, zpusob);
}
void cuda_GJE_global(int N, unsigned int modul, unsigned int* m_matice, unsigned int zpusob)
{
	if(num_of_gpu<=0) return;
	unsigned int *g;
	
	cudaProfilerStart();
	cudaMalloc((void**)&g, (N*(N+1))*sizeof(unsigned int));
	cudaMemcpy(g, m_matice, (N*(N+1))*sizeof(unsigned int), cudaMemcpyHostToDevice);

	int num_of_threads;
	switch( ((zpusob & ZPUSOB_VLAKNA) >> 2) )
	{
	case 0:
		num_of_threads=1;
		break;
	case 1:
		num_of_threads=32;
		break;
	case 2:
		num_of_threads=128;
		break;
	case 3:
		num_of_threads = min( 256*((int)ceil((float)(N+1)/32.0)), gpu_property.maxThreadsPerBlock );
		break;
	}
	cuda_start_measuring();
	cuda_GJE_global1<<<1,num_of_threads>>>(N, modul, g, zpusob);
	
	cudaThreadSynchronize();
	cuda_stop_measuring();
	
	cudaMemcpy(m_matice, g, (N*(N+1))*sizeof(unsigned int), cudaMemcpyDeviceToHost);
	cudaFree(g);

	cudaProfilerStop();
}

__global__ void test_elem_uprava_kernel0(int n, unsigned int modul)
{
	int i=threadIdx.x;
	unsigned int a1=1298161;
	unsigned int a2;
	while(i<n)
	{
		a2=a1;
		a1=cuda_elem_uprava_bez_deleni(modul, a1, 1293001, a2, 1269239);
		i+=blockDim.x;
	}
}
__global__ void test_elem_uprava_kernel1(int n, unsigned int modul)
{
	int i=threadIdx.x;
	unsigned int a1=1298161;
	unsigned int a2;
	while(i<n)
	{
		a2=a1;
		cuda_elem_uprava_s_delenim(modul, a1, a2, 1269239);
		i+=blockDim.x;
	}
}
__global__ void test_elem_uprava_kernel2(int n, unsigned int modul)
{
	int i=threadIdx.x;
	unsigned int a1=1298161;
	unsigned int a2;
	while(i<n)
	{
		a2=a1;
		cuda_elem_uprava_bez_deleni1(modul, a1, 1293001, a2, 1269239);
		i+=blockDim.x;
	}
}
__global__ void test_elem_uprava_kernel3(int n, unsigned int modul)
{
	int i=threadIdx.x;
	unsigned int a1=1298161;
	unsigned int a2;
	while(i<n)
	{
		a2=a1;
		cuda_elem_uprava_s_delenim1(modul, a1, a2, 1269239);
		i+=blockDim.x;
	}
}
void test_elem_uprava(int N, unsigned int modul, unsigned int zpusob)
{
	if(num_of_gpu<=0) return;
	cudaProfilerStart();
	int num_of_threads;
	switch( ((zpusob & ZPUSOB_VLAKNA) >> 2) )
	{
	case 0:
		num_of_threads=1;
		break;
	case 1:
		num_of_threads=32;
		break;
	case 2:
		num_of_threads=128;
		break;
	case 3:
		num_of_threads = min( 256*((int)ceil((float)(N+1)/32.0)), gpu_property.maxThreadsPerBlock );
		break;
	}

	cuda_start_measuring();
	// vypocet
	switch( (zpusob & 0x0003) )
	{
	case 0: test_elem_uprava_kernel0<<<1,num_of_threads>>>(N, modul);
		break;
	case 1: test_elem_uprava_kernel1<<<1,num_of_threads>>>(N, modul);
		break;
	case 2: test_elem_uprava_kernel2<<<1,num_of_threads>>>(N, modul);
		break;
	case 3: test_elem_uprava_kernel3<<<1,num_of_threads>>>(N, modul);
		break;

	}

	cudaThreadSynchronize();
	cuda_stop_measuring();
	
	cudaProfilerStop();
}
