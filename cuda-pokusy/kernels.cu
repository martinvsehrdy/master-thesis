#include <stdio.h>
#include <cstdio>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include "kernels.h"
#include "time_measure.h"
#include "common.h"

void vypsat_mat(int nx, int ny, unsigned int* matice, unsigned int* prava_strana)
{
	printf("\n");
	for(int y=0;y<min(ny,12);y++)
	{
		int x;
		for(x=0;x<min(nx,8);x++)
		{
			unsigned int a=matice[x+y*nx];
			printf("%6u\t", a);
		}
		if(x<nx-1)
		{
			printf("...");
		}
		printf("| ");
		if(prava_strana!=NULL)
		{
			printf("%u", prava_strana[y]);
		}
		printf("\n");
	}
}
int save_matrix(int N, unsigned int* matice, unsigned int* prava_strana, FILE* f)
{
	/*fstream file;
	file.open(filename, fstream::out);
	if(!file.is_open()) return 1;*/
	
	 if (f==NULL) return 1;
	//file << N << endl;
	fprintf(f, "%d\n", N);
	
	for(int y=0;y<N;y++)
	{
		int x;
		for(x=0;x<N;x++)
		{
			//file << matice[get_index(x, y, N)] << "\t";
			fprintf(f, "%8u\t", matice[x+y*N]);
		}
		if(prava_strana!=NULL)
		{
			//file << "| " << prava_strana[y];
			fprintf(f, "| %u", prava_strana[y]);
		}
		//file << endl;
		fprintf(f, "\n");
	}
	return 0;
}


__device__ int cuda_get_index(int X, int Y, int N)	// SLOUPEC, RADEK
{
	return Y*N+X;
}
__device__ unsigned int cuda_compute_inverse_eukleides(unsigned int cislo, unsigned int modul)
{
	unsigned int a, b, a1, b1, q, r;
	a = cislo;
	b = modul;
	a1 = 0;
	b1 = 1;
	int plus = 1;

	while( b!=0 )
	{
		q = a / b;
		r = a % b;
		a = b;
		b = r;
		r = a1;
		a1 = b1 + r*q;
		b1 = r;
		plus=-plus;
	}
	if( a==1 )
	{
		if( 0<plus )
		{
			return (unsigned int)b1;
		}else
		{
			return (unsigned int)(modul-b1);
		}
	}
	return (unsigned int)0;
}
__device__ unsigned int cuda_multiply_add_modulo(unsigned int modul, unsigned int a, unsigned int b, unsigned int c)
// \STATE := (a * b + c) % modul
{
	// integer aritmetika
	unsigned long long pom = a;
	pom *= b;
	pom += c;
	pom %= modul;
	return (unsigned int)pom;

	// float aritmetika
	double p1;
	p1 = ((double)a) * ((double)b) + ((double)c);
	// __fma_rd(a, b, c) k vypoctu (a * b + c)
	//p1 = __fma_rd( (double)a, (double)b, (double)c );

	double q = floor( p1/modul );
	double p2 = q * modul;
	return ((unsigned int)(p1 - p2));
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
__device__ unsigned int cuda_elem_uprava_s_delenim(unsigned int modul, unsigned int a_xy, unsigned int a_xp, unsigned int a_py)
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
__device__ unsigned int cuda_elem_uprava_bez_deleni1(unsigned int modul, unsigned int a_xy, unsigned int a_pp, unsigned int a_xp, unsigned int a_py)
// \STATE $a_{xy} := a_{xy} \cdot a_pp - a_xp \cdot a_py$
{
	double p1 = (((double)a_xy) * ((double)a_pp)) - (((double)a_xp) * ((double)a_py));
	double q = floor(p1/(double)modul);
	double p2 = q*modul;
	
	return ((unsigned int)(p1-p2));
}
// elementarni uprava s delenim
__device__ unsigned int cuda_elem_uprava_s_delenim1(unsigned int modul, unsigned int a_xy, unsigned int a_xp, unsigned int a_py)
// \STATE $a_{xy} := a_xy - a_xp \cdot a_py$
{
	double p1 = ((double)a_xy) - (((double)a_xp) * ((double)a_py));
	double q = floor(p1/(double)modul);
	double p2 = q*modul;
	
	return ((unsigned int)(p1-p2));
}
// elementarni uprava bez deleni
__device__ unsigned int cuda_elem_uprava_bez_deleni2(unsigned int modul, unsigned int a_xy, unsigned int a_pp, unsigned int a_xp, unsigned int a_py)
// \STATE $a_{xy} := a_{xy} \cdot a_pp - a_xp \cdot a_py$
{
	double p1 = __fma_rd((double)a_xp, -((double)a_py), __dmul_rn( (double)a_xy, (double)a_pp ));
	double q = __double2uint_rd( p1/(double)modul );
	double p2 = __dmul_rn(q, (double)modul);
	
	return ((unsigned int)(p1-p2));
}
// elementarni uprava s delenim
__device__ unsigned int cuda_elem_uprava_s_delenim2(unsigned int modul, unsigned int a_xy, unsigned int a_xp, unsigned int a_py)
// \STATE $a_{xy} := a_xy - a_xp \cdot a_py$
{
	double p1 = __fma_rd((double)a_xp, -((double)a_py), (double)a_xy);
	double q = __double2uint_rd( p1/(double)modul );
	double p2 = __dmul_rn(q, (double)modul);
	
	return ((unsigned int)(p1-p2));
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
	unsigned int* actions3=&(actions2[minS]);	// multiplikatory, 'Smin*Sy' cisel
	//cout << "modul = " << modul << endl;
	int tid=threadIdx.x;
	int bdim=blockDim.x;
	// p_mat - pomocna podmatice, max velikost Sx*Sy
	
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
	unsigned int* actions3=&(actions2[minS]);	// multiplikatory, 'Smin*Sy' cisel

	int tid=threadIdx.x;
	int bdim=blockDim.x;
	// p_mat - pomocná podmatice, max velikost Sx*Sy
	
// \FOR{$p$ := $1$ do $Sx$}
	for(int isloupec=0;(isloupec<minS);isloupec++)
	{
		// najit g_diagonalu ve sloupci 'isloupec'
		
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
				actions2[isloupec]=cuda_compute_inverse_eukleides(s_mat[cuda_get_index(isloupec, isloupec, Sx)], modul);
			}
			__syncthreads();
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
			actions2[isloupec] = pom_mat[cuda_get_index(isloupec, isloupec, Sx)];
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
			if( is_podm3 || iY!=isloupec )	// neupravuji pivotni radek pokud je podmatice1
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
						unsigned int a_xp = pom_mat[cuda_get_index(iX, isloupec, Sx)];
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
	//int bid = blockIdx.x;
	//int gdim = gridDim.x;
	int Smin=min(Sx, Sy);
	unsigned int* s_matice=&(shared_memory[0]);	// velikost Sx*Sy+Sx*Smin = Sx*(Sy+Smin)
	unsigned int* actions=&(shared_memory[Sx*(Sy+Smin)]);	// velikost Smin*Sy+2*Smin
	unsigned int mask_copy = 0x0000;
	if( !(zpusob & ZPUSOB_S_DELENIM) ) mask_copy |= COPY_MAT_BEZ_DELENI;
	unsigned int* p_matice=&(s_matice[Sx*Sy]);
	
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
				cuda_copy_podmatice(N, ipivot, Py, Sx, Smin, p_matice, m_matice, m_prava_strana, COPY_MAT_B_GLOB_TO_A_SH );
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
					cuda_copy_podmatice(N, x, Py, Sx, Smin, p_matice, m_matice, m_prava_strana, COPY_MAT_B_GLOB_TO_A_SH );
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
		num_of_threads = min( gpu_property.warpSize*((int)ceil((float)(N+1)/gpu_property.warpSize)), gpu_property.maxThreadsPerBlock );
		break;
	}
	
	int Nt=(int)floor((sqrt(1.0+4*(double)T)-1.0)/2.0);
	if(N<=Nt)
	{
		cuda_GJE_while_kernel<<<1,num_of_threads,(N*(N+1))*sizeof(unsigned int)>>>(N, modul, g_matice, g_prava_strana, zpusob);
	}else
	{
		int Sx;
		int Sy;
		float fSx=sqrt( (float)((N+1)*(N+1)+T) ) - (N+1);
		if( fSx < 1)
		{
			Sx=1;
			Sy=(T-3)/2;
		}else
		{
		Sx=(int)floor(fSx);
			Sy=N;
		}
		int Smin=min(Sx,Sy);
		int size_of_shared= Sx*Sy+Sx*Smin +	// size_s_matice
							Smin*Sy+2*Smin;	// size_actions
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
__global__ void find_inverse(int N, unsigned int modul, int posl_ipivot, unsigned int* m_matice, int* pivot_radek, unsigned int* inverse)
{
	int q=posl_ipivot;
	int novy_ipivot=posl_ipivot;
	novy_ipivot++;
	unsigned int a;
	do
	{
		q++;
		a = m_matice[cuda_get_index(novy_ipivot, q, N)];
	}while( a==0 );
	unsigned int a_inv = cuda_compute_inverse_eukleides(a, modul);

	pivot_radek[0] = q;
	inverse[0] = a_inv;
}
__global__ void cuda_GJE_radky_kernel(int N, unsigned int modul, int ipivot, unsigned int* m_matice, unsigned int* m_prava_strana, 
						int* pivot_radek, unsigned int* inverse, unsigned int zpusob)
{
	int tid=threadIdx.x;
	int bdim=blockDim.x;
	int bid=blockIdx.x;
	int gdim=gridDim.x;
	int bN;
#if defined(SHARED_SIZE) && SHARED_SIZE>0
	bN=SHARED_SIZE;
#else
	bN=(int)ceil((double)(N+1)/gdim);
#endif
	//if(bid==0) return;
	// TODO: velikost sh_mem udelat konstantni - zmerit rychlost vypoctu v zavislosti na teto velikosti
#if defined(SHARED_SIZE) && SHARED_SIZE>0
	__shared__ unsigned int sh_mem[SHARED_SIZE];	// size_sh_mem = N
#endif
	__shared__ int sh_q;	// CUDA: 'q' sdilene, pak si musi kazde vlakno vzit svou kopii
	__shared__ unsigned int sh_a_pq_inv;
	
// \STATE \COMMENT{priprava pivotniho radku}
// \STATE nacist prvek $[p;q]$ do sdilene pameti
	if( tid==0 )
	{
		sh_q = pivot_radek[0];
		sh_a_pq_inv = inverse[0];
	}
	__syncthreads();
	int q = pivot_radek[0];
	unsigned int a_pq_inv=inverse[0];

// \FOR{$x$ := $p+1$ do $N$}
	int iX=tid;
	while(iX<bN)
	{
		int gX=iX+bid*bN;
		if( gX>=ipivot && gX<=N )
		{
		// \STATE nacist, vydelit a ulozit do sdilene pameti
			unsigned long long a;
			if(gX==N) a = m_prava_strana[q];
			else a = m_matice[cuda_get_index(gX, q, N)];
			a = cuda_multiply_add_modulo(modul, a, a_pq_inv, 0);
#if defined(SHARED_SIZE) && SHARED_SIZE>0
			sh_mem[iX] = (unsigned int)a;
#else
			if(gX==N) m_prava_strana[q] = (unsigned int)a;
			else m_matice[cuda_get_index(gX, q, N)] = (unsigned int)a;
#endif
		}
		iX+=bdim;
	}
	__syncthreads();
// \ENDFOR
// \FOR{$y$ := $1$ do $N$}
		// TODO: prehodit cykly pres Y a X mezi sebou, vlakno nebude prochazet radek, ale sloupec
		int iY=tid;	// prochazi pres Y, kazde vlakno samostatny radek
		while(iY<N)
		{
			unsigned int a_py = m_matice[cuda_get_index(ipivot, iY, N)];
		// \FOR{$x$ := $p+1$ do $N$}
			for(int iX=0;iX<bN;iX++)
			{
				int gX=iX+bid*bN;
				if( gX>ipivot && gX<=N )
				{
				// \IF{$y$ == $q$}
					if(iY == q)	// ma na starosti pivotni radek => pouze uklada do globalni
					{
					// \STATE ulozit do globalni pameti prvek $[x;y]=[x;q]$
#if defined(SHARED_SIZE) && SHARED_SIZE>0
						if(gX==N) m_prava_strana[iY] = sh_mem[iX];
						else m_matice[cuda_get_index(gX, iY, N)] = sh_mem[iX];
#endif
				// \ELSE
					}else
					{
					// \STATE upravit prvek $[x;y]$ stejne jako pri nulovani prvku $[p;y]$
						unsigned int a_xy;
						if(gX==N) a_xy = m_prava_strana[iY];
						else a_xy = m_matice[cuda_get_index(gX, iY, N)];
						unsigned int a_xp;
#if defined(SHARED_SIZE) && SHARED_SIZE>0
						a_xp = sh_mem[iX];
#else
						if(gX==N) a_xp=m_prava_strana[q];
						else a_xp=m_matice[cuda_get_index(gX, q, N)];
#endif
						//cout << "  " << a_xy << " * " << a_pp << " - " << a_xp << " * " << a_py << endl;
						
						a_xy = cuda_elem_uprava_s_delenim(modul, a_xy, a_xp, a_py);
						
						if(gX==N) m_prava_strana[iY] = a_xy;
						else m_matice[cuda_get_index(gX, iY, N)] = a_xy;
					}
				}
			// \ENDIF
		// \ENDFOR
			}
			iY+=bdim;
		}
// \ENDFOR

}
	
void cuda_GJE_radky(int N, unsigned int modul, unsigned int* m_matice, unsigned int* m_prava_strana, unsigned int zpusob)
{
	if(num_of_gpu<=0) return;
	int T=(gpu_property.sharedMemPerBlock / sizeof(unsigned int));
	unsigned int *g_matice, *g_prava_strana;
	cudaProfilerStart();
	cudaMalloc((void**)&g_matice, (N*N)*sizeof(unsigned int));
	cudaMalloc((void**)&g_prava_strana, N*sizeof(unsigned int));
	cudaMemcpy(g_matice, m_matice, (N*N)*sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy(g_prava_strana, m_prava_strana, N*sizeof(unsigned int), cudaMemcpyHostToDevice);
	unsigned int* g_inverse;
	unsigned int* m_inverse;
	cudaMalloc((void**)&g_inverse, sizeof(unsigned int));
	int* g_pivot;
	int* m_pivot;
	cudaMalloc((void**)&g_pivot, sizeof(int));
	cuda_start_measuring();
	int num_of_blocks;
#if defined(SHARED_SIZE) && SHARED_SIZE>0
	num_of_blocks = (int)ceil((double)(N+1)/SHARED_SIZE);
#else
	num_of_blocks = gpu_property.multiProcessorCount;
#endif
	// N+1 vlaken = N radku + 1 vl na pocitani inverze
	int num_of_threads = min( gpu_property.warpSize*((int)ceil((float)(N+1)/((float)gpu_property.warpSize))), gpu_property.maxThreadsPerBlock );

	//cuda_GJE_radky_kernel<<<1,num_of_threads,size_of_shared*sizeof(unsigned int)>>>(N, modul, g_matice, g_prava_strana, zpusob);

	//find_inverse(N, modul, -1, g_matice, g_pivot, g_inverse);
// \FOR{$p$ := $1$ do $N$}
#ifdef _DEBUG
	FILE* file=fopen("log", "w");
#endif
	for(int ipivot=0;ipivot<N;ipivot++)
	{
		//printf("pivot = %d\n", ipivot);
	// \STATE \COMMENT{nalezeni radku s nenulovou hodnotou prvku $[p;q]$, kde $p<=q$}
		find_inverse<<<1,1>>>(N, modul, ipivot-1, g_matice, g_pivot, g_inverse);
		cudaThreadSynchronize();
	// \STATE \COMMENT{priprava pivotniho radku, Uprava ostatnich radku}
		cuda_GJE_radky_kernel<<<num_of_blocks,num_of_threads>>>(N, modul, ipivot, g_matice, g_prava_strana, g_pivot, g_inverse, zpusob);
		cudaThreadSynchronize();
#ifdef _DEBUG
		cudaMemcpy(m_matice, g_matice, (N*N)*sizeof(unsigned int), cudaMemcpyDeviceToHost);
		cudaMemcpy(m_prava_strana, g_prava_strana, N*sizeof(unsigned int), cudaMemcpyDeviceToHost);
		vypsat_mat(N, N, m_matice, m_prava_strana);
		fprintf(file, "pivot=%d\n", ipivot);
		save_matrix(N, m_matice, m_prava_strana, file);
	}
	fclose(file);
#else
	}
#endif
// \ENDFOR
	
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
		num_of_threads = min( 32*((int)ceil((float)(N+1)/32.0)), gpu_property.maxThreadsPerBlock );
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

__global__ void test_elem_uprava_kernel_bez(int n, unsigned int modul)
{
	int bdim=blockDim.x;
	unsigned int a1=1298161;
	unsigned int a2;
	unsigned int a3=a1;
	for(int i=0;i<n;i+=bdim)
	{
		a3=a1;
		a1=cuda_elem_uprava_bez_deleni(modul, a1, 1293001, a2, 1269239);
		a2=a3;
	}
}
__global__ void test_elem_uprava_kernel_s(int n, unsigned int modul)
{
	int bdim=blockDim.x;
	unsigned int a1=1298161;
	unsigned int a2;
	unsigned int a3=a1;
	for(int i=0;i<n;i+=bdim)
	{
		a3=a1;
		a1=cuda_elem_uprava_s_delenim(modul, a1, a2, 1269239);
		a2=a3;
	}
}
__global__ void test_elem_uprava_kernel_bez1(int n, unsigned int modul)
{
	int bdim=blockDim.x;
	unsigned int a1=1298161;
	unsigned int a2;
	unsigned int a3=a1;
	for(int i=0;i<n;i+=bdim)
	{
		a3=a1;
		a1=cuda_elem_uprava_bez_deleni1(modul, a1, 1293001, a2, 1269239);
		a2=a3;
	}
}
__global__ void test_elem_uprava_kernel_s1(int n, unsigned int modul)
{
	int bdim=blockDim.x;
	unsigned int a1=1298161;
	unsigned int a2;
	unsigned int a3=a1;
	for(int i=0;i<n;i+=bdim)
	{
		a3=a1;
		a1=cuda_elem_uprava_s_delenim1(modul, a1, a2, 1269239);
		a2=a3;
	}
}
__global__ void test_elem_uprava_kernel_bez2(int n, unsigned int modul)
{
	int bdim=blockDim.x;
	unsigned int a1=1298161;
	unsigned int a2;
	unsigned int a3=a1;
	for(int i=0;i<n;i+=bdim)
	{
		a3=a1;
		a1=cuda_elem_uprava_bez_deleni2(modul, a1, 1293001, a2, 1269239);
		a2=a3;
	}
}
__global__ void test_elem_uprava_kernel_s2(int n, unsigned int modul)
{
	int bdim=blockDim.x;
	unsigned int a1=1298161;
	unsigned int a2;
	unsigned int a3=a1;
	for(int i=0;i<n;i+=bdim)
	{
		a3=a1;
		a1=cuda_elem_uprava_s_delenim2(modul, a1, a2, 1269239);
		a2=a3;
	}
}


void test_elem_uprava(int N, unsigned int modul, unsigned int zpusob)
{
	if(num_of_gpu<=0) return;
	cudaProfilerStart();

	cuda_start_measuring();
	// vypocet
	if( (zpusob & ZPUSOB_S_DELENIM) )
	{
		test_elem_uprava_kernel_s<<<1,1>>>(N, modul);
	}else
	{
		test_elem_uprava_kernel_bez<<<1,1>>>(N, modul);
	}
	cudaThreadSynchronize();
	cuda_stop_measuring();
	cudaProfilerStop();
}

void test_elem_uprava1(int N, unsigned int modul, unsigned int zpusob)
{
	if(num_of_gpu<=0) return;
	cudaProfilerStart();

	cuda_start_measuring();
	// vypocet
	if( (zpusob & ZPUSOB_S_DELENIM) )
	{
		test_elem_uprava_kernel_s1<<<1,1>>>(N, modul);
	}else
	{
		test_elem_uprava_kernel_bez1<<<1,1>>>(N, modul);
	}
	cudaThreadSynchronize();
	cuda_stop_measuring();
	cudaProfilerStop();
}

void test_elem_uprava2(int N, unsigned int modul, unsigned int zpusob)
{
	if(num_of_gpu<=0) return;
	cudaProfilerStart();

	cuda_start_measuring();
	// vypocet
	if( (zpusob & ZPUSOB_S_DELENIM) )
	{
		test_elem_uprava_kernel_s2<<<1,1>>>(N, modul);
	}else
	{
		test_elem_uprava_kernel_bez2<<<1,1>>>(N, modul);
	}
	cudaThreadSynchronize();
	cuda_stop_measuring();
	cudaProfilerStop();
}
