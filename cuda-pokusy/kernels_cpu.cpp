#include "stdafx.h"
#include "kernels_cpu.h"
#include "templates_functions.h"
#include <cstdio>

using namespace std;


// TODO: cuda __mul64hi


/* nacte/ulozi podmatici z globalni p. do sdilene nebo zpet
 * Sx, Sy - velikost podmatice, mela by se vejit do sdilene pameti
 * sx, sy - souradnice zvolene podmatice v matici, sx \in [0; ceil(N/Sx)]
 * mat_A, mat_B - zdrojova nebo cilova adresa
 */
//#define COPY_MAT_B_GLOB_TO_A_SH	1
//#define COPY_MAT_A_SH_TO_B_GLOB	2
//#define COPY_MAT_A_SH_TO_B_SH 	3
void copy_podmatice(int N, int sx, int sy, int Sx, int Sy, unsigned int* mat_A, unsigned int* mat_B, unsigned int* prava_str, int copy_to)
{
	int tid=0;
	int bdim=1;
	int itid=tid;
	while(itid<Sy)
	{
		for(int ix=0;ix<Sx;ix++)
		{
			int glob_x=sx*Sx+ix;
			int glob_y=sy*Sy+itid;
			if(glob_x<=N && glob_y<N)
			{
				if(glob_x<N)
				{
					switch(copy_to)
					{
					case COPY_MAT_A_SH_TO_B_GLOB:
						mat_B[get_index(glob_x, glob_y, N)] = mat_A[get_index(ix, itid, Sx)];
						break;
					case COPY_MAT_B_GLOB_TO_A_SH:
						mat_A[get_index(ix, itid, Sx)] = mat_B[get_index(glob_x, glob_y, N)];
						break;
					}
				}else
				{
					switch(copy_to)
					{
					case COPY_MAT_A_SH_TO_B_GLOB:
						prava_str[glob_y] = mat_A[get_index(ix, itid, Sx)];
						break;
					case COPY_MAT_B_GLOB_TO_A_SH:
						mat_A[get_index(ix, itid, Sx)] = prava_str[glob_y];
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
	if(copy_to == COPY_MAT_A_SH_TO_B_GLOB)
		vypsat_vys<unsigned int>(N, prava_str, NULL);
}
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
	unsigned int minS=min( min(Sx, Sy), min(N-Sx*sx, N-Sy*sy) );
	// \FOR{$p$ := $1$ do $N$}
	for(unsigned int ipivot=0;ipivot<minS;ipivot++)
	{
		// todo: jak se to chova pri S=4, ipivot=3
		cout << endl << "ipivot = " << ipivot << endl;
		vypsat_mat<unsigned int>(Sx, Sy, s_mat, NULL);
		unsigned int novy_pivot;
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
		m1=compute_inverse(s_mat[get_index(ipivot, ipivot, Sx)], modul);
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
void compute_podmatice2(int N, unsigned int modul, int sx, int sy, int Sx, int Sy, unsigned int* s_mat, unsigned int* actions)
{
	// todo: debugovat chovani pro Sx=4, pro pravou stranu, overit spravnost aplikovani cisel z actions
	cout << "modul = " << modul << endl;
	int tid=0;
	int bdim=1;
	unsigned int m1;
	unsigned long long pom;
	unsigned int minS=min( min(Sx, Sy), min(N-Sx*sx, N-Sy*sy) );
	// \FOR{$p$ := $1$ do $N$}
	for(unsigned int ipivot=0;ipivot<minS;ipivot++)
	{
		cout << endl << "ipivot = " << ipivot << endl;
		vypsat_mat<unsigned int>(Sx, Sy, s_mat, NULL);
		unsigned int novy_pivot;
		
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
		m1=actions[minS+ipivot];
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
				int index_actions=2*minS+(Sy-1)*ipivot+iy;
				if(ipivot<iy) index_actions--;
				unsigned int m_py=actions[index_actions];
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
void compute_podmatice13(int N, unsigned int modul, int sx, int sy, int Sx, int Sy, unsigned int* s_mat, unsigned int* actions)
{
	cout << "modul = " << modul << endl;
	for(int i=0;i<(Sx*Sy+Sx);i++)
	{
		actions[i]=modul;
	}
	// TODO: vyuzivat i podmatici s_mat1
	// podmatice s_mat: |-- podmatice, kterou pocitam (Sx*Sy cisel) --| |-- podmatice, kterou potrebuji (Sx^2 cisel) --|
	//unsigned int* s_mat1=
	int tid=0;
	int bdim=1;
	unsigned int m1=modul;
	unsigned long long pom;
	unsigned int minS=min( min(Sx, Sy), min(N-Sx*sx, N-Sy*sy) );
	int pod_diag_x=Sy*sy+Sy-Sx*sx;
// \FOR{$p$ := $1$ do $Sx$}
	for(unsigned int isloupec=0;(isloupec<Sx)&&(isloupec<pod_diag_x);isloupec++)
	{
		cout << "====================\npivot=" << isloupec << endl;
		int gdiagx=Sx*sx+isloupec;
		int sdiagy;
	// \IF{ve sloupci $p$ v podmatici je diagonalni prvek}
		if(gdiagx<Sy*sy+Sy)	// Z=3 nebo Z=1, generuju actions
		{
			if( gdiagx<Sy*sy )
			{
				// Z=3
				actions[isloupec] = isloupec;
				m1=s_mat[get_index(isloupec, sdiagy, Sx)];
				cout << endl << "odecist " << m1 << " nasobek pivotniho radku" << endl;
				int index_actions=minS+Sy*isloupec;
				actions[index_actions]=m1;
			}else
			{
				// Z=1
				unsigned int novy_pivot;
				if(tid==0)
				{
					novy_pivot=isloupec;
					while(s_mat[get_index(novy_pivot, novy_pivot, Sx)]==0 && novy_pivot<minS)
					{
						novy_pivot++;
					}
					if(novy_pivot==minS) novy_pivot=isloupec;
					actions[isloupec] = novy_pivot;
					cout << " radek pivota=" << novy_pivot;
				}
				// cuda: synchronize
				novy_pivot=actions[isloupec];
				if(novy_pivot>=minS) return;
				if(novy_pivot != isloupec)
				{
					unsigned int pom1;
					int ix=isloupec+tid;
					while( (ix<Sx) && (Sx*sx+ix<=N) )
					{
						pom1=s_mat[get_index(ix, novy_pivot, Sx)];
						s_mat[get_index(ix, novy_pivot, Sx)] = s_mat[get_index(ix, isloupec, Sx)];
						s_mat[get_index(ix, isloupec, Sx)] = pom1;
						ix+=bdim;
					}
				}
			// \STATE $(isloupec, sdiagy)$ := souradnice diagonalniho prvku
				sdiagy=gdiagx-Sy*sy;
			// \STATE \COMMENT {vydelit cely $sdiagy$-ty radek cislem $a_{isloupec sdiagy}$, v $isloupec$-tem sloupci na diagonale bude cislo 1}
				m1=compute_inverse(s_mat[get_index(isloupec, sdiagy, Sx)], modul);
				cout << endl << "vydelit " << s_mat[get_index(isloupec, sdiagy, Sx)] << " ~ vynasobit " << m1 << endl;
				int index_actions=minS+Sy*isloupec;
				actions[index_actions]=m1;
			// \FOR{$x$ := $p$ do $Sx$}
				int ix=isloupec+tid;
				while( (ix<Sx) && (Sx*sx+ix<=N) )
				{
				  // \STATE $a_{xp} := \frac{a_{xp}}{a_pp}$
					pom = s_mat[get_index(ix, isloupec, Sx)];
					pom *= m1;
					pom %= modul;
					s_mat[get_index(ix, isloupec, Sx)]=(unsigned int)pom;
			
					ix+=bdim;
				// \ENDFOR
				}
			}
		}else
		{
			// Z=2
		}



		if( (Sy*sy<=gdiagx) && (gdiagx<Sy*(sy+1)) )
		{
			
			vypsat_mat<unsigned int>(Sx, Sy, s_mat, NULL);
		// \STATE ulozit $a_{px py}$ do $actions$
	// \ELSE
		}else
		{
		// \STATE \COMMENT {diagonalni prvek je nekde vejs (pouze nuluju) nebo vlevo (negeneruju actions)}
			// 'sdiagy' je y-ova souradnice prvku v diagovalni radku
			sdiagy = Sy +		// offset do podmatice D
					isloupec;	// offset na radek, kde je diagonalni prvek
	// \ENDIF
		}
	// \STATE \COMMENT {Nulovani vsech prvku ve sloupci $p$}
    // \FOR{$y$ := $1$ do $Sy$}
		int iy=tid;
		while( (iy<Sy) && (Sy*sy+iy<N) )
		{
			int gdiagy=Sy*sy+iy;
			if(gdiagx<Sy*sy+Sy)	// Z=3 nebo Z=1, generuju actions
			{
				if( gdiagx<Sy*sy )
				{
					// Z=3
					cout << "Z=3" << endl;
					// \STATE uložit $a_py$ do $actions$
					unsigned int m_py=s_mat[get_index(isloupec, iy, Sx)];
					int index_actions=minS+Sy*isloupec+iy;
					actions[index_actions]=m_py;
					cout << " | minus " << m_py << " krat pivot";
				// \FOR{$x$ := $p$ do $N$}
					for(int ix=isloupec;(ix<Sx)&&(Sx*sx+ix<=N);ix++)
					{
				  // \STATE $a_{xy} := a_{xy} - a_xp \cdot a_py$
						s_mat[get_index(ix, iy, Sx)]=elem_uprava_s_delenim(modul, 
							s_mat[get_index(ix, iy, Sx)],		// a_xy
							s_mat[get_index(ix, sdiagy, Sx)], // a_xp, muze byt z podmatice D
							m_py);
				// \ENDFOR
					}
				}else
				{
					// Z=1
					cout << "Z=1" << endl;
				}
			}else
			{
				// Z=2
				cout << "Z=2" << endl;
			}


		  // \IF {$p$ != $y$}					// neni tento radek oznacen jako pivotni?
			if( false && 0 != (Sy*sy+iy-isloupec) % Sx )	// todo: pro Z=24 modulovat, aby podminka platila pro ty same sloupce jako Z=13
			{
			// \STATE uložit $a_py$ do $actions$
				unsigned int m_py=s_mat[get_index(isloupec, iy, Sx)];
				int index_actions=minS+Sy*isloupec+iy;
				actions[index_actions]=m_py;
				cout << " | minus " << m_py << " krat pivot";
			// \FOR{$x$ := $p$ do $N$}
				for(int ix=isloupec;(ix<Sx)&&(Sx*sx+ix<=N);ix++)
				{
			  // \STATE $a_{xy} := a_{xy} - a_xp \cdot a_py$
					s_mat[get_index(ix, iy, Sx)]=elem_uprava_s_delenim(modul, 
						s_mat[get_index(ix, iy, Sx)],		// a_xy
						s_mat[get_index(ix, sdiagy, Sx)], // a_xp, muze byt z podmatice D
						m_py);
			// \ENDFOR
				}
		  // \ENDIF
			}
			iy+=bdim;
		// \ENDFOR
		}
		vypsat_mat<unsigned int>(Sx, Sx, &(s_mat[Sx*Sy]), NULL);
		cout << "-------------------" << endl;
		vypsat_mat<unsigned int>(Sx, Sy, s_mat, NULL);
		cout << "-------------------" << endl;
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
	}
// \ENDFOR
}
// S DELENIM
void compute_podmatice13_(int N, unsigned int modul, int sx, int sy, int Sx, int Sy, unsigned int* s_mat, unsigned int* actions)
{
	cout << "modul = " << modul << endl;
	for(int i=0;i<(Sx*Sy+Sx);i++)
	{
		actions[i]=modul;
	}
	// TODO: vyuzivat i podmatici s_mat1
	// podmatice s_mat: |-- podmatice, kterou pocitam (Sx*Sy cisel) --| |-- podmatice, kterou potrebuji (Sx^2 cisel) --|
	//unsigned int* s_mat1=
	int tid=0;
	int bdim=1;
	unsigned int m1=modul;
	unsigned long long pom;
	unsigned int minS=min( min(Sx, Sy), min(N-Sx*sx, N-Sy*sy) );
	int pod_diag_x=Sy*sy+Sy-Sx*sx;
// \FOR{$p$ := $1$ do $Sx$}
	for(unsigned int isloupec=0;isloupec<Sx;isloupec++)
	{
		cout << "====================\npivot=" << isloupec << endl;
		int gdiagx=Sx*sx+isloupec;
		
	}
}

/* celou matici rozdelim do obdelnikovych "podmatic", ktere budu postupne nahravat do sdilene pameti a pocitat
 * podmatice nemusi byt nutne ctvercova
 * zpusob zpracovani: 1, 2, 3, 4
 */
void GJE_podmatice(int N, unsigned int modul, unsigned int* m_matice, unsigned int* m_prava_strana, unsigned int* m_vys_jmenovatel)
{
	int Sx=5;
	int Sy=3;
	int Smin=min(Sx, Sy);
	unsigned int* s_matice=(unsigned int*)malloc(Sx*(Sy+Sx)*sizeof(unsigned int));
	unsigned int* actions=(unsigned int*)malloc((Sx*Sy+Smin)*sizeof(unsigned int));
#ifdef DELENI
	// todo: inicializace inverse, inverze vsech prvku zabere 2 až 4 GB pameti
#else
	// inverse := null
#endif

// \FOR{$p$ := $1$ do $\lceil\frac{N}{\min(S_x, S_y)}\rceil$}
	for(int ipivot=0;ipivot<ceil((double)N/Smin);ipivot++)
	{
		// DEBUG
		cout << endl << ipivot << endl;
	// \STATE \COMMENT{zpracovani radku, kde je Z=1}
	// \STATE nacist a spocitat $podmatice_{pp}$ \COMMENT{Z=1}
		copy_podmatice(N, ipivot, ipivot, Sx, Sy, s_matice, m_matice, m_prava_strana, COPY_MAT_B_GLOB_TO_A_SH);
		// todo: compute_podmatice1
		compute_podmatice13(N, modul, ipivot, ipivot, Sx, Sy, s_matice, actions);
		vypsat_mat<unsigned int>(Sx, Sy, s_matice, NULL);
		copy_podmatice(N, ipivot, ipivot, Sx, Sy, s_matice, m_matice, m_prava_strana, COPY_MAT_A_SH_TO_B_GLOB);
	// \FOR{$x$ := $p+1$ do $\lceil\frac{N+1}{S_x}\rceil$}
		for(int x=ipivot+1;x<ceil((double)(N+1)/Sx);x++)
		{
		// \STATE nacist a aplikovat operace v $actions$ na $podmatice_{xp}$ \COMMENT{Z=2}
			copy_podmatice(N, x, ipivot, Sx, Sy, s_matice, m_matice, m_prava_strana, COPY_MAT_B_GLOB_TO_A_SH);
			// todo: compute_podmatice2
			//for(int i=0;i<Sx*Sy;i++) s_matice[i]=2;
			compute_podmatice2(N, modul, x, ipivot, Sx, Sy, s_matice, actions);
			copy_podmatice(N, x, ipivot, Sx, Sy, s_matice, m_matice, m_prava_strana, COPY_MAT_A_SH_TO_B_GLOB);
		}
	//\ENDFOR
	// \STATE \COMMENT{zpracovani ostatnich radku}
	// \FOR{$y$ := $1$ do $\lceil\frac{N}{S_y}\rceil$}
		for(int y=0;y<=ceil((double)N/Sy);y++)
		{
		// \IF{$y$ != $p$}
			if(y!=ipivot)
			{
			// \STATE nacist a vynulovat $podmatice_{py}$; \COMMENT{Z=3}
				copy_podmatice(N, ipivot, y, Sx, Sy, s_matice, m_matice, m_prava_strana, COPY_MAT_B_GLOB_TO_A_SH);
				copy_podmatice(N, ipivot, ipivot, Sx, Sx, &(s_matice[Sx*Sy]), m_matice, m_prava_strana, COPY_MAT_B_GLOB_TO_A_SH);	// todo: nenacitat prvky, ktere uz jsou v s_matice
				// todo: compute_podmatice3
				//for(int i=0;i<Sx*Sy;i++) s_matice[i]=3;
				compute_podmatice13(N, modul, ipivot, y, Sx, Sy, s_matice, actions);
				copy_podmatice(N, ipivot, y, Sx, Sy, s_matice, m_matice, m_prava_strana, COPY_MAT_A_SH_TO_B_GLOB);
			// \FOR{$x$ := $p+1$ do $\lceil\frac{N+1}{S_x}\rceil$}
				for(int x=ipivot+1;x<ceil((double)(N+1)/Sx);x++)
				{
				// \STATE nacist a aplikovat operace v $actions$ na $podmatice_{xy}$; \COMMENT{Z=4}
					copy_podmatice(N, x, y, Sx, Sy, s_matice, m_matice, m_prava_strana, COPY_MAT_B_GLOB_TO_A_SH);
					// todo: compute_podmatice4
					for(int i=0;i<Sx*Sy;i++) s_matice[i]=4;
					copy_podmatice(N, x, y, Sx, Sy, s_matice, m_matice, m_prava_strana, COPY_MAT_A_SH_TO_B_GLOB);
				}
			// \ENDFOR
			}
		// \ENDIF
		}
		// DEBUG
		vypsat_mat<unsigned int>(N, N, m_matice, m_prava_strana);
	//\ENDFOR
	}
//\ENDFOR
}

unsigned int compute_inverse(unsigned int cislo, unsigned int modul)
{
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
/* inverse = [1;modul-1], size(inverse)=(modul-1)
 * inverzni k cislu A je inverse[A-1]
 */
void gener_inverse(unsigned int modul, unsigned int* inverse)
{
	unsigned int tid=(unsigned int)0;
	int bdim=1;	// blockDim.x;

	unsigned int cislo=tid;
	while(cislo<modul)
	{
		inverse[cislo]=0;
		cislo+=bdim;
	}
	
	cislo=tid+1;
	while(cislo<modul)
	{
		if(inverse[cislo]==0)
		{
			unsigned int inv=compute_inverse(cislo, modul);
			inverse[cislo]=inv;
			inverse[inv]=cislo;
		}
		cislo+=bdim;
	}
}
unsigned int get_inverse(unsigned int prvek, unsigned int* arr_inverse)
{
	if(arr_inverse==NULL) return 0;
	return arr_inverse[prvek];
}

int get_index(int X, int Y, int N)	// SLOUPEC, RADEK
{
	return Y*N+X;
}
