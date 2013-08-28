#include "stdafx.h"
#include "kernels_cpu.h"
#include "templates_functions.h"
#include <cstdio>

using namespace std;


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
void gauss_jordan_elim_while(int N, unsigned int modul, unsigned int* m_matice, unsigned int* m_prava_strana, unsigned int* m_vys_jmenovatel)
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
		vypsat_mat(N, N, m_matice, m_prava_strana);
	}
	// ulozit diagonalu do m_vys_jmenovatel
	itid=tid;
	while(itid<N)
	{
		m_vys_jmenovatel[itid]=m_matice[get_index(itid, itid, N)];
		itid+=1;
	}
}

void gauss_jordan_elim_p1(int modul, int nx, int ny, int sx, int sy, unsigned int* s_matice, unsigned int* actions, unsigned int* diag_pivot, int zpusob_zprac)
/* N, modul - stejne jako v gauss_jordan_elim_..
 * n - velikost s_matice
 * s_matice - pole shared, submatice
 * actions - ny cisel (indexu) radku, kvuli vymene radku s nulovym cislem na diagonale
             nasleduje pole dvojic cisel, kteryma nasobim radek a pivota
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
			if(zpusob_zprac==1 && s_matice[get_index(ipivot, ipivot, nx)]==0)
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
				//  pivotni prvek neni "0" (zpusob_zprac==1) nebo mi tam nula nevadi (zpusob_zprac==2)
				actions[ipivot]=ipivot;	// "vymenit" sam se sebou
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
		int multipl2;
		itid=tid;
		while(itid<sy)	// prochazi jednotlive radky
		{
			int iact=ny+2*(itid+ny*ipivot);
			switch(zpusob_zprac)
			{
			case 1:	// supmatice je na dianogale
				if(itid==ipivot)	// prvek je na diagonale ve velke matici
				{
					actions[iact]=1;
					actions[iact+1]=0;
				}else
				{
					multipl1=s_matice[get_index(ipivot, ipivot, nx)];
					actions[iact]=multipl1;
					multipl2=s_matice[get_index(ipivot, itid, nx)];
					actions[iact+1]=multipl2;
				}
				cout << actions[iact] << "(" << (iact) << ")," << actions[iact+1] << "(" << (iact+1) << ") | ";
				break;
			case 2:
				multipl1=diag_pivot[ipivot];
				actions[iact]=multipl1;
				multipl2=s_matice[get_index(ipivot, itid, nx)];
				actions[iact+1]=multipl2;
				cout << actions[iact] << "(" << (iact) << ")," << actions[iact+1] << "(" << (iact+1) << ") | ";
				break;
			case 3:
				multipl1=actions[iact];
				multipl2=actions[iact+1];
				break;
			}
			if(actions[iact]!=1 || actions[iact+1]!=0)	// jinak upravuju radek a sloupec, ktery se protina na diagonale -> ten prvek je pivot - neupravuje se
			{
				long long pom;
				for(int iX=0;iX<sx;iX++)	// prochazi cisla v i1-tem radku
				{
					// TODO: atomicOperators
					long long m1=(long long)s_matice[get_index(iX, itid, nx)];
					long long m2;
					if(zpusob_zprac==2) m2=(long long)(iX==ipivot ? diag_pivot[iX] : 0);
					else m2=(long long)s_matice[get_index(iX, ipivot, nx)];
					pom = multipl1*m1-multipl2*m2;
					//pom=pom % modul;
					s_matice[get_index(iX, itid, nx)]=(int)pom;
				}
			}
			itid+=1;
		}
	}
}

void gauss_jordan_elim_part(int N, unsigned int modul, unsigned int* m_matice, unsigned int* m_prava_strana, unsigned int* m_vys_jmenovatel)
{
	// nahraje submatici do shared
	int nx, ny;	// velikost submatice
	nx = ny = 3;
	int sx, sy;	// indexy urcujici submatici ve velke matici
	int px, py;	// indexy urcujici pozici prvku v submatici
	// __shared__
	unsigned int* sub_m=(unsigned int*)malloc(nx*ny*sizeof(unsigned int));
	unsigned int* citatele=(unsigned int*)malloc((2*ny*ny+ny)*sizeof(unsigned int));
	unsigned int* diag_pivot=(unsigned int*)malloc(nx*sizeof(unsigned int));

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
			int nx1=min(N-sx*nx+1, nx);	// aktualni x-velikost submatice; +1 kvuli prave strane
			int ny1=min(N-sy*ny, ny);	// aktualni y-velikost submatice
			while(px<nx1)
			{
				int x1=sx*nx+px;	// x-index ve velke matici prvku, ktery prijde do submatice
				for(py=0;py<ny1;py++)
				{
					int y1=sy*ny+py;	// y-index ve velke matici prvku, ktery prijde do submatice
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
				if(x1<N) diag_pivot[px] = m_matice[get_index(x1, x1, N)];

				px++;
			}
			vypsat_mat<unsigned int>(nx, ny, sub_m, NULL);
			// spusti .._p1
			gauss_jordan_elim_p1(modul, nx, ny, nx1, ny1, sub_m, citatele, diag_pivot, zpusob_zpracovani);
			
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
			vypsat_mat(N, N, m_matice, m_prava_strana);


		}
	}
	free(sub_m);
	free(citatele);
	free(diag_pivot);
}

/* nacte/ulozi podmatici z globalni p. do sdilene nebo zpet
 * Sx, Sy - velikost podmatice, mela by se vejit do sdilene pameti
 * sx, sy - souradnice zvolene podmatice v matici, sx \in [0; ceil(N/Sx)]
 * mat_source, mat_dest - zdrojova resp. cilova adresa
 */
//#define COPY_TO_SHARED_MEM	1
//#define COPY_TO_GLOBAL_MEM	2
void copy_podmatice(int N, int sx, int sy, int Sx, int Sy, unsigned int* mat_shared, unsigned int* mat_global, unsigned int* prava_str, int copy_to)
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
					case COPY_TO_GLOBAL_MEM:
						mat_global[get_index(glob_x, glob_y, N)] = mat_shared[get_index(ix, itid, Sx)];
						break;
					case COPY_TO_SHARED_MEM:
						mat_shared[get_index(ix, itid, Sx)] = mat_global[get_index(glob_x, glob_y, N)];
						break;
					}
				}else
				{
					switch(copy_to)
					{
					case COPY_TO_GLOBAL_MEM:
						prava_str[glob_y] = mat_shared[get_index(ix, itid, Sx)];
						break;
					case COPY_TO_SHARED_MEM:
						mat_shared[get_index(ix, itid, Sx)] = prava_str[glob_y];
						break;
					}
				}
			}else
			{
				if(copy_to == COPY_TO_SHARED_MEM) mat_shared[get_index(ix, itid, Sx)] = 0;
			}
		}
		itid+=bdim;
	}
	if(copy_to == COPY_TO_GLOBAL_MEM)
		vypsat_vys<unsigned int>(N, prava_str, NULL);
}

void compute_podmatice1(int N, int Sx, int Sy, unsigned int* s_mat, unsigned int* actions)
{
	
}

/* celou matici rozdelim do obdelnikovych "podmatic", ktere budu postupne nahravat do sdilene pameti a pocitat
 * podmatice nemusi byt nutne ctvercova
 * zpusob zpracovani: 1, 2, 3, 4
 */
void GJE_podmatice(int N, unsigned int modul, unsigned int* m_matice, unsigned int* m_prava_strana, unsigned int* m_vys_jmenovatel)
{
	int Sx=3;
	int Sy=3;
	int Smin=min(Sx, Sy);
	unsigned int* s_matice=(unsigned int*)malloc(Sx*Sy*sizeof(unsigned int));
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
		copy_podmatice(N, ipivot, ipivot, Sx, Sy, s_matice, m_matice, m_prava_strana, COPY_TO_SHARED_MEM);
		// todo: compute_podmatice1
		//for(int i=0;i<Sx*Sy;i++) s_matice[i]=1;
		copy_podmatice(N, ipivot, ipivot, Sx, Sy, s_matice, m_matice, m_prava_strana, COPY_TO_GLOBAL_MEM);
	// \FOR{$x$ := $p+1$ do $\lceil\frac{N+1}{S_x}\rceil$}
		for(int x=ipivot+1;x<ceil((double)(N+1)/Sx);x++)
		{
		// \STATE nacist a aplikovat operace v $actions$ na $podmatice_{xp}$ \COMMENT{Z=2}
			copy_podmatice(N, x, ipivot, Sx, Sy, s_matice, m_matice, m_prava_strana, COPY_TO_SHARED_MEM);
			// todo: compute_podmatice2
			for(int i=0;i<Sx*Sy;i++) s_matice[i]=2;
			copy_podmatice(N, x, ipivot, Sx, Sy, s_matice, m_matice, m_prava_strana, COPY_TO_GLOBAL_MEM);
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
				copy_podmatice(N, ipivot, y, Sx, Sy, s_matice, m_matice, m_prava_strana, COPY_TO_SHARED_MEM);
				// todo: compute_podmatice3
				for(int i=0;i<Sx*Sy;i++) s_matice[i]=3;
				copy_podmatice(N, ipivot, y, Sx, Sy, s_matice, m_matice, m_prava_strana, COPY_TO_GLOBAL_MEM);
			// \FOR{$x$ := $p+1$ do $\lceil\frac{N+1}{S_x}\rceil$}
				for(int x=ipivot+1;x<ceil((double)(N+1)/Sx);x++)
				{
				// \STATE nacist a aplikovat operace v $actions$ na $podmatice_{xy}$; \COMMENT{Z=4}
					copy_podmatice(N, x, y, Sx, Sy, s_matice, m_matice, m_prava_strana, COPY_TO_SHARED_MEM);
					// todo: compute_podmatice4
					for(int i=0;i<Sx*Sy;i++) s_matice[i]=4;
					copy_podmatice(N, x, y, Sx, Sy, s_matice, m_matice, m_prava_strana, COPY_TO_GLOBAL_MEM);
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
