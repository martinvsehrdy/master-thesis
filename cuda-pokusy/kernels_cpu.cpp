#include "stdafx.h"
#include "kernels_cpu.h"
#include "templates_functions.h"
#include <cstdio>

using namespace std;

//#define S_DELENIM

unsigned int elem_uprava_s_delenim(unsigned int modul, unsigned int a_xy, unsigned int a_xp, unsigned int a_py);	// \STATE $a_{xy} := a_{xy} - a_xp \cdot a_py$
unsigned int elem_uprava_bez_deleni(unsigned int modul, unsigned int a_xy, unsigned int a_pp, unsigned int a_xp, unsigned int a_py);	// \STATE $a_{xy} := a_{xy} \cdot a_pp - a_xp \cdot a_py$

/* 
 * gauss-jordanova eliminace, jednovlaknova, ve for-cyklech, primo na datech ve vstupnim poli, 
 * bez deleni - nasobim oba mergujici radky, po vypoctu kazde bunky se moduluje
 */
void gauss_jordan_elim_for(int N, int modul, unsigned int* m_matice, unsigned int* m_prava_strana, unsigned int* m_vys_jmenovatel)
{
	// TODO: posouvat cisla v radcich doleva, kvuli CUDA, aby se pristupovalo stale na ty stejna mista v pameti, 
	//       vysledek bude v prvnim sloupci matice
	for(int ipivot=0;ipivot<N;ipivot++)
	{
		cout << ipivot << endl;
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
					if(m_vys_jmenovatel!=NULL) m_vys_jmenovatel[i]=1;
				}
				return;
			}
		}
#ifdef S_DELENIM
		unsigned int a_pp_inv = compute_inverse(m_matice[get_index(ipivot, ipivot, N)], modul);
		cout << endl << "vydelit " << a_pp_inv << ": ";
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

#else
		unsigned int a_pp = m_matice[get_index(ipivot, ipivot, N)];
		cout << endl << a_pp << ": ";
#endif
		for(int iY=0;iY<N;iY++)	// prochazi jednotlive radky
		{
			if(iY==ipivot) continue;
			unsigned int a_py = m_matice[get_index(ipivot, iY, N)];
			cout << a_py << ", ";
			for(int iX=0;iX<N;iX++)	// prochazi cisla v i1-tem radku
			{
				unsigned int a_xy = m_matice[get_index(iX, iY, N)];
				unsigned int a_xp = m_matice[get_index(iX, ipivot, N)];
#ifdef S_DELENIM
				m_matice[get_index(iX, iY, N)]=elem_uprava_s_delenim(modul, a_xy, a_xp, a_py);
#else
				m_matice[get_index(iX, iY, N)]=elem_uprava_bez_deleni(modul, a_xy, a_pp, a_xp, a_py);
#endif

			}
#ifdef S_DELENIM
			m_prava_strana[iY]=elem_uprava_s_delenim(modul, m_prava_strana[iY], m_prava_strana[ipivot], a_py);
#else
			m_prava_strana[iY]=elem_uprava_bez_deleni(modul, m_prava_strana[iY], a_pp, m_prava_strana[ipivot], a_py);
#endif

		}
		//cout << "pivot: " << ipivot << endl;
		//vypsat_matlab(N, m_matice, m_prava_strana);
		vypsat_mat(N, N, m_matice, m_prava_strana);
		cout << endl;
	}
	// ulozit diagonalu do m_vys_jmenovatel
#ifndef S_DELENIM
	unsigned long long pom;
	for(int i=0;i<N;i++)
	{
		pom = m_prava_strana[i];
		pom *= compute_inverse(m_matice[get_index(i, i, N)], modul);
		pom %= modul;
		m_prava_strana[i] = pom;
	}
#endif
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
		unsigned int a_pp_inv = compute_inverse(m_matice[get_index(ipivot, ipivot, Sx)], modul);
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
		pom *= compute_inverse(m_matice[get_index(i, i, Sx)], modul);
		pom %= modul;
		m_matice[get_index(Sx-1, i, Sx)] = pom;
	}
#endif
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
				if(copy_to == COPY_TO_SHARED_MEM)
				{
					//if( sx==sy && ix==itid )
					//mat_shared[get_index(ix, itid, Sx)] = 1;
					//else
					mat_shared[get_index(ix, itid, Sx)] = 0;
				}
			}
		}
		itid+=bdim;
	}
	if(copy_to == COPY_TO_GLOBAL_MEM)
		vypsat_vys<unsigned int>(N, prava_str, NULL);
}
// S DELENIM
void compute_podmatice1(int N, unsigned int modul, int sx, int sy, int Sx, int Sy, unsigned int* s_mat, unsigned int* actions)
{
	// todo: moznost zjistit u prvku podmatice souradnice ve velke matici
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
/* celou matici rozdelim do obdelnikovych "podmatic", ktere budu postupne nahravat do sdilene pameti a pocitat
 * podmatice nemusi byt nutne ctvercova
 * zpusob zpracovani: 1, 2, 3, 4
 */
void GJE_podmatice(int N, unsigned int modul, unsigned int* m_matice, unsigned int* m_prava_strana, unsigned int* m_vys_jmenovatel)
{
	int Sx=4;
	int Sy=Sx;
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
		compute_podmatice1(N, modul, ipivot, ipivot, Sx, Sy, s_matice, actions);
		vypsat_mat<unsigned int>(Sx, Sy, s_matice, NULL);
		copy_podmatice(N, ipivot, ipivot, Sx, Sy, s_matice, m_matice, m_prava_strana, COPY_TO_GLOBAL_MEM);
	// \FOR{$x$ := $p+1$ do $\lceil\frac{N+1}{S_x}\rceil$}
		for(int x=ipivot+1;x<ceil((double)(N+1)/Sx);x++)
		{
		// \STATE nacist a aplikovat operace v $actions$ na $podmatice_{xp}$ \COMMENT{Z=2}
			copy_podmatice(N, x, ipivot, Sx, Sy, s_matice, m_matice, m_prava_strana, COPY_TO_SHARED_MEM);
			// todo: compute_podmatice2
			//for(int i=0;i<Sx*Sy;i++) s_matice[i]=2;
			compute_podmatice2(N, modul, x, ipivot, Sx, Sy, s_matice, actions);
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
