#ifndef _KERNELS_CPU_H_
#define _KERNELS_CPU_H_

#define TYPE	int

/* kernels.cpp */
//template<class TYPE>
int load_matrix(int* N, TYPE** matice, TYPE** prava_strana, char* filename);
void gener_primes();
void gener_inverse(int modul, int* inverse);
void gauss_jordan_elim_for(int N, int modul, int* m_matice, int* m_prava_strana, int* m_vys_jmenovatel);
void gauss_jordan_elim_while(int N, int modul, int* m_matice, int* m_prava_strana, int* m_vys_jmenovatel);
void gauss_jordan_elim_part(int N, int modul, int* m_matice, int* m_prava_strana, int* m_vys_jmenovatel);
void cpu_kernel1(int N, int modul,  int* m_matice, int* m_prava_strana, int* m_vys_jmenovatel);
//int get_index(int X, int Y, int N);

void vypsat_mat(int nx, int ny, TYPE* matice, TYPE* prava_strana);
void vypsat_vys(int N, TYPE* citatel, TYPE* jmenovatel);
void vypsat_matlab(int nx, int ny, TYPE* matice, TYPE* prava_strana);

#endif /* _KERNELS_CPU_H_ */


