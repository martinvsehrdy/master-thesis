#ifndef _KERNELS_CPU_H_
#define _KERNELS_CPU_H_


/* kernels.cpp */
void gener_primes();
unsigned int compute_inverse(unsigned int cislo, unsigned int modul);
unsigned int compute_inverse_eukleides(unsigned int cislo, unsigned int modul);
void gener_inverse(unsigned int modul, unsigned int* inverse);
unsigned int get_inverse(unsigned int prvek, unsigned int* arr_inverse);
void gauss_jordan_elim_for(int N, int modul, unsigned int* m_matice, unsigned int* m_prava_strana, unsigned int* m_vys_jmenovatel);
void gauss_jordan_elim_while(int Sx, int Sy, unsigned int modul, unsigned int* m_matice);
int get_index(int X, int Y, int N);

void GJE_podmatice(int N, unsigned int modul, unsigned int* m_matice, unsigned int* m_prava_strana, unsigned int* m_vys_jmenovatel);
void copy_podmatice(int N, int sx, int sy, int Sx, int Sy, unsigned int* mat_A, unsigned int* mat_B, unsigned int* prava_str, int copy_to);
#define COPY_MAT_B_GLOB_TO_A_SH	1
#define COPY_MAT_A_SH_TO_B_GLOB	2
#define COPY_MAT_A_SH_TO_B_SH 	3

#endif /* _KERNELS_CPU_H_ */


