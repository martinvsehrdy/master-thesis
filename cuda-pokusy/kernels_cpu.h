#ifndef _KERNELS_CPU_H_
#define _KERNELS_CPU_H_
#include "common.h"

/* kernels.cpp */
unsigned int compute_inverse(unsigned int cislo, unsigned int modul);
unsigned int compute_inverse_eukleides(unsigned int cislo, unsigned int modul);
void copy_podmatice(int N, int gx, int gy, int Sx, int Sy, unsigned int* mat_A, unsigned int* mat_B, unsigned int* prava_str, int copy_to);
void gauss_jordan_elim_for(int N, int modul, unsigned int* m_matice, unsigned int* m_prava_strana, unsigned int zpusob);
void gauss_jordan_elim_while(int Sx, int Sy, unsigned int modul, unsigned int* m_matice);
int get_index(int X, int Y, int N);

void GJE_podmatice(int N, unsigned int modul, unsigned int* m_matice, unsigned int* m_prava_strana, unsigned int* m_vys_jmenovatel, unsigned int zpusob);
//void copy_podmatice(int N, int sx, int sy, int Sx, int Sy, unsigned int* mat_A, unsigned int* mat_B, unsigned int* prava_str, int typ_podmatice, int copy_to);

#endif /* _KERNELS_CPU_H_ */


