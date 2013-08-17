#ifndef _KERNELS_CPU_H_
#define _KERNELS_CPU_H_


/* kernels.cpp */
void gener_primes();
void gener_inverse(unsigned int modul, unsigned int* inverse);
unsigned int get_inverse(unsigned int prvek, unsigned int* arr_inverse);
void gauss_jordan_elim_for(int N, int modul, unsigned int* m_matice, unsigned int* m_prava_strana, unsigned int* m_vys_jmenovatel);
void gauss_jordan_elim_while(int N, unsigned int modul, unsigned int* m_matice, unsigned int* m_prava_strana, unsigned int* m_vys_jmenovatel);
void gauss_jordan_elim_part(int N, unsigned int modul, unsigned int* m_matice, unsigned int* m_prava_strana, unsigned int* m_vys_jmenovatel);
void cpu_kernel1(int N, int modul, unsigned int* m_matice, unsigned int* m_prava_strana, unsigned int* m_vys_jmenovatel);
int get_index(int X, int Y, int N);



#endif /* _KERNELS_CPU_H_ */


