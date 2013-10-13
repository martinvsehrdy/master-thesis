#ifndef _KERNELS_CU_
#define _KERNELS_CU_

#include <cuda_runtime.h>

static unsigned int gpu_time;

/* kernels.cu */
void print_gpus_info(void);
void print_cuda_err(cudaError_t cudaErr);
void init_gpu_compute(void);

void cuda_GJE_while(int N, int modul, unsigned int* m_matice, unsigned int* m_prava_strana);
int get_index(int X, int Y, int N);
void copy_podmatice(int N, int sx, int sy, int Sx, int Sy, unsigned int* mat_A, unsigned int* mat_B, unsigned int* prava_str, int copy_to);
#define COPY_MAT_B_GLOB_TO_A_SH	1
#define COPY_MAT_A_SH_TO_B_GLOB	2
#define COPY_MAT_A_SH_TO_B_SH 	3

#endif /* _KERNELS_CU_ */