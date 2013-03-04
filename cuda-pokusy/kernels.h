#ifndef _KERNELS_CU_
#define _KERNELS_CU_

#include <cuda_runtime.h>

/* kernesl.cu */
__global__ void cuda_kernel(int N, int modul,  int* m_matice, int* m_prava_strana, int* m_vys_jmenovatel);
void cuda_gauss_jordan_elim(int N, int modul, int* m_matice, int* m_prava_strana, int* m_vys_jmenovatel);
__device__ int cuda_get_index(int X, int Y, int N);
__device__ int cuda_m_inv(int modulo, int cislo);
void print_gpus_info(void);
void print_cuda_err(cudaError_t cudaErr);


#endif /* _KERNELS_CU_ */