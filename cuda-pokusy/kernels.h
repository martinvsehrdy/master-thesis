#ifndef _KERNELS_CU_
#define _KERNELS_CU_

#include <cuda_runtime.h>

/* kernesl.cu */
void cuda_gauss_jordan_elim(int N, int modul, int* m_matice, int* m_prava_strana, int* m_vys_jmenovatel);
void print_gpus_info(void);
void print_cuda_err(cudaError_t cudaErr);


#endif /* _KERNELS_CU_ */