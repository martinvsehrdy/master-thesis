#ifndef _KERNELS_CU_
#define _KERNELS_CU_

#include <cuda_runtime.h>

static unsigned int gpu_time;
static int num_of_gpu=0;
static cudaDeviceProp gpu_property;

/* kernels.cu */
void print_gpus_info(void);
void print_cuda_err(cudaError_t cudaErr);
void init_gpu_compute(void);

void cuda_GJE_podmatice(int N, unsigned int modul, unsigned int* m_matice, unsigned int* m_prava_strana, unsigned int zpusob);
void cuda_GJE_global(int N, unsigned int modul, unsigned int* m_matice, unsigned int zpusob);
void test_elem_uprava(int N, unsigned int modul, unsigned int zpusob);


#endif /* _KERNELS_CU_ */