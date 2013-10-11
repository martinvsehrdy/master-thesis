#ifndef _KERNELS_CU_
#define _KERNELS_CU_

#include <cuda_runtime.h>

/* kernesl.cu */
void print_gpus_info(void);
void print_cuda_err(cudaError_t cudaErr);
void init_gpu_compute(void);

#endif /* _KERNELS_CU_ */