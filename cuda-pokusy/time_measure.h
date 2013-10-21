#ifndef _TIME_MEASURE_H_
#define _TIME_MEASURE_H_


unsigned int get_milisec_from_startup(void);
float get_measured_time();
void start_measuring();
void stop_measuring();
// CUDA
float cuda_get_measured_time();
void cuda_start_measuring();
void cuda_stop_measuring();



#endif	/* _TIME_MEASURE_H_ */