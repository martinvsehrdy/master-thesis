//#ifdef WIN32
#include "stdafx.h"
//#include <Windows.h>
#include <time.h>
//#endif
#include <cuda_runtime.h>

clock_t start_time;
clock_t stop_time;
cudaEvent_t cuda_start,cuda_stop;

int poc_vlaken;
int poc_bloku;
int poc_Sx;
int poc_Sy;

void set_pocty(int b, int t, int Sx, int Sy)
{
	poc_vlaken=t;
	poc_bloku=b;
	poc_Sx=Sx;
	poc_Sy=Sy;
}
void get_pocty(int* b, int* t, int* Sx, int* Sy)
{
	if(t!=NULL) *t = poc_vlaken;
	if(b!=NULL) *b = poc_bloku;
	if(Sx!=NULL) *Sx = poc_Sx;
	if(Sy!=NULL) *Sy = poc_Sy;
}


clock_t get_milisec_from_startup(void)
{
	unsigned int retval=0;
#ifdef WIN32
	//retval = (unsigned int) GetTickCount();
	retval = clock();
#endif

#ifdef UNIX
	// TODO: pouziti fce gettimeofday()
#endif

	return retval;
}
float get_measured_time()
{
	return (float)((stop_time - start_time));
}
void start_measuring()
{
	start_time = get_milisec_from_startup();
}
void stop_measuring()
{
	stop_time = get_milisec_from_startup();
}
// CUDA
float cuda_get_measured_time()
{
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime,cuda_start,cuda_stop);
	return elapsedTime;
}
void cuda_start_measuring()
{
	cudaEventRecord(cuda_start,0);
	// cudaSuccess, cudaErrorInvalidValue, cudaErrorInitializationError, cudaErrorInvalidResourceHandle, cudaErrorLaunchFailure
}
void cuda_stop_measuring()
{
	cudaEventRecord(cuda_stop,0);
	cudaEventSynchronize(cuda_stop);
}