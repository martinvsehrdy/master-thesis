#include "stdafx.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <list>
#include <cuda.h>
#include "kernels.h"
#include "kernels_cpu.h"
#include <cuda_runtime.h>

using namespace std;

#define POC_OPAKOVANI 5
float cuda_time1, cuda_time2, cuda_time3;
cudaError cudaErr;

void statistic(list<float> l, float* quartal1, float* quartal2, float* quartal3, float* avg)
{
	l.sort();
	unsigned int poc=0;
	*avg=0.0;
	list<float> l1, l2;
	while(l.size()>0)
	{
		*avg+=l.front();
		poc++;
		if(l.size()>1)	// pro pripad, ze v seznamu zbude jeden posledni prvek
		{
			*avg+=l.back();
			poc++;
		}
		l1.push_back(l.front());
		l2.push_front(l.back());
		l.pop_back();
		l.pop_front();
	}
	if(poc>0) *avg/=poc;
	if(l1.size()>0 && l2.size()>0) *quartal2=(l1.back() + l2.front())/2.0;
	else *quartal2=0;
	while(l1.size()>2 && l2.size()>2)
	{
		l1.pop_back();
		l1.pop_front();
		l2.pop_back();
		l2.pop_front();
	}
	poc=0;
	*quartal1=0;
	*quartal3=0;
	for(list<float>::iterator iter=l1.begin();iter!=l1.end();iter++)
	{
		*quartal1+=*iter;
		poc++;
	}
	if(poc>0) *quartal1/=poc;
	poc=0;
	for(list<float>::iterator iter=l2.begin();iter!=l2.end();iter++)
	{
		*quartal3+=*iter;
		poc++;
	}
	if(poc>0) *quartal3/=poc;
	
}
int main(int argc, char** argv)
// argv[0] <N> <modul>
{
	int N;
	int modul;
	if(argc>2)
	{
		N=atoi(argv[1]);
		modul=atoi(argv[2]);
	}else
	{
		cout << "#Program spustte ve tvaru:" << argv[0] << " <N> <modul>" << endl;
		cout << "#Vystup: <velikost N> <na GPU [ms]>\t<z GPU [ms]>\tprumer\tnejrychlejsi\t1.quartal\tmedian\t3.quartal\tnejpomalejsi\t<celkem [ms]>" << endl;
		return 0;
	}
	int* A=new int[N*N];
	int* b=new int[N];
	int* jm=new int[N];
	int vysledek=8;
	// inicializace CUDA
	init_gpu_compute();
	int* cudaVysl;
	cudaErr=cudaMalloc((void**)&cudaVysl, sizeof(int));
	int* cuda_A;
	cudaErr=cudaMalloc((void**)&cuda_A, N*N*sizeof(int));
	int* cuda_b=NULL;
	cudaErr=cudaMalloc((void**)&cuda_b, N*sizeof(int));
	int* cuda_jm=NULL;
	cudaErr=cudaMalloc((void**)&cuda_jm, N*sizeof(int));
	int poc_casu=3;
	cudaEvent_t event1, event2;
	cudaErr=cudaEventCreate(&event1);
	cudaErr=cudaEventCreate(&event2);
	list<float>* time_arr=new list<float>[poc_casu];

	int poc_opakovani=POC_OPAKOVANI;
	for(int opakovani=0;opakovani<1000 && poc_opakovani>0;opakovani++)
	{
		for(int i=0;i<N*N;i++) A[i]=rand() % modul;
		for(int i=0;i<N;i++) { b[i]=i; }

		//load_matrix(&N, &A, &b, "../diplomka/mat-int.txt");
		//cout << N << endl;
		//vypsat_mat(N, A, b);
	
		// kopirovani na GPU
		cudaErr=cudaEventRecord(event1, 0);
		cudaErr=cudaMemcpy(cuda_A, A, N*N*sizeof(int), cudaMemcpyHostToDevice);
		cudaErr=cudaMemcpy(cuda_b, b, N*sizeof(int), cudaMemcpyHostToDevice);
		cudaErr=cudaEventRecord(event2, 0);
		cudaErr=cudaEventSynchronize(event2);
		cudaErr=cudaEventElapsedTime(&cuda_time1, event1, event2);
		// vypocet v CUDA
		cudaErr=cudaEventRecord(event1, 0);
		cuda_gauss_jordan_elim(N, modul, cuda_A, cuda_b, cuda_jm, cudaVysl);
		cudaErr=cudaEventRecord(event2, 0);
		cudaErr=cudaEventSynchronize(event2);
		cudaEventElapsedTime(&cuda_time2, event1, event2);
		// kopirovani z GPU
		cudaErr=cudaEventRecord(event1, 0);
		cudaErr=cudaMemcpy(&vysledek, cudaVysl, sizeof(int), cudaMemcpyDeviceToHost);
		cudaErr=cudaMemcpy(b, cuda_b, N*sizeof(int), cudaMemcpyDeviceToHost);
		cudaErr=cudaMemcpy(jm, cuda_jm, N*sizeof(int), cudaMemcpyDeviceToHost);
		cudaErr=cudaEventRecord(event2, 0);
		cudaErr=cudaEventSynchronize(event2);
		cudaErr=cudaEventElapsedTime(&cuda_time3, event1, event2);
		// kontrola regularity
		if (vysledek==0)
		{
			poc_opakovani--;
			time_arr[0].push_back(cuda_time1);
			time_arr[2].push_back(cuda_time2);
			time_arr[1].push_back(cuda_time3);
		}
		//cout << "vysledek z CUDA vypoctu (" << opakovani << "): " << vysledek << endl;

	}
	// zjistovani casu
	
	if(time_arr[0].size()>POC_OPAKOVANI)
	{
		cout << "# nedostatecny pocet mereni: " << time_arr[0].size() << " misto " << POC_OPAKOVANI << endl;
	}
	float suma=0.0;
	cout << N << "\t";
	for(int i=0;i<poc_casu;i++)
	{
		time_arr[i].sort();
		float q1, q3, avg;
		float med;
		statistic(time_arr[i], &q1, &med, &q3, &avg);
		if(i==2) // jedna se o samotnej vypocet => vypisu podrobne
		{
			cout << avg << "\t" << time_arr[i].front() << "\t" << q1 << "\t" << med << "\t" << q3 << "\t" << time_arr[i].back() << "\t";
		}else
		{
			cout << med << "\t";
		}
		suma+=med;
	}
	cout << suma << endl;


	//print_gpus_info();
	cudaFree(cuda_A);
	cudaFree(cuda_b);
	cudaFree(cuda_jm);
#ifdef _DEBUG
	vypsat_vys(N, b, jm);
	cin.get();
#else
	delete[] A;
	delete[] b;
	delete[] jm;
#endif
}