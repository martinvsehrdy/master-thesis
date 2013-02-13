// cuda-pokusy.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "cuda.cu"
#include <iostream>

using namespace std;

int _tmain(int argc, _TCHAR* argv[])
{
	int count=0;
	cudaGetDeviceCount( &count);
	cout << count << endl;

	VecAdd<<<1, N>>>(A, B, C);
	cin.get();
	return 0;
}

