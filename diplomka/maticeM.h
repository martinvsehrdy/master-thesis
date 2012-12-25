#pragma once
#include "matice.h"

template<class T>
class maticeM<T> :
	public matice<T>
{
private:

public:
	maticeM();
	maticeM(int size);
	maticeM(int size, T value);
	~maticeM(void);
	
	int get_size();
	int set_cell(int X, int Y, T value);
	T get_cell(int X, int Y);	// SLOUPEC, RADEK
	void fill_random(void);
	void fill_hilbert(void);
	void save_matrix(char* filename);
	int load_matrix(char* filename);
	void execute();
	void vypsat();
};

