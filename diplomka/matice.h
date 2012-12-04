#pragma once
class matice
{
private:
	double* pointer;
	int N;
	double* prava_strana;
public:
	matice();
	matice(int size);
	matice(int size, double value);
	~matice(void);
	
	int get_size();
	int set_cell(int X, int Y, double value);
	double get_cell(int X, int Y);	// SLOUPEC, RADEK
	int set_cell1(int ind, double value);
	double get_cell1(int ind);
	void fill_random(void);
	void save_matrix(char* filename);
	int load_matrix(char* filename);
	void do_gauss(void);
	void vypsat();
};

