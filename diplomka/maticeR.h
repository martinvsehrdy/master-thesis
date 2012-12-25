#ifndef MATICER_H
#define MATICER_H

template<class T>
class maticeR<T> : public matice<T>
{
private:
	T* pointer;
	int N;
	T* prava_strana;
public:
	maticeR();
	maticeR(int size);
	maticeR(int size, T value);
	~maticeR(void);
	
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



#endif