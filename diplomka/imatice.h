#ifndef IMATICE_H
#define IMATICE_H

template<class T>
class Imatice<T>
{
public:
	virtual Imatice();
	virtual Imatice(int size);
	virtual Imatice(int size, T value);
	virtual ~Imatice(void);
	
	virtual int get_size();
	virtual T get_cell(int X, int Y);	// SLOUPEC, RADEK
	virtual void fill_random(void);
	virtual void fill_hilbert(void);
	virtual void save_matrix(char* filename);
	virtual int load_matrix(char* filename);
	virtual void execute();
	virtual void vypsat();
};

#endif