#ifndef _COMMON_H_
#define _COMMON_H_

#define COPY_MAT_B_GLOB_TO_A_SH	1
#define COPY_MAT_A_SH_TO_B_GLOB	2
#define COPY_MAT_A_SH_TO_B_SH 	3
#define COPY_MAT_BEZ_DELENI		0x0010	// 4.bit oznacuje zda nacitat/ukladat diagonalni prvek na konec upravovanych radku za pravou stranu

#define ZPUSOB_WF			0x000001	// 0. bit for/while(0) while/for(1)
#define ZPUSOB_S_DELENIM	0x000002	// 1. bit bez deleni(0) s delenim(1)
#define ZPUSOB_VLAKNA		0x00000C	// 2.3.bit \t1(00) 32(01) 128(10) vláken
#define ZPUSOB_CPU			0x000010	// 4.bit \tGPU(0) CPU(1)
#define ZPUSOB_GLOBAL_MEM	0x000020	// 5.bit \tmatice v sdilene(0), globalni(1) pameti
#define ZPUSOB_RADKY		0x000040	// 6.bit \tmetoda: podmatice(0), radky(1)
#define ZPUSOB_POMER		0x0F0000
// zpusoby zpracovani pro Radkovou metodu
#define ZPUSOB_GLOB_PRISTUP	0x000100	// pristup vlaken z bloku do globalni pameti, vlakno zpracovava radek(0) nebo sloupec(1)
#define ZPUSOB_CUDA_UPRAVA	0x000200	// deleni bude v integer(0) nebo pomoci CUDA fci(1)
#define ZPUSOB_HILBERT_MAT	0x000400	// Hilbertova matice, jinak bude tridiagonalni

#define PODMATICE_12		0x008000	// 15.bit rezervovan na rozliseni podmatice12 a 34

#ifndef min
#define min(a,b)	(a<b ? a : b)
#endif

#ifndef max
#define max(a,b)	(a>b ? a : b)
#endif

#endif /* _COMMON_H_ */