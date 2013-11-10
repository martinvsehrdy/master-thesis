#ifndef _COMMON_H_
#define _COMMON_H_

#define COPY_MAT_B_GLOB_TO_A_SH	1
#define COPY_MAT_A_SH_TO_B_GLOB	2
#define COPY_MAT_A_SH_TO_B_SH 	3
#define COPY_MAT_BEZ_DELENI		0x0010	// 4.bit oznacuje zda nacitat/ukladat diagonalni prvek na konec upravovanych radku za pravou stranu

#define ZPUSOB_WF			0x0001	// 0. bit for/while(0) while/for(1)
#define ZPUSOB_S_DELENIM	0x0002	// 1. bit bez deleni(0) s delenim(1)
#define ZPUSOB_VLAKNA		0x000C	// 2.3.bit \t1(00) 32(01) 128(10) vláken
#define ZPUSOB_CPU			0x0010	// 4.bit \tGPU(0) CPU(1)
#define ZPUSOB_GLOBAL_MEM	0x0020	// 5.bit \tmatice v sdilene(0), globalni(1) pameti
#define ZPUSOB_RADKY		0x0040	// 6.bit \tmetoda: podmatice(0), radky(1)
#define PODMATICE_12		0x8000	// 15.bit rezervovan na rozliseni podmatice12 a 34

#ifndef min
#define min(a,b)	(a<b ? a : b)
#endif

#ifndef max
#define max(a,b)	(a>b ? a : b)
#endif

#endif /* _COMMON_H_ */