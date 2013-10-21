#ifndef _COMMON_H_
#define _COMMON_H_

#define COPY_MAT_B_GLOB_TO_A_SH	1
#define COPY_MAT_A_SH_TO_B_GLOB	2
#define COPY_MAT_A_SH_TO_B_SH 	3

#define ZPUSOB_WF			0x0001	// 0. bit for/while(0) while/for(1)
#define ZPUSOB_S_DELENIM	0x0002	// 1. bit bez deleni(0) s delenim(1)
#define ZPUSOB_VLAKNA		0x000C	// 2.3.bit \t1(00) 32(01) 128(10) vláken
#define ZPUSOB_CPU			0x0010	// 4.bit \tGPU(0) CPU(1)


#endif /* _COMMON_H_ */