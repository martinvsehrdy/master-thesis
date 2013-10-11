//#ifdef WIN32
#include "stdafx.h"
#include <Windows.h>
//#endif

unsigned int get_milisec_from_startup(void)
{
	unsigned int retval=0;
#ifdef WIN32
	retval = (unsigned int) GetTickCount();
#endif

#ifdef UNIX
	// TODO: pouziti fce gettimeofday()
#endif

	return retval;
}
