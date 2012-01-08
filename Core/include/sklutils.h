#ifndef __SKL_UTILS__
#define __SKL_UTILS__

#include <cstdlib>
#include <cmath>
namespace skl{
	double gauss_rand();
	double rayleigh_rand(double sigma);
#ifdef DEBUG
#define printLocation() std::cerr << "in " << __FILE__ << ": at " << __LINE__ << std::endl
#else
#define printLocation() 
#endif
}
#endif // __SKL_UTILS__
