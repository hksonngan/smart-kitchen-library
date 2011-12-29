#include "sklutils.h"

namespace skl{
	/*
	 * @brief 平均0,標準偏差1のガウス分布を作成する
	 * */
	double gauss_rand(){
		double v1,v2,s;
		do{
			double u1 = static_cast<double>(rand())/RAND_MAX;
			double u2 = static_cast<double>(rand())/RAND_MAX;
			v1 = 2*u1-1;
			v2 = 2*u2-1;
			s = v1*v1 + v2*v2;
		}while(s>=1||s<0);

		return v1*sqrt(-2*log(s)/s);
	}

	double rayleigh_rand(double sigma){
		return sqrt(pow(gauss_rand()*sigma,2)+pow(gauss_rand()*sigma,2));
	}


}
