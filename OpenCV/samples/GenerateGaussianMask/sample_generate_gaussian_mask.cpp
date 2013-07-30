#include "skl.h"
#include "sklcv.h"

double _covariance[4] = {2,0,0,1};
std::vector<double> covariance(_covariance,_covariance+4);
opt_on_container(std::vector,double,covariance,"","<DBL:DBL:DBL:DBL>","set covariance. elems are m_{0,0}:m_{0,1}:m_{1,0}:m_{1,1}.",":",4);



int ___main(int argc,char* argv[]){
	skl::OptParser options;
	std::vector<std::string> args;
	opt_parse(options,argc,argv,args);

	if(options.help()){
		options.usage();
		return EXIT_FAILURE;
	}

	cv::Mat covariance_mat(cv::Size(2,2),CV_64FC1,&covariance[0]);

	cv::Mat gaussian_mask = skl::generateGaussianMask(covariance_mat);
	std::cout << "mask size: " << gaussian_mask.cols << "x" << gaussian_mask.rows << std::endl;
//	std::cout << gaussian_mask << std::endl;
	std::cout << "sum of mask: " << cv::sum(gaussian_mask)[0] << std::endl;
	cv::namedWindow("gaussian mask",0);
	cv::imshow("gaussian mask",gaussian_mask*10);
	cv::waitKey(-1);

	return EXIT_SUCCESS;
}

