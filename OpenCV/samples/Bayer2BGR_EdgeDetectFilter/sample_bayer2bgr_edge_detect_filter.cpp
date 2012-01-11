#include <iostream>
#include "skl.h"
#include "sklcv.h"

int main(int argc, char* argv[]){
	if(argc < 2){
		std::cerr << "Usage: " << argv[0] << " bayer.img output.img" << std::endl;
	}
	std::string filename(argv[1]);
	cv::Mat bayer = cv::imread(filename,0);
	cv::Mat image;
	skl::cvtBayer2BGR(bayer,image,CV_BayerGB2BGR,skl::BAYER_EDGE_SENSE);
	cv::imwrite(filename, image);
	return EXIT_SUCCESS;
}
