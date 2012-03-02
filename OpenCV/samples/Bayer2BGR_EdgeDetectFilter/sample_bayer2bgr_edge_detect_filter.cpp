#include <iostream>
#include "skl.h"
#include "sklcv.h"

int main(int argc, char* argv[]){
	if(argc < 2){
		std::cerr << "Usage: " << argv[0] << " bayer.img output.img" << std::endl;
		return EXIT_FAILURE;
	}
	std::string input_filename(argv[1]);
	cv::Mat bayer = cv::imread(input_filename,0);
	cv::Mat image;
	skl::cvtBayer2BGR(bayer,image,CV_BayerGB2BGR,skl::BAYER_EDGE_SENSE);
	std::string output_filename(argv[2]);
	cv::imwrite(output_filename, image);
	return EXIT_SUCCESS;
}
