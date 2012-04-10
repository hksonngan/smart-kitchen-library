#include <iostream>
#include <highgui.h>
#include "skl.h"
#include "sklcv.h"

opt_on_bool(visualize,"-v","show source/destination images on window");
opt_on_bool(gray_scale,"-g","read source images as gray scale image");
opt_on(std::string, output_filename, "", "-o","<FILE>", "set output file name");
opt_on_bool(float_mask,"-f","input float mask image");

int main(int argc, char* argv[]){
	skl::OptParser options;
	std::vector<std::string> args;
	opt_parse(options,argc,argv,args);

	// user need to derect -v option or -o option to get result.
	if(options.help() || args.size() < 4 || (!visualize && output_filename.empty())){
		std::cerr << "Usage: " << argv[0] << " src1.img src2.img mask.img [-v | -o <FILE>]" << std::endl;
		options.usage();
		return -1;
	}

	int readAsColor = 1;
	if(gray_scale){
		std::cerr << "flag 'gray scale' on!" << std::endl;
		readAsColor = 0;
	}

	cv::Mat src1 = cv::imread(args[1],readAsColor);
	cv::Mat src2 = cv::imread(args[2],readAsColor);
	cv::Mat mask = cv::imread(args[3],0);

	if(float_mask){
		std::cerr << "flag 'float mask' on!" << std::endl;
		mask.convertTo(mask, CV_32FC1, 1.0f/255.0f);
	}

	cv::Mat dest;

	if(gray_scale){
		if(float_mask){
			dest = skl::blending<unsigned char,float>(src1,src2,mask);
		}
		else{
			dest = skl::blending<unsigned char,unsigned char>(src1,src2,mask);
		}
	}
	else{
		if(float_mask){
			dest = skl::blending<cv::Vec3b,float>(src1,src2,mask);
		}
		else{
			dest = skl::blending<cv::Vec3b,unsigned char>(src1,src2,mask);
		}
	}

	if(visualize){
		cv::namedWindow("src1",0);
		cv::imshow("src1",src1);
		cv::namedWindow("src2",0);
		cv::imshow("src2",src2);
		cv::namedWindow("mask",0);
		cv::imshow("mask",mask);
		cv::namedWindow("dest",0);
		cv::imshow("dest",dest);

		cv::waitKey(-1);
		cv::destroyAllWindows();
	}
	if(!output_filename.empty()){
		cv::imwrite(output_filename,dest);
	}
	return 1;
}
