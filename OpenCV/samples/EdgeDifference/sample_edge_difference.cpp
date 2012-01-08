#include <iostream>
#include <highgui.h>
#include "skl.h"
#include "sklcv.h"

opt_on(double,thresh1,16,"","<DOUBLE>","1st threshold for Canny Filter");
opt_on(double,thresh2,32,"","<DOUBLE>","2nd threshold for Canny Filter");
opt_on(int,aperture_size,3,"","<INT>","aperture_size for Canny Filter");
opt_on(int,dilate_size,4,"","<INT>","dilation size for remove bg/fg edge.");

opt_on_bool(visualize,"-v","show source/destination images on window");
opt_on(std::string, output_filename, "", "-o","<FILE>", "set output file name");

int main(int argc, char* argv[]){
	skl::OptParser options;
	std::vector<std::string> args;
	opt_parse(options,argc,argv,args);

	cv::Mat mat1(cv::Size(3,3),CV_8UC1);
	cv::Mat mat2(mat1.size(),mat1.type());
	for(int y=0;y<mat1.rows;y++){
		for(int x=0;x<mat1.cols;x++){
			mat1.at<unsigned char>(y,x) = mat1.cols * y + x;
			mat2.at<unsigned char>(y,x) = mat1.cols * (mat1.rows - (y+1)) + (mat1.cols - (x+1));
		}
	}
	cv::Mat mat3 = mat1 - mat2;
	for(int y=0;y<mat1.rows;y++){
		for(int x=0;x<mat1.cols;x++){
			std::cerr << (int)mat1.at<unsigned char>(y,x) << " - " << (int)mat2.at<unsigned char>(y,x) << " = " << (int)mat3.at<unsigned char>(y,x) << std::endl;
		}
	}


	// user need to derect -v option or -o option to get result.
	if(options.help() || args.size() < 3 || (!visualize && output_filename.empty())){
		std::cerr << "Usage: " << argv[0] << " src1.img src2.img [-v | -o <FILE>]" << std::endl;
		options.usage();
		return -1;
	}

	cv::Mat src1 = cv::imread(argv[1],-1);
	cv::Mat src2 = cv::imread(argv[2],-1);


	cv::Mat edge1,edge2;
	skl::edge_difference(src1,src2,edge1,edge2,thresh1,thresh2,aperture_size,dilate_size);

	if(visualize){
		cv::namedWindow("src1",0);
		cv::imshow("src1",src1);
		cv::namedWindow("src2",0);
		cv::imshow("src2",src2);
		cv::namedWindow("edge1",0);
		cv::imshow("edge1",edge1);
		cv::namedWindow("edge2",0);
		cv::imshow("edge2",edge2);

		cv::waitKey(-1);
		cv::destroyAllWindows();
	}
	if(!output_filename.empty()){
		std::string dirname = skl::DirName(output_filename);
		if(!dirname.empty()){
			dirname += "/";
		}
		std::string extname = skl::ExtName(output_filename);
		std::string basename = skl::BaseName(output_filename,extname);
		cv::imwrite(dirname + basename + "1" + extname,edge1);
		cv::imwrite(dirname + basename + "2" + extname,edge2);
	}
	return 1;
}
