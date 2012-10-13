#include "skl.h"
#include "sklcv.h"
#include <sstream>

opt_on(std::string, output_file,"","-o","<DIR>","save image file");

opt_on(float, sigma,0.5,"-s","<FLOAT>","parameter for smoothing preprocess.")
opt_on(float, unit_segment_size,300,"-k","<FLOAT>","parameter for deciding unit of homogenious block.");
opt_on(int, min_size, 30,"-m","<UINT>","minimum segment size");


int main(int argc,char* argv[]){
	skl::OptParser options;
	std::vector<std::string> args;
	opt_parse(options,argc,argv,args);

	if(options.help() || args.size()<2){
		std::cerr << "USAGE: " << args[0] << " input.img" << std::endl;
		options.usage();
		return EXIT_FAILURE;
	}
	std::string input_file = args[1];

	cv::Mat src,label;
	src = cv::imread(input_file,1);
	assert(checkMat(src,CV_8U,3));

	// prepare the algorithm
	skl::RegionLabelingEfficientGraph region_labeling_algorithm(sigma,unit_segment_size,min_size);

	size_t region_num = region_labeling_algorithm.compute(src,label);
	if(region_num >= SHRT_MAX){
		std::cerr << "ERROR: too many segments. Change the parameter to get corser segments." << std::endl;
		return EXIT_FAILURE;
	}
	assert(checkMat(label,CV_16S,1,src.size()));

	std::cerr << "INFO: the image is segmented into " << region_num << " regions." << std::endl;

	// convert "label IDs" to "label colors" for visualization.
	cv::Mat visualized_segments = skl::visualizeRegionLabel( label, region_num);
	assert(checkMat(visualized_segments,CV_8U,3,src.size()));

	// save
	if(!output_file.empty()){
		cv::imwrite(output_file,visualized_segments);
	}

	// show on display
	cv::namedWindow("source",0);
	cv::namedWindow("segmentation result",0);

	cv::imshow("source",src);
	cv::imshow("segmentation result",visualized_segments);

	cv::waitKey(-1);

	cv::destroyAllWindows();
	return EXIT_SUCCESS;
}
