#define USE_VIDEO_CAPTURE_OPT_PARSER
#include "sklcv.h"

opt_on(std::string, camera_setting, "", "-C","<FILE>","load camera conf parameters.");
opt_on(std::string, input_file,"","-i","<FILE>","load video file");
opt_on(int,dev,0,"-d","<DEVICE_ID>","direct device id.");

// params for background cut
opt_on(float,thresh_bg,2,"","<FLOAT>","parameter thresh_bg");
opt_on(float,thresh_fg,5,"","<FLOAT>","parameter thresh_fg");
opt_on(float,sigma_KL,0.1,"","<FLOAT>","parameter sigma_KL");
opt_on(float,paramK,5,"-K","<FLOAT>","parameter K");
opt_on(float,sigma_z,10,"","<FLOAT>","parameter sigma_z");
opt_on(float,learning_rate,0.2,"","<FLOAT>","parameter learning_rate");
opt_on(int,bg_cluster_num,15,"","<INT>","parameter bg_cluster_num");
opt_on(int,fg_cluster_num,5,"","<INT>","parameter fg_cluster_num");


int main(int argc,char* argv[]){
	skl::OptParser options;
	std::vector<std::string> args;
	opt_parse(options,argc,argv,args);

	if(options.help()){
		options.usage();
		return EXIT_FAILURE;
	}

	skl::VideoCapture cam;
	skl::VideoCaptureParams params;

	if(!camera_setting.empty()){
		params.load(camera_setting);
	}
	// CAUTION: call opt_parse_cap_prop after opt_parse
	opt_parse_cap_prop(params);



	if(input_file.empty()){
		cam.open(dev);
	}
	else{
		cam.open(input_file);
	}

	// CAUTION: set params after open the camera.
	cam.set(params);

	std::cout << "Camera Parameter Settings" << std::endl;
	std::cout << cam.get();

	if(!cam.isOpened()){
		std::cerr << "ERROR: failed to open video." << std::endl;
		std::cerr << "       Maybe, camera is not connected to the PC??" << std::endl;
		return EXIT_FAILURE;
	}

	cv::namedWindow("result",0);
	cv::namedWindow("background",0);
	cv::namedWindow("image",0);
	cv::Mat image,result;

	skl::BackgroundCut bgs_algo(thresh_bg,thresh_fg,sigma_KL,paramK,sigma_z,learning_rate,bg_cluster_num,fg_cluster_num);
	cam >> image;
	bgs_algo.background(image);


	while('q'!=cv::waitKey(10)){
		cam >> image;
		if(image.empty()) break;
		cv::imshow("image",image);
		bgs_algo.compute(image,result);
		cv::imshow("result",result);
		cv::imshow("background",bgs_algo.background());
//		bgs_algo.update(image);
	}
	cv::destroyWindow("image");

	return EXIT_SUCCESS;
}
