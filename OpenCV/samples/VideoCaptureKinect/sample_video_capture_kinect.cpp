#define USE_VIDEO_CAPTURE_OPT_PARSER
#include "skl.h"
#include "sklcv.h"

opt_on(std::string, camera_setting, "", "-C","<FILE>","load camera conf parameters.");
opt_on(std::string, input_file,"","-i","<FILE>","load video file");


opt_on(int,dev,0,"-d","<DEVICE_ID>","direct device id.");

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
		cam.open(CV_CAP_OPENNI + dev);
//		cam.set(skl::OPENNI_IMAGE_GENERATOR,true);
//		cam.set(skl::OPENNI_DEPTH_GENERATOR,true);
	}
	else{
		cam.open(input_file);
	}

	// CAUTION: set params after open the camera.
	cam.set(params);

//	std::cout << "Camera Parameter Settings" << std::endl;
//	std::cout << cam.get();

	if(!cam.isOpened()){
		std::cerr << "ERROR: failed to open video." << std::endl;
		std::cerr << "       Maybe, camera is not connected to the PC??" << std::endl;
		return EXIT_FAILURE;
	}

	cv::namedWindow("image",0);
	cv::namedWindow("depth",0);
	cv::namedWindow("valid depth mask",0);
	cv::Mat image, depth, valid_depth_mask;


	while('q'!=cv::waitKey(10) && cam.grab()){
		cam.retrieve(depth,CV_CAP_OPENNI_DEPTH_MAP) ;
		assert(checkMat(depth,CV_16U,1));
		cam.retrieve(image,CV_CAP_OPENNI_BGR_IMAGE) ;
		assert(checkMat(image,CV_8U,3,depth.size()));
		cam.retrieve(valid_depth_mask,CV_CAP_OPENNI_VALID_DEPTH_MASK);
		assert(checkMat(valid_depth_mask,CV_8U,1,depth.size()));

		cv::imshow("image",image);
		cv::imshow("depth",depth);
		cv::imshow("valid depth mask",valid_depth_mask);
	}
	cv::destroyWindow("image");
	cv::destroyWindow("depth");
	cv::destroyWindow("valid depth mask");
	return EXIT_SUCCESS;
}
