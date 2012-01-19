#define USE_VIDEO_CAPTURE_OPT_PARSER
#include "skl.h"
#include "sklcv.h"
#include <sstream>

opt_on(std::string, camera_setting, "", "-C","<FILE>","load camera conf parameters.");

std::vector<std::string> input_files;
opt_on_container(std::vector,std::string, input_files,"-i","<FILE> | <FILE:FILE:...>","load video file",":",-1);

std::vector<int> devices(1,0);
opt_on_container(std::vector,int,devices,"-d","<DEVICE_ID>|<DEV_ID:DEV_ID:...>","direct device id.",":",-1);

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



	if(input_files.empty()){
		assert(!devices.empty());
		assert(cam.open(devices.begin(),devices.end()));
	}
	else{
		assert(cam.open(input_files.begin(),input_files.end()));
	}

	// CAUTION: set params after open the camera.
	for(size_t i=0;i<cam.size();i++){
		cam[i].set(params);
		std::cout << "=== Parameter Setting of camera " << i << " ===" << std::endl;
		std::cout << cam[i].get();
		std::cerr << std::endl;
	}


	if(!cam.isOpened()){
		std::cerr << "ERROR: failed to open video." << std::endl;
		std::cerr << "       Maybe, camera is not connected to the PC??" << std::endl;
		return EXIT_FAILURE;
	}

	std::stringstream ss;
	for(size_t i=0;i<cam.size();i++){
		ss << "image " << i;
		cv::namedWindow(ss.str(),0);
		ss.str("");
	}

	std::vector<cv::Mat> images;
	while('q'!=cv::waitKey(10)){
//		if(!cam.grab()) break;
		cam >> images;
		if(images.empty()) break;

		for(size_t i=0;i<cam.size();i++){
			ss << "image " << i;
//			cam[i].retrieve(images[i]);
			cv::imshow(ss.str(),images[i]);
			ss.str("");
		}
	}
	cv::destroyAllWindows();

	return EXIT_SUCCESS;
}
