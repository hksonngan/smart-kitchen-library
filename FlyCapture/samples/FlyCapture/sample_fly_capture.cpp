#define USE_VIDEO_CAPTURE_OPT_PARSER
#include "skl.h"
#include "sklcv.h"
#include "VideoCaptureFlyCap2.h"
#include <sstream>

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

	skl::FlyCapture cam;
	skl::VideoParams params;

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
	for(size_t i=0;i<cam.size();i++){
		cam.set(params);
		std::cout << "Camera Parameter Settings for cam " << i << std::endl;
		std::cout << cam.get();
	}

	if(!cam.isOpened()){
		std::cerr << "ERROR: failed to open video." << std::endl;
		std::cerr << "       Maybe, camera is not connected to the PC??" << std::endl;
		return EXIT_FAILURE;
	}

	std::stringstream ss;
	for(size_t i=0;i<cam.size();i++){
		ss << "image" << i;
		cv::namedWindow(ss.str(),0);
		ss.str("");
	}
	std::vector<cv::Mat> images(cam.size());
	while('q'!=cv::waitKey(10)){
		if(cam.size()==1){
			cam >> images[0];
			cv::imshow("image0",images[0]);
		}
		else{
			assert(cam.grab());
			for(size_t i=0;i<cam.size();i++){
				cam[i] >> images[i];
				if(images[i].empty()) break;

				ss << "image" << i;
				cv::imshow(ss.str(),images[i]);
				ss.str("");
			}
		}
	}
	cv::destroyAllWindows();

	return EXIT_SUCCESS;
}
