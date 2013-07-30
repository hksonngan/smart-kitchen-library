#define USE_VIDEO_CAPTURE_OPT_PARSER
#include "skl.h"
#include "sklcv.h"
#include "sklflycap.h"
#include <sstream>

opt_on(std::string, camera_setting, "", "-C","<FILE>","load camera conf parameters.");
opt_on(std::string, input_file,"","-i","<FILE>","load video file");

opt_on(std::string, output_dir,"","-o","<DIR>","direct directory for saving images");
opt_on_bool(time_stamp,"-t","save images with time stamp filename.");

int main(int argc,char* argv[]){
	skl::OptParser options;
	std::vector<std::string> args;
	opt_parse(options,argc,argv,args);

	if(options.help()){
		options.usage();
		return EXIT_FAILURE;
	}

	skl::FlyCapture cam;
	skl::VideoCaptureParams params;

	if(!camera_setting.empty()){
		params.load(camera_setting);
	}
	// CAUTION: call opt_parse_cap_prop after opt_parse
	opt_parse_cap_prop(params);



	if(input_file.empty()){
		if(!cam.open()) return EXIT_FAILURE;
	}
	else{
		cam.open(input_file);
	}

	// CAUTION: set params after open the camera.
	for(size_t i=0;i<cam.size();i++){
		cam.set(params);
		std::cout << "=== Parameter Settings of Camera ===" << i << std::endl;
		std::cout << cam[i].get();
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

	int frame_num = 0;
	while('q'!=cv::waitKey(10) && cam.grab() && cam.isOpened()){

		for(size_t i=0;i<cam.size();i++){
			cam[i].retrieve(images[i]);
		}

		for(size_t i=0;i<cam.size();i++){
			ss << "image" << i;
			cv::imshow(ss.str(),images[i]);
			if(!output_dir.empty()){
				if(time_stamp){
					ss << "_" << skl::Time::now() << ".bmp";
				}
				else{
					ss << "_" << std::setw(6) << std::setfill('0') << frame_num << ".bmp";
				}
				cv::imwrite(output_dir + "/" + ss.str(),images[i]);
			}
			ss.str("");
			frame_num++;
		}
	}
	cv::destroyAllWindows();

	return EXIT_SUCCESS;
}
