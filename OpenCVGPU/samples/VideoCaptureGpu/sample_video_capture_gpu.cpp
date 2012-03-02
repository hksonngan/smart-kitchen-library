#define USE_VIDEO_CAPTURE_OPT_PARSER
#include "skl.h"
#include "sklcv.h"
#include "sklcvgpu.h"

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

	skl::gpu::VideoCaptureGpu cam;
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

	cv::namedWindow("image",0);
	cv::gpu::GpuMat image;
	cv::gpu::GpuMat edge;
	cv::Mat edge_cpu;
	while('q'!=cv::waitKey(10)){
		// set関数を呼ぶと非同期アップロードが無効化する
//		cam.set(skl::POS_FRAMES,0);
		cam >> image;
		if(image.empty()) break;
		cv::gpu::Scharr(image,edge,CV_8U,1,0);
		edge_cpu = cv::Mat(edge);
		cv::imshow("edge x",edge_cpu);
	}
	cv::destroyWindow("image");

	return EXIT_SUCCESS;
}
