#define USE_VIDEO_CAPTURE_OPT_PARSER
#include "skl.h"
#include "sklcv.h"

opt_on(std::string, camera_setting, "", "-C","<FILE>","load camera conf parameters.");
opt_on(std::string, input_file,"","-i","<FILE>","load video file");

opt_on(std::string, output_file,"","-o","<FILE>","save video");

opt_on(int,dev,0,"-d","<DEVICE_ID>","direct device id.");
opt_on_bool(trackbar,"","create track bar with the window.");

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
	int pos_frames;
	if(trackbar && !input_file.empty()){
		cv::createTrackbar("pos","image",&pos_frames, cam.get(skl::FRAME_COUNT)-1);
	}
	cv::Mat image, depth;

	cv::VideoWriter writer;
	if(!output_file.empty()){
		cam >> image;
		if(!writer.open(
					output_file,
					CV_FOURCC('D','V','I','X'),
					cam.get(skl::FPS),
					image.size(),
					(cam.get(skl::MONOCROME)<=0))){
			std::cerr << "WARNING: failed to open " << output_file << ".";
		}
		writer << image;
	}

	while('q'!=cv::waitKey(10) && cam.grab()){
		if(trackbar && !input_file.empty()){
			int frame_id = cv::getTrackbarPos("pos","image");
			cam.set(skl::POS_FRAMES,frame_id);
		}
		cam.retrieve(depth,CV_CAP_OPENNI_DEPTH_MAP) ;
		cam.retrieve(image,CV_CAP_OPENNI_BGR_IMAGE) ;
		if(writer.isOpened()){
			writer << image;
		}
		cv::imshow("image",image);
		cv::imshow("depth",depth);
	}
	cv::destroyWindow("image");
	cv::destroyWindow("depth");

	return EXIT_SUCCESS;
}
