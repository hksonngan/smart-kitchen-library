#define USE_VIDEO_CAPTURE_OPT_PARSER
#include "skl.h"
#include "sklcv.h"
#include "sklcvgpu.h"

opt_on(std::string, camera_setting, "", "-C","<FILE>","load camera conf parameters.");
opt_on(std::string, input_file,"","-i","<FILE>","load video file");
opt_on(int,dev,0,"-d","<DEVICE_ID>","direct device id.");

// options for optical flow

opt_on(float, _alpha,0.187f,"-a", "<FLOAT>","flow smoothiness");
opt_on(float, _gamma,50.0f,"-g","<FLOAT>","gradient constancy importance");
opt_on(float, scale_factor, 0.8f,"-s","<FLOAT>","pyramid scale factor");
opt_on(int, inner_iterations, 10, "","<INT>","number of lagged non-lineariity iterations (inner loop)");

opt_on(int, outer_iterations, 77, "","<INT>","number of warping iterations (number of pyramid levels)");

opt_on(int, solver_iterations, 10, "","<INT>","number of linear system solver iterations");

opt_on(int, waiting_time,10,"-w","<MSEC|-1>","set waiting time (milli second). the process stops at every frame with -1.");

opt_on(std::string, output, "","-o","<FILE_FORMAT>","Output format for saving the visualized flow.(e.g. output/test.png => saved as output/test_\%d.png");
opt_on_bool(silent,"","do not open image windows");

void convFloatMat2UCharMat(cv::Mat fmat,cv::Mat& ucmat,float scale){
	fmat.convertTo(ucmat,CV_8UC1,scale);
}

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
	params.set(skl::MONOCROME,1);
	cam.set(params);

	std::cout << "Camera Parameter Settings" << std::endl;
	std::cout << cam.get();

	if(!cam.isOpened()){
		std::cerr << "ERROR: failed to open video." << std::endl;
		std::cerr << "       Maybe, camera is not connected to the PC??" << std::endl;
		return EXIT_FAILURE;
	}

	if(!silent){
		cv::namedWindow("raw img",0);
		cv::namedWindow("prev",0);
		cv::namedWindow("flow",0);
		cv::namedWindow("distance",0);
		cv::namedWindow("angle",0);
	}
	cv::gpu::GpuMat temp,temp_gray,temp_div,image, prev,swap;
	cv::gpu::GpuMat u, v;

	skl::Flow flow;
	cv::Mat result, visualize,imgcpu,prevcpu;
	cv::Mat distance,angle;

	// OpticalFlow calculator
	cv::gpu::BroxOpticalFlow of_algo(_alpha,_gamma,scale_factor,inner_iterations, outer_iterations, solver_iterations);

	cam >> temp;
	if(temp.channels()>1){
		cv::gpu::cvtColor(temp,temp_gray,CV_BGR2GRAY);
	}
	else{
		temp_gray = temp.clone();
	}
	temp_gray.convertTo(temp_div,CV_32FC1,1);
	cv::gpu::divide(temp_div,255,prev);
	if(prev.empty()) return EXIT_FAILURE;


	while('q'!=cv::waitKey(waiting_time)){
		cam >> temp;
		if(temp.channels()>1){
			cv::gpu::cvtColor(temp,temp_gray,CV_BGR2GRAY);
		}
		else{
			temp_gray = temp.clone();
		}
		temp_gray.convertTo(temp_div,CV_32FC1,1);
		cv::gpu::divide(temp_div,255,image);
		if(image.empty()) break;

		of_algo(prev,image,u,v);

		flow.u = cv::Mat(u);
		flow.v = cv::Mat(v);
		convFloatMat2UCharMat(cv::Mat(image),imgcpu,255);
		visualize = flow.visualize(imgcpu,8,cv::Scalar(0,0,255));
		if(!output.empty()){
			std::string extname = skl::ExtName(output);
			int pos_frames = cam.get("POS_FRAMES");
			std::stringstream ss;
			ss << skl::DirName(output) << "/" << skl::BaseName(output,extname) << "_" << std::setw(6) << std::setfill('0') << pos_frames;
			std::cerr << ss.str() + extname << std::endl;
			cv::imwrite(ss.str() + extname,visualize);
			flow.write(ss.str()+".flow");
		}

		if(!silent){
			cv::imshow("flow",visualize);
			convFloatMat2UCharMat(cv::Mat(prev),prevcpu,255);

			cv::imshow("raw img",imgcpu);
			cv::imshow("prev",prevcpu);
			convFloatMat2UCharMat(flow.distance(),distance,8);
			convFloatMat2UCharMat(flow.angle(),angle,255.0/(2*M_PI));
			cv::imshow("distance",distance);
			cv::imshow("angle",angle);
			cv::imwrite("flow.png",visualize);
			cv::imwrite("prev.png",prevcpu);
			cv::imwrite("raw.png",imgcpu);
			cv::imwrite("distance.png",distance);
			cv::imwrite("angle.png",angle);
		}
		prev = image.clone();
	}
	if(!silent){
		cv::destroyAllWindows();
	}

	return EXIT_SUCCESS;
}

