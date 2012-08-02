#define USE_VIDEO_CAPTURE_OPT_PARSER
#include "skl.h"
#include "sklcv.h"
#include "sklflycap.h"
#include <sstream>
#include "opencv2/gpu/gpu.hpp"

#include "sklcvgpu.h"

opt_on(std::string, camera_setting, "", "-C","<FILE>","load camera conf parameters.");
opt_on(std::string, input_file,"","-i","<FILE>","load video file");
opt_on(std::string, output_dir,"","-o","<DIR>","save image files");
opt_on(float,alpha,1.5,"-a","<FLOAT>","set TexCut parameter alpha");
opt_on(float,smoothing_term_weight,1.0,"-s","<FLOAT>","set TexCut parameter smoothing_term_weight");
opt_on(float,thresh_tex_diff,0.4,"-t","<FLOAT>","set TexCut parameter thresh_tex_diff");
opt_on(unsigned char,over_exposure_thresh,248,"","<UCHAR>","set TexCut parameter over_exposure_thresh");
opt_on(unsigned char,under_exposure_thresh,8,"","<UCHAR>","set TexCut parameter under_exposure_thresh");

opt_on_bool(do_cpu,"","do TexCut on CPU for comparison");
opt_on(bool,smooth,true,"","<BOOL>","do smoothing before TexCut.");

skl::TexCut* bgs_algo_cpu;

int main(int argc,char* argv[]){
	skl::OptParser options;
	std::vector<std::string> args;
	opt_parse(options,argc,argv,args);

	if(options.help()){
		options.usage();
		return EXIT_FAILURE;
	}

	skl::gpu::VideoCaptureGpu cam(new skl::FlyCapture());
	skl::VideoCaptureParams params;

	if(!camera_setting.empty()){
		params.load(camera_setting);
	}
	// CAUTION: call opt_parse_cap_prop after opt_parse
	opt_parse_cap_prop(params);



	if(input_file.empty()){
		cam.open(0);
	}
	else{
		cam.open(input_file);
	}

	// CAUTION: set params after open the camera.
	cam.set(params);
	std::cout << "=== Parameter Settings of Camera ===" << std::endl;
	std::cout << cam.get();

	if(!cam.isOpened()){
		std::cerr << "ERROR: failed to open video." << std::endl;
		std::cerr << "       Maybe, camera is not connected to the PC??" << std::endl;
		return EXIT_FAILURE;
	}

	std::stringstream ss;
	cv::namedWindow("image",0);
	cv::namedWindow("result",0);
		
	if(do_cpu){
		cv::namedWindow("result_cpu",0);
		ss.str("");
	}
	cv::Mat mat,result,result_cpu;
	cv::gpu::GpuMat gpu_mat;

	// prepare background subtraction algorithms
	skl::gpu::TexCut bgs_algo;
	bgs_algo.doSmoothing(smooth);
	if(do_cpu){
		bgs_algo_cpu = new skl::TexCut();
		bgs_algo_cpu->doSmoothing(smooth);
	}


	// get first background image
	cam >> gpu_mat;
	bgs_algo.setParams(alpha,smoothing_term_weight,thresh_tex_diff,over_exposure_thresh,under_exposure_thresh);
	bgs_algo.setBackground(gpu_mat);
	if(do_cpu){
		bgs_algo_cpu->setParams(alpha,smoothing_term_weight,thresh_tex_diff,over_exposure_thresh,under_exposure_thresh);
		cam.retrieve(mat);
		bgs_algo_cpu->setBackground(mat);
	}

	// get second background image
	cam >> gpu_mat;
	bgs_algo.learnImageNoiseModel(gpu_mat);
	if(do_cpu){
		cam.retrieve(mat);
		bgs_algo_cpu->learnImageNoiseModel(mat);
	}


	int frame = 0;
	skl::StopWatch swatch;
	while('q'!=cv::waitKey(10)){
		std::cout << "frame: " << frame++ << std::endl;
		if(!cam.grab()) break;
		if(!cam.retrieve(gpu_mat)) break;
		if(!cam.isOpened()||gpu_mat.cols==0||gpu_mat.rows==0) break;

		cam.retrieve(mat);
		cv::imshow("image",mat);


		if(do_cpu){
			swatch.start();
			bgs_algo_cpu->compute(mat,result_cpu);
			std::cerr << "CPU time: " << swatch.lap() << std::endl;
			cv::imshow("result_cpu",result_cpu);
		}
		swatch.start();
		bgs_algo.compute(gpu_mat,result);
		std::cerr << "GPU time: " << swatch.lap() << std::endl;
		if(!output_dir.empty()){
			ss << std::setw(6) << std::setfill('0') << frame;
			cv::imwrite(output_dir + "/raw_" + ss.str() + ".bmp",mat);
			cv::imwrite(output_dir + "/fg_mask_" + ss.str() + ".bmp",result*255);
			ss.str("");
		}

		cv::imshow("result",result*255);
//		bgs_algo.setBackground(gpu_mat);
	}
	cv::destroyAllWindows();
	if(do_cpu){
		delete bgs_algo_cpu;
	}
	return EXIT_SUCCESS;
}
