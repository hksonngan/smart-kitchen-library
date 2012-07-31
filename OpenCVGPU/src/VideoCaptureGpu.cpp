/*!
 * @file VideoCaptureGpu.cpp
 * @author a_hasimoto
 * @date Date Created: 2012/Feb/10
 * @date Last Change: 2012/Feb/17.
 */
#include "VideoCaptureGpu.h"
#include "sklcv.h"
using namespace skl;
using namespace skl::gpu;

/*!
 * @brief デフォルトコンストラクタ
*/
VideoCaptureGpu::VideoCaptureGpu(cv::Ptr<_VideoCaptureInterface> video_capture_cpu):video_capture_cpu(video_capture_cpu),isNextFrameUploaded(false),_switch(false){
}


/*!
 * @brief デストラクタ
 */
VideoCaptureGpu::~VideoCaptureGpu(){
	release();
}

bool VideoCaptureGpu::grab(){
	if(!isNextFrameUploaded){
		if(!video_capture_cpu->grab()){
			return false;
		}
		if(!video_capture_cpu->retrieve(switching_mat_cpu[!_switch])){
			return false;
		}
//		std::cerr << switching_mat_cpu[!_switch].cols << " x " << switching_mat_cpu[!_switch].rows << std::endl;
		cv::gpu::ensureSizeIsEnough(switching_mat_cpu[!_switch].size(),switching_mat_cpu[!_switch].type(),switching_mat[!_switch]);
		s.enqueueUpload(switching_mat_cpu[!_switch],switching_mat[!_switch]);
	}

	// get NextFrame;
	s.waitForCompletion();
	_switch = !_switch;
	if(!video_capture_cpu->grab()){
		isNextFrameUploaded = false;
		return true;
	}
	if(!video_capture_cpu->retrieve(switching_mat_cpu[!_switch])){
		isNextFrameUploaded = false;
		return true;
	}
//	std::cerr << switching_mat_cpu[!_switch].cols << " x " << switching_mat_cpu[!_switch].rows << std::endl;
	cv::gpu::ensureSizeIsEnough(switching_mat_cpu[!_switch].size(),switching_mat_cpu[!_switch].type(),switching_mat[!_switch]);
	s.enqueueUpload(switching_mat_cpu[!_switch],switching_mat[!_switch]);
	isNextFrameUploaded = true;
	return true;
}


