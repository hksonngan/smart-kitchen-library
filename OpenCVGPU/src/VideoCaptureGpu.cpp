/*!
 * @file VideoCaptureGpu.cpp
 * @author a_hasimoto
 * @date Date Created: 2012/Feb/10
 * @date Last Change: 2012/Jul/31.
 */
#include "VideoCaptureGpu.h"
#include "sklcv.h"
using namespace skl;
using namespace skl::gpu;

/*!
 * @brief デフォルトコンストラクタ
*/
VideoCaptureGpu::VideoCaptureGpu(cv::Ptr<_VideoCaptureInterface> video_capture_cpu):video_capture_cpu(video_capture_cpu),isNextFrameUploaded(false),_switch(false){
	if(this->video_capture_cpu.empty()){
		this->video_capture_cpu = cv::Ptr<_VideoCaptureInterface>(new skl::VideoCapture());
	}
}


/*!
 * @brief デストラクタ
 */
VideoCaptureGpu::~VideoCaptureGpu(){
	release();
}

bool VideoCaptureGpu::grab(){
	if(!isNextFrameUploaded){
		if(!grabNextFrame()){
			return false;
		}
	}

	// get NextFrame;
	s.waitForCompletion();
	_switch = !_switch;

	if(!grabNextFrame()){
		return false;
	}
	isNextFrameUploaded = true;
	return true;
}

bool VideoCaptureGpu::grabNextFrame(){
	bool next_frame = !_switch;
	if(!video_capture_cpu->grab()){
		isNextFrameUploaded = false;
		return false;
	}
	if(!video_capture_cpu->retrieve(switching_mat_cpu[next_frame])){
		isNextFrameUploaded = false;
		return false;
	}
//	std::cerr << switching_mat_cpu[next_frame].cols << " x " << switching_mat_cpu[next_frame].rows << std::endl;
	cv::gpu::ensureSizeIsEnough(switching_mat_cpu[next_frame].size(),switching_mat_cpu[next_frame].type(),switching_mat[next_frame]);
	s.enqueueUpload(switching_mat_cpu[next_frame],switching_mat[next_frame]);
	return true;
}


