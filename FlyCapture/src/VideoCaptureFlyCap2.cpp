/*!
 * @file FlyCapture.cpp
 * @author a_hasimoto
 * @date Date Created: 2012/Jan/12
 * @date Last Change: 2012/Jan/17.
 */
#include "VideoCaptureFlyCap2.h"

#define CHECK_ERROR(error) checkError(error, __FILE__, __LINE__)
using namespace skl;

FlyCapture2::BusManager* FlyCapture::busMgr(NULL);
/*!
 * @brief デフォルトコンストラクタ
 */
FlyCapture::FlyCapture():is_file_loading(false),is_opened(false),is_started(false){
}

/*!
 * @brief デストラクタ
 */
FlyCapture::~FlyCapture(){

}

VideoCapture& FlyCapture::operator>> (cv::Mat& image){
	if(!grab()){
		image.release();
	}
	else{
		retrieve(image);
	}
	return *this;
}

bool FlyCapture::open(const std::string& filename){
	release();
	is_file_loading = true;
	cam_interface.resize(1);
	cam_interface[0] = new VideoCaptureCameraInterface(this);
	return is_opened = VideoCapture::open(filename);
}

bool FlyCapture::open(int device){
	is_file_loading = false;
	return is_opened = initialize(device);
}

bool FlyCapture::isOpened()const{
	return is_opened;
}
void FlyCapture::release(){
	// delete cam_interface always when release is called for simplicity.
	for(size_t i=0;i<cam_interface.size();i++){
		delete cam_interface[i];
	}
	cam_interface.clear();

	if(!is_opened)return;

	if(!is_file_loading){
		for(size_t i=0;i<numCameras;i++){
			ppCameras[i]->StopCapture();
			ppCameras[i]->Disconnect();
			delete ppCameras[i];
		}
		delete busMgr;
		delete[] ppCameras;
		camImages.clear();
		camInfo.clear();
	}

	is_opened = false;
	is_file_loading = false;
	is_started = false;
	return;
}

bool FlyCapture::initialize(int device){
	FlyCapture2::Error error;

	if(busMgr == NULL){

#ifdef DEBUG
		PrintBuildInfo();
#endif // DEBUG

		busMgr = new FlyCapture2::BusManager();
		error = busMgr->GetNumOfCameras(&numCameras);
		if(!CHECK_ERROR(error)) return false;

#ifdef DEBUG
		printf( "Number of flycapture cameras: %u\n", numCameras );
#endif // DEBUG

	}

	ppCameras = new FlyCapture2::Camera*[numCameras];
	cam_interface.resize(numCameras);
	camInfo.resize(numCameras);
	camImages.resize(numCameras);
	for(size_t i=0;i<numCameras;i++){
		ppCameras[i] = new FlyCapture2::Camera();
		cam_interface[i] = new FlyCaptureCameraInterface(this,i);

		FlyCapture2::PGRGuid guid;
		error = busMgr->GetCameraFromIndex( i, &guid );
		if(!CHECK_ERROR(error)) return false;

		// Connect to cameras
		error = ppCameras[i]->Connect( &guid );
		if(!CHECK_ERROR(error)) return false;

		// Get the camera information
		error = ppCameras[i]->GetCameraInfo( &camInfo[i] );
		if(!CHECK_ERROR(error)) return false;

#ifdef DEBUG
		PrintCameraInfo(i);
#endif

		error = ppCameras[i]->SetVideoModeAndFrameRate(
				FlyCapture2::VIDEOMODE_1024x768Y8,
				FlyCapture2::FRAMERATE_15);
		if(!CHECK_ERROR(error)) return false;

		(*this)[i].set(CONVERT_RGB,1.0);
	}

	// Y8でBayerPattern画像を取得するために
	// Registerをいじる。
	return true;
}

bool FlyCapture::grab(){
	if(is_file_loading){
		return skl::VideoCapture::grab();
	}

	if(!is_started) sync_capture_start();

	return grab_flycap();
}

bool FlyCapture::retrieve(cv::Mat& image,int channel){
	if(is_file_loading){
		return skl::VideoCapture::retrieve(image,channel);
	}
	switch(numCameras){
		case 0:
			return false;
		case 1:
			if(channel == 0) return develop(image,0);
			else return false;
		default:
			channel--;
			if(numCameras <= static_cast<unsigned int>(channel)) return false;
			return develop(image,channel);
	}
}



