#include "VideoCaptureFlyCap2.h"
using namespace skl;
#define CHECK_ERROR(error) checkError(error, __FILE__, __LINE__)

bool FlyCapture::sync_capture_start(){
	if(numCameras==0) return false;
	FlyCapture2::Error error = FlyCapture2::Camera::StartSyncCapture( numCameras, (const FlyCapture2::Camera**)ppCameras );
	if(CHECK_ERROR(error)){
		is_started=true;
		return true;
	}
	return false;
}


bool FlyCapture::grab_flycap(){
	if(!is_opened) return false;
	assert(camImages.size() == numCameras);
	FlyCapture2::Error error;
	for(size_t i=0;i<numCameras;i++){
		error = ppCameras[i]->RetrieveBuffer(&camImages[i]);
		if(!CHECK_ERROR(error)) return false;
	}
	return true;
}

/*
 * @class grabで読み取っておいた画像を(カラーに変更するなどの指示に従って)現像する
 * */
bool FlyCapture::develop(cv::Mat& image, int device){
	FlyCapture2::PixelFormat pixFormat;
	FlyCapture2::BayerTileFormat bayerTileFormat;
	unsigned int rows,cols,stride;
	camImages[device].GetDimensions( &rows, &cols, &stride, &pixFormat, &bayerTileFormat);
	cv::Size size(cols,rows);
	if(size.height==0 || size.width==0) return false;

	if(get(skl::CONVERT_RGB)>=0){
		// create header to treat camImage[device] as cv::Mat
		cv::Mat bayer = cv::Mat(size,CV_8UC1,camImages[device].GetData(),stride);

		if(image.size()!=size || image.depth() != CV_8UC3 || !image.isContinuous()){
			image = cv::Mat(size,CV_8UC3);
		}
		assert(image.isContinuous());

		// get bayer pattern for cvtBayer2BGR
		int bayer_pattern;
		switch(bayerTileFormat){
			case FlyCapture2::RGGB:
				bayer_pattern = CV_BayerBG2BGR;
				break;
			case FlyCapture2::GRBG:
				bayer_pattern = CV_BayerGB2BGR;
				break;
			case FlyCapture2::GBRG:
				bayer_pattern = CV_BayerGR2BGR;
				break;
			case FlyCapture2::BGGR:
				bayer_pattern = CV_BayerRG2BGR;
				break;
			default:
				return false;
		}
		skl::cvtBayer2BGR(bayer,image,bayer_pattern,skl::BAYER_EDGE_SENSE);
	}
	else{
		if(image.size()!=size || image.depth() != CV_8UC1 || !image.isContinuous()){
			image = cv::Mat(size,CV_8UC1);
		}
		assert(image.isContinuous());
		assert(image.elemSize()*size.width*size.height == camImages[device].GetDataSize());
		memcpy(image.data,camImages[device].GetData(),camImages[device].GetDataSize());
	}

	return true;
}
