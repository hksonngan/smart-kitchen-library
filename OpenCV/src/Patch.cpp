/*!
 * @file Patch.cpp
 * @author a_hasimoto
 * @date Last Change:2012/Jan/05.
 */

#include "PatchModel.h"
#include "sklutils.h"
#include "sklcvutils.h"
#include <highgui.h>
#include <iostream>

using namespace skl;

Patch::Patch(){

}

Patch::~Patch(){

}

Patch::Patch(const cv::Mat& mask, const cv::Mat& img, const cv::Mat& current_bg, const cv::Mat& fg_edge, const cv::Rect& roi){
	set(mask,img,current_bg,fg_edge,roi);
}


void Patch::cvtG2L(int* x,int* y, Type type)const{
	*x -= _roi[type].x;
	*y -= _roi[type].y;
}

bool Patch::isIn(int local_x,int local_y, Type type)const{
	if( local_x < 0 || _roi[type].width <= local_x ) return false;
	if( local_y < 0 || _roi[type].height <= local_y ) return false;
	return true;
}

void Patch::setCoveredState(const cv::Rect& rect, const cv::Mat& mask, bool isCovered){
	cv::Rect common_region = rect & _roi[original];
	common_region.height += common_region.y;
	common_region.width += common_region.x;
	for(int y = common_region.y;y < common_region.height;y++){
		for(int x=common_region.x;x < common_region.width;x++){
			if(mask.at<float>(y,x)>0.0)
				setCoveredState(x,y,isCovered);
		}
	}

	//CHECK!
//	cvShowImage("visibility",this->visibility.getIplImage());
//	cvShowImage("patch",this->image[original].getIplImage());
//	cvWaitKey(-1);
}

void Patch::setCoveredState(int x,int y,bool isCovered){
	CvRect roi = _roi[original];
	assert(roi.x <= x && x < roi.x + roi.width);
	assert(roi.y <= y && y < roi.y + roi.height);
	cvtG2L(&x,&y,original);

	unsigned char val = 0;
	if(isCovered) val = 255;
	_covered_state.at<unsigned char>(y,x) = val;
}



void Patch::set(const cv::Mat& __mask, const cv::Mat& img, const cv::Mat& current_bg, const cv::Mat& fg_edge, const cv::Rect& roi){
	_roi[original] = roi;

	base_size.width = __mask.cols;
	base_size.height = __mask.rows;

	_roi[dilate] = cv::Rect(
		roi.x - PATCH_DILATE,
		roi.y - PATCH_DILATE,
		roi.x + roi.width + PATCH_DILATE,
		roi.y + roi.height + PATCH_DILATE);
	_roi[dilate].x = _roi[dilate].x > 0 ? _roi[dilate].x : 0;
	_roi[dilate].y = _roi[dilate].y > 0 ? _roi[dilate].y : 0;
	_roi[dilate].width = _roi[dilate].width < img.cols ? _roi[dilate].width - _roi[dilate].x : img.cols - _roi[dilate].x;
	_roi[dilate].height = _roi[dilate].height < img.rows ? _roi[dilate].height - _roi[dilate].y : img.rows - _roi[dilate].y;

	_image[dilate] = cv::Mat(img, _roi[dilate]).clone();
	_background[dilate] = cv::Mat(current_bg, _roi[dilate]).clone();
	_mask[dilate] = cv::Mat(_image[dilate].size(),CV_32FC1);
	cv::Mat(__mask, _roi[dilate]).convertTo(_mask[dilate],CV_32F,1.f/255.f);
/*
	std::cerr << "show dilate" << std::endl;
	cv::imshow("patch_image",_image[dilate]);
	cv::imshow("patch_bg",_background[dilate]);
	cv::imshow("patch_mask",_mask[dilate]);
	cv::waitKey(-1);
*/

	_mask[dilate] = blur_mask(_mask[dilate],PATCH_DILATE);


	_image[original] = cv::Mat(img,_roi[original]).clone();
	_background[original] = cv::Mat(current_bg,_roi[original]).clone();
	_mask[original] = cv::Mat(_image[original].size(),CV_32FC1);
	cv::Mat(__mask, _roi[original]).convertTo(_mask[original],CV_32F,1.f/255.f);
	_covered_state = cv::Mat::zeros(_mask[original].size(),CV_32FC1);

	for(int y = 0; y < _mask[original].rows; y += PATCH_MODEL_BLOCK_SIZE){
		for(int x = 0; x < _mask[original].cols; x += PATCH_MODEL_BLOCK_SIZE){
			if(0.0f == _mask[original].at<float>(y,x)) continue;
			_points.push_back(cv::Point(x,y));
		}
	}

	_edge = cv::Mat(fg_edge,roi).clone();
	_edge &= cv::Mat(__mask,_roi[original]);
/*
	std::cerr << "show dilate" << std::endl;
	cv::imshow("patch_image",_image[original]);
	cv::imshow("patch_bg",_background[original]);
	cv::imshow("patch_mask",_mask[original]);
	cv::imshow("patch_covered_state",_covered_state);
	cv::imshow("patch_edge",_edge);
	cv::waitKey(-1);
*/
}

void Patch::save(const std::string& filename,Type type,const std::string& edge_filename)const{
	CvRect roi = _roi[type];
	assert(roi.width <= base_size.width);
	assert(roi.height <= base_size.height);
	cv::Mat dest = cv::Mat(base_size,_image[type].type(),SKL_GRAY);
	cv::Mat temp = cv::Mat(dest,_roi[type]);
	skl::blending<cv::Vec3b,float>(temp,_image[type],_mask[type],temp);
	// CHECK!

	cv::imwrite(filename,dest);
	if(!edge_filename.empty()){
		dest = cv::Mat::zeros(dest.size(),_edge.type());
		temp = cv::Mat(dest,_roi[type]);
		// cv::Matに部分的に書きこむ方法を後で調べる

		cv::imwrite(edge_filename,dest);
	}
}

cv::Mat Patch::blur_mask(const cv::Mat& mask, size_t blur_width){
	cv::Mat dest = mask.clone();
	cv::Size kernel_size(blur_width,blur_width);

	cv::blur(dest, dest, kernel_size);
	if(mask.depth()==CV_8U){
		cv::threshold(dest, dest, 0,255,CV_THRESH_BINARY);
	}
	else if(mask.depth()==CV_32F){
		cv::threshold(dest, dest, 0.0f,1.0f,CV_THRESH_BINARY);
	}
	else{
		bool DepthIsNotSupported = true;
		assert(DepthIsNotSupported);
	}
	cv::blur(dest, dest, kernel_size);

	return dest;
}

/********** Accessor **********/
float Patch::maskValue(int x, int y, Type type)const{
	cvtG2L(&x,&y,type);
	if(!isIn(x,y,type)) return 0.0;
	return _mask[type].at<float>(y,x);
}


