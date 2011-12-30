/*!
 * @file Patch.cpp
 * @author a_hasimoto
 * @date Last Change:2011/Nov/01.
 */

#include "PatchModel.h"
#include "sklcvutils.h"
#include <iostream>

using namespace mmpl;
using namespace mmpl::image;

Patch::Patch(){

}

Patch::~Patch(){

}

Patch::Patch(const cv::Mat& __mask, const cv::Mat& __newest_image, const cv::Mat& __current_bg, const cv::Rect& roi){
	set(__mask,__newest_image,__current_bg,roi);
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
	unsigned char val = 0;
	if(isCovered) val = 255;
	for(int y = common_region.y;y < common_region.height;y++){
		for(int x=common_region.x;x < common_region.width;x++){
			if(mask.at<float>(y,x)>0.0)
				setCoveredState(x,y,val);
		}
	}

	//CHECK!
//	cvShowImage("visibility",this->visibility.getIplImage());
//	cvShowImage("patch",this->image[original].getIplImage());
//	cvWaitKey(-1);

}


bool Patch::set(const cv::Mat& __mask, const cv::Mat& __newest_image,const cv::Mat& __current_bg, cv::Rect roi){
	base_width = __mask.cols;
	base_height = __mask.rows;

	_roi[dilate] = cv::Rect(
		roi.x - PATCH_DILATE,
		roi.y - PATCH_DILATE,
		roi.x + roi.width + PATCH_DILATE,
		roi.y + roi.height + PATCH_DILATE);
	_roi[dilate].x = _roi[dilate].x > 0 ? _roi[dilate].x : 0;
	_roi[dilate].y = _roi[dilate].y > 0 ? _roi[dilate].y : 0;
	_roi[dilate].width = _roi[dilate].width < __newest_image.cols ? _roi[dilate].width - _roi[dilate].x : __newest_image.cols - _roi[dilate].x;
	_roi[dilate].height = _roi[dilate].height < __newest_image.rows ? _roi[dilate].height - _roi[dilate].y : __newest_image.rows - _roi[dilate].y;

	_image[dilate] = cv::Mat(__newest_image, _roi[dilate]).clone();
	_hidden[dilate] = cv::Mat(__current_bg, _roi[dilate]).clone();
	_mask[dilate] = cv::Mat(__mask, _roi[dilate]).clone();
	std::vector<cv::Point> edge_points;

	_edge = get_edge(_image[dilate],_mask[dilate],&edge_points);

	edge_counts = edge_points.size()  // è„éËÇ¢Ç‚ÇËï˚Ç™Ç†ÇËÇªÇ§
	if(_edge_count == 0) return;

	roi = fitRect(edge_points);
	_roi[original] = roi;
	_image[original] = cv::Mat(__newest_image,roi).clone();
	_hidden[original] = cv::Mat(__current_bg,roi).clone();
	_mask[original] = cv::Mat(__mask,roi).clone();	
	_covered_state = _mask[original].clone();

	cv::Rect roi_dilate = _roi[dilate];
	_roi[dilate] = cv::Rect(
	roi.x - PATCH_DILATE,
	roi.y - PATCH_DILATE,
	roi.x + roi.width + PATCH_DILATE,
	roi.y + roi.height + PATCH_DILATE);
	_roi[dilate].x = _roi[dilate].x > 0 ? _roi[dilate].x : 0;
	_roi[dilate].y = _roi[dilate].y > 0 ? _roi[dilate].y : 0;
	_roi[dilate].width = _roi[dilate].width < __newest_image.cols ? _roi[dilate].width - _roi[dilate].x : __newest_image.cols - _roi[dilate].x;
	_roi[dilate].height = _roi[dilate].height < __newest_image.rows ? _roi[dilate].height - _roi[dilate].y : __newest_image.rows - _roi[dilate].y;

	if(roi_dilate != _roi[dilate]){
		_image[dilate] = cv::Mat(__newest_image,_roi[dilate]).clone();
		_hidden[dilate] = cv::Mat(__current_bg,_roi[dilate]).clone();
		_mask[dilate] = cv::Mat(__mask,_roi[dilate]).clone();	
	}
	CvMat ___mask = _mask[dilate];
	cv::Mat _temp = _mask[dilate].clone();
	CvMat ___temp = _temp;

	cvSmooth(&___mask,&___mask,CV_BLUR,PATCH_DILATE,PATCH_DILATE);
	cvThreshold(&__mask,&___mask,0.0f,1.0f,CV_THRESH_BINARY);
	cvSmooth(&___mask,&___mask,CV_BLUR,PATCH_DILATE,PATCH_DILATE);
}

void Patch::save(const std::string& filename,Type type,const std::string& edge_filename)const{
	CvRect roi = roi(type);
	assert(rect.width <= base_width);
	assert(rect.height <= base_height);

	cv::Mat dest = cv::Mat(base_width,base_height,image[type].depth(),image[type].height(),skl::GRAY);
	cv::Mat temp = cv::Mat(dest,_roi[type]);
	skl::blending<cv::Vec3b,float>(temp,_image[type],_mask[type],temp);
	// CHECK!

	cv::imwrite(filename,dest);
	if(!edge_filename.empty()){
		dest = cv::Mat::zeros(dest.size(),_edge.type());
		temp = cv::Mat(dest,_roi[type]);
		// cv::MatÇ…ïîï™ìIÇ…èëÇ´Ç±Çﬁï˚ñ@Çå„Ç≈í≤Ç◊ÇÈ

		cv::imwrite(edge_filename,dest);
	}
}

/********** Accessor **********/
float Patch::maskValue(int x, int y, Type type)const{
	cvtG2L(&x,&y,type);
	if(!isIn(x,y,type)) return 0.0;
	return mask[type].at<float>(y,x);
}


void Patch::covered_state(int x,int y,bool isCovered){
	CvRect roi = _roi[original];
	assert(roi.x <= x && x < roi.x + rect.width);
	assert(roi.y <= y && y < roi.y + rect.height);
	cvtG2L(&x,&y,original);

	unsigned char val = 0;
	if(isCovered) val = 255;
	_covered_state.at<unsigned char>(y,x) = val;
}



CvRect Patch::extractEdges(cv::Mat& edge,const cv::Mat& mask,const cv::Mat& _src,const cv::Mat& _bg, size_t* edge_pix_num)const{
	assert(CV_32FC1 = mask.type());
	assert(mask.cols ==_src.cols);
	assert(mask.rows ==_src.rows);
	assert(mask.cols ==_bg.cols);
	assert(mask.rows ==_bg.rows);
	assert(edge!=NULL);

	cv::Mat _edge(mask.size(),CV_8UC1,0);

	// blending diff, bg with black color bg by dilated mask
	cv::Mat diff = _src.clone();
	cvAbsDiff(_src.getIplImage(),_bg.getIplImage(),diff.getIplImage());
	// CHECK How to calc cvAbsDiff in OpenCV2.0

	cv::Mat black = cv::Mat::zeros(_src.size(),CV_8UC3);
	skl::blending(diff.getIplImage(),black.getIplImage(),mask.getIplImage(),diff.getIplImage());
	cvSmooth(diff.getIplImage(),diff.getIplImage(),CV_GAUSSIAN,0,0,1);

	Image bg(_bg);
//	cvBlending(bg.getIplImage(),black.getIplImage(),mask.getIplImage(),bg.getIplImage());
	bg.setChannels(1);
	cvSmooth(bg.getIplImage(),bg.getIplImage(),CV_GAUSSIAN,0,0,1);
	Image bg_edge(_edge);

	// extract edge by canny filter
	diff.setChannels(1);
	double thresh1 = PATCH_EDGE_CANNY_THRESH1;
	double thresh2 = PATCH_EDGE_CANNY_THRESH2;
	cvCanny(diff.getIplImage(),_edge.getIplImage(),thresh1,thresh2);
	cvCanny(bg.getIplImage(),bg_edge.getIplImage(),thresh1,thresh2);
	cvSmooth(bg_edge.getIplImage(),bg_edge.getIplImage(),CV_BLUR,4,4);
	cvThreshold(bg_edge.getIplImage(),bg_edge.getIplImage(),0,255,CV_THRESH_BINARY);
	cvSub(_edge.getIplImage(),bg_edge.getIplImage(),_edge.getIplImage());
//	cvShowImage("edge",_edge.getIplImage());
//	cvWaitKey(-1);

	// check if it has edge
	CvRect eb_rect = getRect(_edge,edge_pix_num);

	// exit when no edges are detect.
	if(*edge_pix_num < 10){
//	if(eb_rect.width <= 1 || eb_rect.height <= 1){
		edge->setWidth(1);
		edge->setHeight(1);
		eb_rect.width = -1;
		eb_rect.height = -1;
		return eb_rect;
	}


//	std::cerr << "orig_rect: " << eb_rect.x << "," << eb_rect.y << "," << eb_rect.width << ", " << eb_rect.height << std::endl;

	// shift eb_rect from _roi[dilate]-relative to
	// to image-absolute coordinate.
	CvRect abs_eb_rect = eb_rect;
	abs_eb_rect.x += _roi[dilate].x;
	abs_eb_rect.y += _roi[dilate].y;

	// edge bounding box must be smaller then _roi[original].
	abs_eb_rect.width += abs_eb_rect.x;
	abs_eb_rect.height += abs_eb_rect.y;


	if(abs_eb_rect.x < _roi[original].x){
		eb_rect.x += _roi[original].x - abs_eb_rect.x;
		abs_eb_rect.x = _roi[original].x;
	}
	if(abs_eb_rect.y < _roi[original].y){
		eb_rect.y += _roi[original].y - abs_eb_rect.y;
		abs_eb_rect.y = _roi[original].y;
	}

	if(abs_eb_rect.width > _roi[original].x + _roi[original].width){
		abs_eb_rect.width = _roi[original].x + _roi[original].width;
	}
	if(abs_eb_rect.height > _roi[original].y + _roi[original].height){
		abs_eb_rect.height = _roi[original].y + _roi[original].height;
	}

	abs_eb_rect.width -= abs_eb_rect.x;
	abs_eb_rect.height -= abs_eb_rect.y;

	eb_rect.width = abs_eb_rect.width;
	eb_rect.height = abs_eb_rect.height;

	if(eb_rect.width <= 1 || eb_rect.height <= 1){
		edge->setWidth(1);
		edge->setHeight(1);
		return abs_eb_rect;
	}


/*	std::cerr << "_edge: " << _edge.getWidth() << "," << _edge.getHeight() << ", " << _edge.getDepth() << ", " << _edge.getChannels() << std::endl;
	std::cerr << "abs_rect: " << abs_eb_rect.x << "," << abs_eb_rect.y << "," << abs_eb_rect.width << ", " << abs_eb_rect.height << std::endl;
	std::cerr << "rect: " << eb_rect.x << "," << eb_rect.y << "," << eb_rect.width << ", " << eb_rect.height << std::endl;
*/
	Image temp;
	crop(&temp,_edge,eb_rect);
	*edge = temp;
	cvZero(edge->getIplImage());

	Image mask_8U = this->mask[original];
	mask_8U.setDepth(IPL_DEPTH_8U);
	cvConvertScale(this->mask[original].getIplImage(),mask_8U.getIplImage(),255.0,0);

//	std::cerr << mask_8U.getDepth() << ", " << mask_8U.getChannels() << ", " << mask_8U.getWidth() << ", " << mask_8U.getHeight() << std::endl;
	CvRect ro_rect = abs_eb_rect;
	ro_rect.x -= _roi[original].x;
	ro_rect.y -= _roi[original].y;
//	std::cerr << ro_rect.x << ", " << ro_rect.y << ", " << ro_rect.width << ", " << ro_rect.height << std::endl;
	crop(&mask_8U,mask_8U,ro_rect);
	cvCopy(temp.getIplImage(),edge->getIplImage(),mask_8U.getIplImage());

	size_t edge_num = 0;
	for(int y=0; y < edge->getHeight(); y++){
		unsigned char* pedge = (unsigned char*)(edge->getIplImage()->imageData + y * edge->getIplImage()->widthStep);
		for(int x=0; x < edge->getWidth(); x++){
			if(pedge[x]>0){
				edge_num++;
			}
		}
	}

	if(edge_pix_num != NULL){
		*edge_pix_num = edge_num;
		if(edge_num < 10){
			abs_eb_rect.width = 0;
			abs_eb_rect.height = 0;
			return abs_eb_rect;
		}
	}

	return abs_eb_rect;
}

