/*!
 * @file Patch.cpp
 * @author a_hasimoto
 * @date Last Change:2011/Nov/01.
 */

#include "PatchModel.h"
#include "sklcvutils.h"
#include <highgui.h>
#include <iostream>

using namespace skl;

Patch::Patch(){

}

Patch::~Patch(){

}

Patch::Patch(const cv::Mat& __mask, const cv::Mat& __newest_image, const cv::Mat& __current_bg, const std::vector<cv::Point>& points){
	set(__mask,__newest_image,__current_bg,points);
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



void Patch::set(const cv::Mat& __mask, const cv::Mat& __newest_image,const cv::Mat& __current_bg,const std::vector<cv::Point>& points){

	cv::Rect roi = fitRect(points);

	base_size.width = __mask.cols;
	base_size.height = __mask.rows;

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
	_background[dilate] = cv::Mat(__current_bg, _roi[dilate]).clone();
	_mask[dilate] = cv::Mat(__mask, _roi[dilate]).clone();

	CvMat ___mask = _mask[dilate];
	cvSmooth(&___mask,&___mask,CV_BLUR,PATCH_DILATE,PATCH_DILATE);
	cvThreshold(&__mask,&___mask,0.0f,1.0f,CV_THRESH_BINARY);
	cvSmooth(&___mask,&___mask,CV_BLUR,PATCH_DILATE,PATCH_DILATE);

	std::vector<cv::Point> edge_points;
	_roi[original] = extractEdges(_edge,&edge_points);

	_edge_count = edge_points.size();
	if(_edge_count == 0) return;

	_image[original] = cv::Mat(__newest_image,_roi[original]).clone();
	_background[original] = cv::Mat(__current_bg,_roi[original]).clone();
	_mask[original] = cv::Mat(__mask,_roi[original]).clone();	
	_covered_state = _mask[original].clone();

	if(roi == _roi[original]){
		_points = points;
		for(size_t i=0;i<_points.size();i++){
			_points[i].x -= roi.x;
			_points[i].y -= roi.y;
		}
		return;
	}
	roi = _roi[original];
	_points.clear();
	for(size_t i=0;i<points.size();i++){
		cv::Point pt(points[i].x - roi.x, points[i].y - roi.y);
		if(pt.x < 0 || roi.width <= pt.x) continue;
		if(pt.y < 0 || roi.height <= pt.y) continue;		
		_points.push_back(pt);
	}

	// set dilate params again with the roi fitted to edge.
	_roi[dilate] = cv::Rect(
		roi.x - PATCH_DILATE,
		roi.y - PATCH_DILATE,
		roi.x + roi.width + PATCH_DILATE,
		roi.y + roi.height + PATCH_DILATE);
	_roi[dilate].x = _roi[dilate].x > 0 ? _roi[dilate].x : 0;
	_roi[dilate].y = _roi[dilate].y > 0 ? _roi[dilate].y : 0;
	_roi[dilate].width = _roi[dilate].width < __newest_image.cols ? _roi[dilate].width - _roi[dilate].x : __newest_image.cols - _roi[dilate].x;
	_roi[dilate].height = _roi[dilate].height < __newest_image.rows ? _roi[dilate].height - _roi[dilate].y : __newest_image.rows - _roi[dilate].y;

	_image[dilate] = cv::Mat(__newest_image,_roi[dilate]).clone();
	_background[dilate] = cv::Mat(__current_bg,_roi[dilate]).clone();
	_mask[dilate] = cv::Mat(__mask,_roi[dilate]).clone();	

	cv::Mat temp = _mask[dilate].clone();
	cvSmooth(&temp,&_mask[dilate],CV_BLUR,PATCH_DILATE,PATCH_DILATE);
	cvThreshold(&_mask[dilate],&temp,0.0f,1.0f,CV_THRESH_BINARY);
	cvSmooth(&temp,&_mask[dilate],CV_BLUR,PATCH_DILATE,PATCH_DILATE);
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
		// cv::MatÇ…ïîï™ìIÇ…èëÇ´Ç±Çﬁï˚ñ@Çå„Ç≈í≤Ç◊ÇÈ

		cv::imwrite(edge_filename,dest);
	}
}

/********** Accessor **********/
float Patch::maskValue(int x, int y, Type type)const{
	cvtG2L(&x,&y,type);
	if(!isIn(x,y,type)) return 0.0;
	return _mask[type].at<float>(y,x);
}



cv::Rect Patch::extractEdges(cv::Mat& edge, std::vector<cv::Point>* edge_points)const{
	edge = cv::Mat(_mask[dilate].size(),CV_8UC1,0);

	// blending diff, bg with black color bg by dilated mask
	cv::Mat diff = cv::Mat::zeros(_image[dilate].size(),CV_8UC3);
	
	cvAbsDiff(&_image[dilate],&_background[dilate],&diff);
	cv::Mat diff_gray = cv::Mat::zeros(diff.size(),CV_8UC1);
	cv::cvtColor(diff,diff_gray,CV_BGR2GRAY);

	diff *= _mask[dilate];

	// extract edge by canny filter
	double thresh1 = PATCH_EDGE_CANNY_THRESH1;
	double thresh2 = PATCH_EDGE_CANNY_THRESH2;
	cv::Canny(diff,edge,thresh1,thresh2);


	// extract background_gray
	cv::Mat bg_edge = cv::Mat(_background[dilate].size(),CV_8UC1);
	cv::Mat bg_gray = cv::Mat(_background[dilate].size(),CV_8UC1);
	cv::cvtColor(_background[dilate],bg_gray,CV_BGR2GRAY);

	cv::Canny(bg_gray,bg_edge,thresh1,thresh2);
	cv::Mat temp = bg_gray; // reuse of bg_gray as temp
	cvSmooth(&CvMat(bg_edge),&CvMat(temp),CV_BLUR,4,4);
	cvThreshold(&CvMat(temp),&CvMat(bg_edge),0,255,CV_THRESH_BINARY);
	cvSub(&CvMat(edge),&CvMat(bg_edge),&CvMat(edge));

	cv::imshow("edge",edge);
	cv::waitKey(-1);

	// check if it has edge
	cv::Rect roi_original(_roi[original].x,_roi[original].y,_roi[original].x + _roi[original].width, _roi[original].y + _roi[original].height);
	for(int y = 0; y < _edge.rows; y++){
		for(int x = 0; x < _edge.cols; x++){
			int abs_x = x + _roi[dilate].x;
			int abs_y = y + _roi[dilate].y;
			if(abs_x < roi_original.x || roi_original.width <= abs_x ) continue;
			if(abs_y < roi_original.y || roi_original.height <= abs_x) continue;
			if(0 == _edge.at<unsigned char>(y,x)) continue;
			if(0.0 == _mask[original].at<float>(abs_y - roi_original.y, abs_x - roi_original.x)) continue;
			edge_points->push_back(cv::Point(x,y));
		}
	}

	if(0 == edge_points->size()) return cv::Rect(0,0,0,0);
	cv::Rect eb_rect = fitRect(*edge_points);
	edge = cv::Mat(edge,eb_rect).clone();

	eb_rect.x += _roi[dilate].x;
	eb_rect.y += _roi[dilate].y;
	return eb_rect;
}

