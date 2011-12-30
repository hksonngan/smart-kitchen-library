/*!
 * @file Patch.cpp
 * @author 橋本敦史
 * @date Last Change:2011/Nov/01.
 */

#include "PatchModel.h"
#include "cvBlending.h"
#include <iostream>

using namespace mmpl;
using namespace mmpl::image;

Patch::Patch(){

}

Patch::~Patch(){

}

Patch::Patch(const Image& mask, const Image& newest_image, const Image& current_bg){
	set(mask,newest_image,current_bg);
}

/*Patch::Patch(const Patch& other){
	for(size_t i=0;i<2;i++){
		image[i] = other.image[i];
		mask[i] = other.mask[i];
		hidden[i] = other.hidden[i];
		rect[i] = other.rect[i];
	}
	visibility = other.visibility;
	local_features = other.local_features;
}*/

void Patch::convXY_G2L(int* x,int* y, Type type)const{
	*x -= rect[type].x;
	*y -= rect[type].y;
}

bool Patch::isIn(int local_x,int local_y, Type type)const{
	if( local_x < 0 || rect[type].width <= local_x ) return false;
	if( local_y < 0 || rect[type].height <= local_y ) return false;
	return true;
}

void Patch::set(const Image& mask, const Image& newest_image,const Image& current_bg){

	setDilate(getRect(mask),mask,newest_image,current_bg);

	// get edge
	CvRect edge_bounding_rect = extractEdges(
			&this->edge,
			this->mask[dilate],
			this->image[dilate],
			this->hidden[dilate],
			&this->edge_pixel_num);
//	std::cerr << edge_bounding_rect.x << ", " << edge_bounding_rect.y << ", " << edge_bounding_rect.width << ", " << edge_bounding_rect.height << std::endl;

	if(edge_bounding_rect.width<=1 || edge_bounding_rect.height<=1){
		// failed to extract edge -> not an object.
		this->edge.setWidth(1);
		this->edge.setHeight(1);
	}
	else{
		rect[original] = edge_bounding_rect;
		setDilate(edge_bounding_rect,mask,newest_image,current_bg);
	}

	crop(&(this->image[original]),newest_image,rect[original]);
	crop(&(this->hidden[original]),current_bg,rect[original]);

	visibility = this->mask[original];
	setPoints();
}

void Patch::crop(Image* dist,const Image& _src,const CvRect& rect,const IplImage* mask){
	Image src(_src);
	src.setROI(rect);
	dist->setWidth(rect.width);
	dist->setHeight(rect.height);
	dist->setChannels(src.getChannels());
	dist->setDepth(src.getDepth());
	cvCopy(src.getIplImage(),dist->getIplImage(),mask);
}

CvRect Patch::getRect(const Image& mask,size_t* pix_num){
	CvRect rect;
	rect.x = INT_MAX;
	rect.y = INT_MAX;
	rect.width = 0;
	rect.height = 0;

	size_t count = 0;

	for(int y=0;y<mask.getHeight();y++){
		unsigned char* pmask = (unsigned char*)(mask.getIplImage()->imageData
				+ mask.getIplImage()->widthStep * y);
		for(int x=0;x<mask.getWidth();x++){
			if(pmask[x]!=0){
				count++;
				rect.x = rect.x < x ? rect.x : x;
				rect.y = rect.y < y ? rect.y : y;
				rect.width = x < rect.width ? rect.width : x;
				rect.height = y < rect.height ? rect.height : y;
			}
		}
	}
	rect.width -= rect.x;
	rect.height -= rect.y;
	if(pix_num!=NULL){
		*pix_num = count;
	}
	return rect;
}


void Patch::setVisibility(const CvRect& rect, const Image& mask, bool isVisible){
	CvRect common_region = getCommonRectanglarRegion(
			rect,
			this->getRect(original)
			);
	common_region.height += common_region.y;
	common_region.width += common_region.x;
	for(int y = common_region.y;y < common_region.height;y++){
		for(int x=common_region.x;x < common_region.width;x++){
			if(((float*)(mask.getIplImage()->imageData + 
							mask.getIplImage()->widthStep * (y - rect.y)
							))[x - rect.x]>0.0f){
				this->setVisibility(x,y,isVisible);
			}
		}
	}

//	cvShowImage("visibility",this->visibility.getIplImage());
//	cvShowImage("patch",this->image[original].getIplImage());
//	cvWaitKey(-1);

}


void Patch::save(const std::string& filename,Type type,int width,int height,const std::string& edge_filename)const{
	CvRect rect = getRect(type);
	assert(rect.width <= width);
	assert(rect.height <= height);

	Image temp(width,height,image[type].getDepth(),image[type].getChannels());
	cvSet(temp.getIplImage(),CV_RGB(128,128,128));
	temp.setROI(rect);

	cvBlending(
			image[type].getIplImage(),
			temp.getIplImage(),
			mask[type].getIplImage(),
			temp.getIplImage());
	temp.removeROI();
	temp.saveImage(filename);
	if(!edge_filename.empty()){
		temp.setChannels(1);
		temp.setDepth(edge.getDepth());

//		std::cerr << edge.getWidth() << ", " << edge.getHeight() << std::endl;
//		std::cerr << this->rect[original].width << ", " << this->rect[original].height << std::endl;

		cvZero(temp.getIplImage());
		temp.setROI(this->rect[original]);
		cvCopy(edge.getIplImage(),temp.getIplImage());
		temp.removeROI();
		temp.saveImage(edge_filename);
	}
}

void Patch::setPoints(){
	points.clear();
	CvPoint base;
	base.x = rect[original].x;
	base.y = rect[original].y;
	for(int y=0;y<mask[original].getHeight();y+=BLOCK_SIZE){
		float* pmask = (float*)(mask[original].getIplImage()->imageData + mask[original].getIplImage()->widthStep * y);
		for(int x=0;x<mask[original].getWidth();x+=BLOCK_SIZE){
			if(pmask[x]!=0){
				points.push_back(cvPoint(x + base.x,y + base.y));
			}
		}
	}
}

/********** Accessor **********/
const CvRect& Patch::getRect(Type type)const{
	return rect[type];
}

float Patch::maskValue(int x, int y, Type type)const{
	convXY_G2L(&x,&y,type);
	if(!isIn(x,y,type)) return 0.0;

	return ((float*)(mask[type].getIplImage()->imageData
			+ mask[type].getIplImage()->widthStep * y) )[x];
}

Image& Patch::getImage(Type type){
	return image[type];
}

const Image& Patch::getImage(Type type)const{
	return image[type];
}

Image& Patch::getBG(Type type){
	return hidden[type];
}

const Image& Patch::getBG(Type type)const{
	return hidden[type];
}
const Image& Patch::getMask(Type type)const{
	return mask[type];
}

const Image& Patch::getEdgeImage()const{
	return edge;
}

void Patch::setEdgeImage(const Image& _edge){
	assert(edge.getWidth()==_edge.getWidth());
	assert(edge.getHeight()==_edge.getHeight());
	assert(edge.getDepth()==_edge.getDepth());
	assert(edge.getChannels()==_edge.getChannels());
	edge = _edge;
}

const std::vector<CvPoint>& Patch::getPoints()const{
	return points;
}

CvRect Patch::getCommonRectanglarRegion(const CvRect& r1,const CvRect& r2){
	CvRect dst = r1;
	dst.width += dst.x;
	dst.height += dst.y;

	dst.x = dst.x > r2.x ? dst.x:r2.x;
	dst.y = dst.y > r2.y ? dst.y:r2.y;
	dst.width = dst.width < r2.x+r2.width ? dst.width:r2.x+r2.width;
	dst.height = dst.height < r2.y+r2.height ? dst.height:r2.y+r2.height;
	dst.width -= dst.x;
	dst.height -= dst.y;
	return dst;
}

void Patch::setVisibility(int x,int y,bool isVisible){
	CvRect rect = this->getRect(original);
	assert(rect.x <= x && x < rect.x + rect.width);
	assert(rect.y <= y && y < rect.y + rect.height);
	convXY_G2L(&x,&y,original);

	unsigned char val = 0.0f;
	if(isVisible) val = 1.0f;
	((float*)(visibility.getIplImage()->imageData
			+ visibility.getIplImage()->widthStep * y) )[x] = val;
}

CvRect Patch::extractEdges(Image* edge,const Image& mask,const Image& _src,const Image& _bg,size_t* edge_pix_num)const{
	assert(1==mask.getChannels());
	assert(IPL_DEPTH_32F==mask.getDepth());
	assert(mask.getWidth()==_src.getWidth());
	assert(mask.getHeight()==_src.getHeight());
	assert(mask.getWidth()==_bg.getWidth());
	assert(mask.getHeight()==_bg.getHeight());
	assert(edge!=NULL);

	Image _edge(mask.getWidth(),mask.getHeight(),IPL_DEPTH_8U,1);

	// blending diff, bg with black color bg by dilated mask
	Image diff = _src;
	cvAbsDiff(_src.getIplImage(),_bg.getIplImage(),diff.getIplImage());
	Image black(_src);
	cvZero(black.getIplImage());
	cvBlending(diff.getIplImage(),black.getIplImage(),mask.getIplImage(),diff.getIplImage());
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

	// shift eb_rect from rect[dilate]-relative to
	// to image-absolute coordinate.
	CvRect abs_eb_rect = eb_rect;
	abs_eb_rect.x += rect[dilate].x;
	abs_eb_rect.y += rect[dilate].y;

	// edge bounding box must be smaller then rect[original].
	abs_eb_rect.width += abs_eb_rect.x;
	abs_eb_rect.height += abs_eb_rect.y;

	if(abs_eb_rect.x < rect[original].x){
		eb_rect.x += rect[original].x - abs_eb_rect.x;
		abs_eb_rect.x = rect[original].x;
	}
	if(abs_eb_rect.y < rect[original].y){
		eb_rect.y += rect[original].y - abs_eb_rect.y;
		abs_eb_rect.y = rect[original].y;
	}

	if(abs_eb_rect.width > rect[original].x + rect[original].width){
		abs_eb_rect.width = rect[original].x + rect[original].width;
	}
	if(abs_eb_rect.height > rect[original].y + rect[original].height){
		abs_eb_rect.height = rect[original].y + rect[original].height;
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
	ro_rect.x -= rect[original].x;
	ro_rect.y -= rect[original].y;
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

void Patch::setDilate(const CvRect& _rect, const Image& mask, const Image& src, const Image& bg){
	rect[original] = _rect;


	Image temp;
	crop(&temp,mask,rect[original]);
	this->mask[original] = Image(rect[original].width,rect[original].height,IPL_DEPTH_32F,1);
	cvConvertScale(temp.getIplImage(),this->mask[original].getIplImage(),1.0/255,0);

	// dilate画像をぼかす前に計算の効率化のために
	// dilate後の画像領域を先にクリップする
	rect[dilate].x = rect[original].x - PATCH_DILATE;
	rect[dilate].x = 0 > rect[dilate].x ? 0 : rect[dilate].x;
	rect[dilate].width = rect[original].x + rect[original].width + PATCH_DILATE;
	rect[dilate].width = rect[dilate].width > mask.getWidth() ? mask.getWidth() : rect[dilate].width;
	rect[dilate].width -= rect[dilate].x;

	rect[dilate].y = rect[original].y - PATCH_DILATE;
	rect[dilate].y = 0 > rect[dilate].y ? 0 : rect[dilate].y;

	rect[dilate].height = rect[original].y + rect[original].height + PATCH_DILATE;
	rect[dilate].height = rect[dilate].height > mask.getHeight() ? mask.getHeight() : rect[dilate].height;
	rect[dilate].height -= rect[dilate].y;
	
//	crop(&temp,mask,rect[dilate]);
	CvRect do_roi = rect[original];
	do_roi.x -= rect[dilate].x;
	do_roi.y -= rect[dilate].y;
	this->mask[dilate] = Image(rect[dilate].width,rect[dilate].height,IPL_DEPTH_32F,1);
	cvZero(this->mask[dilate].getIplImage());

	this->mask[dilate].setROI(do_roi);
	cvCopy(this->mask[original].getIplImage(),this->mask[dilate].getIplImage());
	this->mask[dilate].removeROI();
//	cvShowImage("dilated mask",this->mask[dilate].getIplImage());
//	cvWaitKey(-1);

	// mask[dilate]の境界をぼかす
	cvSmooth(this->mask[dilate].getIplImage(),this->mask[dilate].getIplImage(),CV_BLUR,PATCH_DILATE,PATCH_DILATE);
	cvThreshold(this->mask[dilate].getIplImage(),this->mask[dilate].getIplImage(),0.0f,1.0f,CV_THRESH_BINARY);
	cvSmooth(this->mask[dilate].getIplImage(),this->mask[dilate].getIplImage(),CV_BLUR,PATCH_DILATE,PATCH_DILATE);

	// 元画像を入れておく
	crop(&(this->image[dilate]),src,rect[dilate]);

	// 自分の裏側の背景も記憶しておく
	crop(&(this->hidden[dilate]),bg,rect[dilate]);
}

size_t Patch::getEdgePixelNum()const{
	return edge_pixel_num;
}
