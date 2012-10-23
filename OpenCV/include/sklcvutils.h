#ifndef __SKL_CV_UTILS_H__
#define __SKL_CV_UTILS_H__

#ifdef _WIN32
#pragma warning(disable:4996)
#endif
#include <cv.h>


// &(get the "and" region, cap operation)
// |(get the "or" region, cup operation)
cv::Rect operator&(const cv::Rect& left, const cv::Rect& right);
cv::Rect operator|(const cv::Rect& left, const cv::Rect& right);

// && returns true if two rectanbles are overlapped.
bool operator&&(const cv::Rect& left, const cv::Rect& right);

// Bayer to BGR image interpolation
// Used in class skl::VideoCaptureFlyCapture
#include "sklcvutils_bayer_conversion.h"

// template subfunctions for skl::blending
#include "sklcvutils_blending.h"



// use with a template function like below
// int FuncName<MatType>(...);
// sample code are available in 
#define cvMatTypeTemplateCall(MatType,FuncName,ReturnVal,DefaultType,...)\
	switch(MatType){\
		case CV_8U:\
			ReturnVal = FuncName<unsigned char>(__VA_ARGS__);\
			break;\
		case CV_8S:\
			ReturnVal = FuncName<char>(__VA_ARGS__);\
			break;\
		case CV_16U:\
			ReturnVal = FuncName<unsigned short>(__VA_ARGS__);\
			break;\
		case CV_16S:\
			ReturnVal = FuncName<short>(__VA_ARGS__);\
			break;\
		case CV_32S:\
			ReturnVal = FuncName<int>(__VA_ARGS__);\
			break;\
		case CV_32F:\
			ReturnVal = FuncName<float>(__VA_ARGS__);\
			break;\
		case CV_64F:\
			ReturnVal = FuncName<double>(__VA_ARGS__);\
			break;\
		default:\
			ReturnVal = FuncName<DefaultType>(__VA_ARGS__);\
	}

namespace skl{
	template <typename MatElem> int convTypeName2CVTypeID(){
		return -1;
	}
	template <> inline int convTypeName2CVTypeID<unsigned char>(){return CV_8U;}
	template <> inline int convTypeName2CVTypeID<char>(){return CV_8S;}
	template <> inline int convTypeName2CVTypeID<unsigned short>(){return CV_16U;}
	template <> inline int convTypeName2CVTypeID<short>(){return CV_16S;}
	template <> inline int convTypeName2CVTypeID<int>(){return CV_32S;}
	template <> inline int convTypeName2CVTypeID<float>(){return CV_32F;}
	template <> inline int convTypeName2CVTypeID<double>(){return CV_64F;}


	// check cv::Mat type/size and return debug message.	
	bool checkMat(const cv::Mat& mat, int depth = -1,int channels = 0,cv::Size size = cv::Size(0,0) );
	// check cv::Mat type/size and allocate the appropriate matrix if needed.
	bool ensureMat(cv::Mat& mat, int depth, int channels, cv::Size);

	cv::Rect fitRect(const std::vector< cv::Point >& points);

	// a simple ransac estimation.
	// the error for model evaluation is calcurated by sum of L2 norm between "estimated model parameter" and sample parameters.
	// see http://en.wikipedia.org/wiki/RANSAC for algorithm details.
	cv::Mat ransac(const cv::Mat& samples, cv::TermCriteria termcrit, double thresh_outliar, double sampling_rate = 0.2, double minimum_inliar_rate = 0.2);

	//! fill color palette acording to palette.size(). Different colors are set to each elem in palette.
	//! the difference are decided in HLS color space.
	void fillColorPalette(
			std::vector<cv::Scalar>& palette,
			size_t hue_pattern_num=7,
			bool use_gray=true,
			int min_luminance_value=20);

	cv::Vec3b assignColor(size_t ID);

	// for visualize the result returned by RegionLabelingAlgorith classes.
	template<typename MatElem> cv::Mat visualizeRegionLabel4MultiType(const cv::Mat& label,size_t region_num){
		cv::Mat vis = cv::Mat::zeros(label.size(),CV_8UC3);
		if(label.type()!=convTypeName2CVTypeID<MatElem>()){
			assert(label.type()==CV_16SC1);
		}
		if(region_num == 0){
			return vis;
		}
		std::vector<cv::Vec3b> colors(region_num);
		for(size_t i=0;i<region_num;i++){
			if(region_num<32){
				colors[i] = assignColor(i);
			}
			else{
				for(size_t c=0;c<3;c++){
					colors[i][c] = rand() % UCHAR_MAX;
				}
			}
			//		std::cerr << (int)colors[i][0] << ", " << (int)colors[i][1] << ", " << (int)colors[i][2] << std::endl;
		}

		for(int y=0;y<label.rows;y++){
			for(int x=0;x<label.cols;x++){
				MatElem l = label.at<MatElem>(y,x);
				if(l==0) continue;
				vis.at<cv::Vec3b>(y,x) = colors[l-1];
			}
		}
		return vis;
	}
	inline cv::Mat visualizeRegionLabel(const cv::Mat& label,size_t region_num){
		return visualizeRegionLabel4MultiType<short>(label,region_num);
	}

	enum ArrowType{
		NONE,
		ARROW,
		ARROW_FILL,
		INV_ARROW,
		INV_ARROW_FILL,
		CIRCLE,
		CIRCLE_FILL,
		SQUARE,
		SQUARE_FILL,
		ABS_SQUARE,
		ABS_SQUARE_FILL,
		DIAMOND,
		DIAMOND_FILL,
		ABS_DIAMOND,
		ABS_DIAMOND_FILL
	};

	// you can easily draw an arrow instead of cv::line!
	void arrow(
		cv::Mat& img,
		cv::Point pt1,
		cv::Point pt2,
		const cv::Scalar& color,
		int thickness = 1,
		int lineType = 8,
		ArrowType head_type = ARROW,
		ArrowType tail_type = NONE,
		int head_size = 6,
		int tail_size = 6,
		int head_degree = 30,
		int tail_degree = 30,
		int shift=0);


	template<class ElemType,class WeightType> void blending(const cv::Mat& src1,const cv::Mat& src2, const cv::Mat& weight_mask, cv::Mat& dest){
		assert(weight_mask.size()==src1.size());
		assert(weight_mask.size()==src2.size());
		assert(src1.type()==src2.type());
		assert(src1.size()==dest.size());
		assert(src1.type()==dest.type());
		cv::parallel_for(
				cv::BlockedRange(0,src1.rows*src1.cols),
				ParallelBlending<ElemType,WeightType>(src1,src2,weight_mask,dest)
				);
	}

	template<class ElemType,class WeightType> cv::Mat blending(const cv::Mat& src1,const cv::Mat& src2, const cv::Mat& weight_mask){
		cv::Mat dest = cv::Mat::zeros(src1.size(),src1.type());
		blending<ElemType,WeightType>(src1,src2,weight_mask,dest);
		return dest;
	}



	/********************************************************/
	/*                                                      */
	/*        "not very common" functions are below.        */
	/*                                                      */
	/********************************************************/


	// used in Patch.cpp
	cv::Mat blur_mask(const cv::Mat& mask, size_t blur_width);

	// used in Patch.cpp
	void edge_difference(const cv::Mat& src1, const cv::Mat& src2, cv::Mat& edge1, cv::Mat& edge2, double canny_thresh1=24, double canny_thresh2=24, int aperture_size=3, int dilate_size=3);

	// used in TableObjectManager.cpp
	template <class LabelType> void resize_label(const cv::Mat& label_small,size_t scale, cv::Mat& label){
		assert(label_small.channels()==1);
		assert(label_small.type()==label.type());
		cv::Size label_size(label_small.size());
		label_size.width *= (int)scale;
		label_size.height *= (int)scale;
		assert(label_size == label.size());

		label = cv::Scalar(0);
		if(label.isContinuous()){
			label.cols *= label.rows;
			label.rows = 1;
		}

		LabelType *plabel;
		const LabelType *plabel_small = 0;
		int y = -1, x = 0;
		int y_counter = 0;
		LabelType elem;
		for(int ly = 0; ly < label.rows; ly++){
			//			assert(0<=ly);
			//			assert(ly<label.rows);
			plabel = label.ptr<LabelType>(ly);

			for(int lx = 0; lx < label.cols; lx+=(int)scale,x++){
				if(lx % label_size.width == 0){
					if(y_counter%scale == 0){
						y++;
						//						assert(0<=y);
						//						assert(y < label_small.rows);
						plabel_small = label_small.ptr<const LabelType>(y);
					}
					y_counter++;
					x = 0;
				}
				//				assert(0 <= x);
				//				assert(x < label_small.cols);
				//				assert(0 <= lx);
				//				assert(lx < label.cols);
				if(0==(elem = plabel_small[x])) continue;
				for(size_t s=0;s<scale;s++){
					plabel[lx+s] = elem;
				}
			}
		}
		if(label.isContinuous()){
			label.cols = label_size.width;
			label.rows = label_size.height;
		}
	}

	// used in TableObjectManager.cpp
	template <class LabelType> cv::Mat resize_label(const cv::Mat& label_small,size_t scale){
		cv::Size label_size = label_small.size();
		label_size.width *= (int)scale;
		label_size.height *= (int)scale;
		cv::Mat label = cv::Mat(label_size,label_small.type());
		resize_label<LabelType>(label_small,scale,label);
		return label;
	}

}

#endif // __SKL_CV_UTILS_H__
