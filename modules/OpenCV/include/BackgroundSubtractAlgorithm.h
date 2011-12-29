#ifndef __BACKGROUND_SUBTRACT_ALGORITHM_H__
#define __BACKGROUND_SUBTRACT_ALGORITHM_H__
#include "FilterMat2Mat.h"
#include <cv.h>

namespace skl{
	class BackgroundSubtractAlgorithm:public FilterMat2Mat<int>{
		public:
			BackgroundSubtractAlgorithm(){};
			virtual ~BackgroundSubtractAlgorithm(){};
			virtual int compute(const cv::Mat& src, cv::Mat& dest){
				cv::Mat mask = cv::Mat(src.size(),CV_8UC1,255);
				return compute(src, mask, dest);
			}
			virtual int compute(const cv::Mat& src,const cv::Mat& mask, cv::Mat& dest)=0;
			virtual void updateBackgroundModel(const cv::Mat& bg,const cv::Mat& mask)=0;
	};

}

#endif // __BACKGROUND_SUBTRACT_ALGORITHM_H__
