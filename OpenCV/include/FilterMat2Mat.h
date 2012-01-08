#ifndef __FILTER_MAT2MAT_H__
#define __FILTER_MAT2MAT_H__
#include <cv.h>

namespace skl{
	template<class T=void> class FilterMat2Mat{
		public:
			FilterMat2Mat();
			virtual ~FilterMat2Mat();
			virtual T compute(
					const cv::Mat& src,
					cv::Mat& dest);

			virtual T compute(
					const cv::Mat& src,
					const cv::Mat& mask,
					cv::Mat& dest)=0;
	};

	template<class T> FilterMat2Mat<T>::FilterMat2Mat(){}
	template<class T> FilterMat2Mat<T>::~FilterMat2Mat(){}
	template<class T> T FilterMat2Mat<T>::compute(
			const cv::Mat& src,
			cv::Mat& labels){
		cv::Mat mask = cv::Mat(src.size(),src.depth(),1);
		return compute(src,src,labels);
	}
}
#endif // __FILTER_MAT2MAT_H__
