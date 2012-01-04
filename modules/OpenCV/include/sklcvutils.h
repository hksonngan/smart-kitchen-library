#ifndef __SKL_CV_UTILS_H__
#define __SKL_CV_UTILS_H__

#ifdef _WIN32
#pragma warning(disable:4996)
#endif
#include <cv.h>

/*
 * return common region of left and right;
 */
cv::Rect operator&(const cv::Rect& left, const cv::Rect& right);

/*!
 * return true when left and right intersects.
 */
bool operator&&(const cv::Rect& left, const cv::Rect& right);

/*!
 * return minimum rectangle which include both left and right
 */
cv::Rect operator|(const cv::Rect& left, const cv::Rect& right);

namespace skl{
#define SKL_GRAY 128
	typedef enum{
		BAYER_SIMPLE,	//!< 単純なベイヤー変換
		BAYER_NN,		//!< NNを考慮したベイヤー
		BAYER_EDGE_SENSE,//,//!< エッジを考慮したベイヤー
	} ColorInterpolationType;
	void cvtBayer2BGR(const cv::Mat& bayer, cv::Mat& bgr, int code=CV_BayerBG2BGR, int	algo_type=BAYER_SIMPLE);


	cv::Rect fitRect(const std::vector< cv::Point >& points);

	cv::Vec3b convHLS2BGR(const cv::Vec3b& hls);
	cv::Vec3b assignColor(size_t ID);
	cv::Mat visualizeRegionLabel(const cv::Mat& label,size_t region_num);




	template<class T> void setWeight(const T& mask,double* w1, double* w2){
		*w1 = mask;
		*w2 = 1.0 - mask;
	}
	template<> void setWeight<unsigned char>(const unsigned char& mask, double* w1, double* w2);

	template<class T> T blend(const T& pix1, const T& pix2, double w1,double w2){
		return static_cast<T>(w1 * pix1 + w2 * pix2);
	}
	template<> cv::Vec3b blend(const cv::Vec3b& pix1,const cv::Vec3b& pix2, double w1, double w2);


	template <class ElemType,class WeightType> class ParallelBlending{
		public:
			ParallelBlending(
					const cv::Mat& src1,
					const cv::Mat& src2,
					const cv::Mat& mask,
					cv::Mat& dest):
				src1(src1),src2(src2),mask(mask),dest(dest){}
			~ParallelBlending(){}
			void operator()(const cv::BlockedRange& range)const{
				for(int i=range.begin();i!=range.end();i++){
					int y = i / mask.cols;
					int x = i % mask.cols;
					double weight1, weight2;
					setWeight<WeightType>(mask.at<WeightType>(y,x), &weight1,&weight2);
					dest.at<ElemType>(y,x) = blend<ElemType>(
							src1.at<ElemType>(y,x),
							src2.at<ElemType>(y,x),
							weight1,weight2);
				}
			}
		protected:
			const cv::Mat& src1;
			const cv::Mat& src2;
			const cv::Mat& mask;
			cv::Mat& dest;

	};

	template<class ElemType,class WeightType> cv::Mat blending(const cv::Mat& src1,const cv::Mat& src2, const cv::Mat& weight_mask){
		assert(weight_mask.size()==src1.size());
		assert(weight_mask.size()==src2.size());
		assert(src1.type()==src2.type());
		cv::Mat dest = cv::Mat::zeros(src1.size(),src1.type());
		cv::parallel_for(
				cv::BlockedRange(0,src1.rows*src1.cols),
				ParallelBlending<ElemType,WeightType>(src1,src2,weight_mask,dest)
				);
		return dest;
	}

	template<class ElemType,class WeightType> void blending(const cv::Mat& src1,const cv::Mat& src2, const cv::Mat& weight_mask, cv::Mat& dest){
		dest = blending<ElemType, WeightType>(src1,src2,weight_mask);
	}
	cv::Mat blur_mask(const cv::Mat& mask, size_t blur_width);
	void edge_difference(const cv::Mat& src1, const cv::Mat& src2, cv::Mat& edge1, cv::Mat& edge2, double canny_thresh1=16, double canny_thresh2=32, int aperture_size=3, int dilate_size=4);
}

#endif // __SKL_CV_UTILS_H__
