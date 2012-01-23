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

	/*!
	 * @brief get non Zero value points
	 * */
	template<class ElemType> void getPoints(
			const cv::Mat& mask, std::vector<cv::Point>& points,
			ElemType val = 0){
		points.clear();
		for(int y=0;y<mask.rows;y++){
			const ElemType* pix = mask.ptr<const ElemType>(y);
			for(int x=0;x<mask.rows;x++){
				if(val == pix[x]) continue;
				points.push_back(cv::Point(x,y));
			}
		}
	}

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


	cv::Mat blur_mask(const cv::Mat& mask, size_t blur_width);

	void edge_difference(const cv::Mat& src1, const cv::Mat& src2, cv::Mat& edge1, cv::Mat& edge2, double canny_thresh1=32, double canny_thresh2=64, int aperture_size=3, int dilate_size=4);


	template <class LabelType> void resize_label(const cv::Mat& label_small,size_t scale, cv::Mat& label){
		assert(label_small.channels()==1);
		assert(label_small.type()==label.type());
		cv::Size label_size(label_small.size());
		label_size.width *= scale;
		label_size.height *= scale;
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

			for(int lx = 0; lx < label.cols; lx+=scale,x++){
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

	template <class LabelType> cv::Mat resize_label(const cv::Mat& label_small,size_t scale){
		cv::Size label_size = label_small.size();
		label_size.width *= scale;
		label_size.height *= scale;
		cv::Mat label = cv::Mat(label_size,label_small.type());
		resize_label<LabelType>(label_small,scale,label);
		return label;
	}

}

#endif // __SKL_CV_UTILS_H__
