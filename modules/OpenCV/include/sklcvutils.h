#pragma warning(disable:4996)
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

	cv::Rect fitRect(const std::vector< cv::Point >& points);

	cv::Vec3b convHLS2BGR(const cv::Vec3b& hls);
	cv::Vec3b assignColor(size_t ID);
	cv::Mat visualizeRegionLabel(const cv::Mat& label,size_t region_num);

	template<class ElemType,class WeightType> void blending(const cv::Mat& src1,const cv::Mat& src2, const cv::Mat& weight_mask, cv::Mat& dest);
}
