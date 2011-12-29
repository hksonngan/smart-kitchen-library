#include <cv.h>

cv::Vec3b convHLS2BGR(const cv::Vec3b& hls);
cv::Vec3b assignColor(size_t ID);
cv::Mat visualizeRegionLabel(const cv::Mat& label,size_t region_num);

template<class ElemType,class WeightType> void blending(const cv::Mat& src1,const cv::Mat& src2, const cv::Mat& weight_mask, cv::Mat& dest);
