#ifndef __SKL_UTILS_BAYER_CONVERSION_H__
#define __SKL_UTILS_BAYER_CONVERSION_H__

namespace skl{
	typedef enum{
		BAYER_SIMPLE,	//!< $BC1=c$J%Y%$%d!<JQ49(B
		BAYER_NN,		//!< NN$B$r9MN8$7$?%Y%$%d!<(B
		BAYER_EDGE_SENSE,//,//!< $B%(%C%8$r9MN8$7$?%Y%$%d!<(B
	} ColorInterpolationType;
	void cvtBayer2BGR(const cv::Mat& bayer, cv::Mat& bgr, int code=CV_BayerBG2BGR, int	algo_type=BAYER_SIMPLE);



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

}

#endif // __SKL_CVUTILS_BAYER_CONVERSION__