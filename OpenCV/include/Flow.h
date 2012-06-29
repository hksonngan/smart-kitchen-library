/*!
 * @file Flow.h
 * @author a_hasimoto
 * @date Date Created: 2012/Jun/29
 * @date Last Change:2012/Jun/29.
 */
#ifndef __SKL_FLOW_H__
#define __SKL_FLOW_H__

#include <cv.h>
namespace skl{

/*!
 * @class Flow
 * @brief Class for optical flow
 */
class Flow{

	public:
		Flow();
		Flow(const cv::Mat& u,const cv::Mat& v);
		virtual ~Flow();
		cv::Mat u;
		cv::Mat v;
		void distance(cv::Mat& r);
		inline cv::Mat distance(){
			cv::Mat r;
			distance(r);
			return r;
		}
		void angle(cv::Mat& rad, float offset_rad=0,float origin_return_value=0);
		inline cv::Mat angle(float offset_rad=0,float origin_return_value=0){
			cv::Mat rad;
			angle(rad,offset_rad,origin_return_value);
			return rad;
		}

		cv::Mat visualize(cv::Mat& base, int interval=8,const cv::Scalar& arrow_color=CV_RGB(255,0,0));
		bool read(const std::string& filename);
		bool write(const std::string& filename);
	protected:
		bool isValid()const;
	private:
};

} // skl

#endif // __SKL_FLOW_H__

