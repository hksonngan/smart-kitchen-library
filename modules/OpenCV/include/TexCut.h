#ifndef __TEXCUT_H__
#define __TEXCUT_H__
#include "skl.h"
#include "sklcv.h"
#include <highgui.h>
#include "BackgroundSubtractAlgorithm.h"
#include "TexCut_def.h"

namespace skl{
	class TexCut: public BackgroundSubtractAlgorithm{
		public:
			TexCut();
			TexCut(const cv::Mat& bg1, const cv::Mat& bg2, float alpha=1.5, float smoothing_term_weight=1.5, float thresh_tex_diff = 0.4,unsigned char over_exposure_thresh = 248,unsigned char under_exposure_thresh = 8);
			~TexCut();
			void setBackground(const cv::Mat& bg);
			void setParams(float alpha=1.5, float smoothing_term_weight=1.5, float thresh_tex_diff = 0.4, unsigned char over_exposure_thresh = 248,unsigned char under_exposure_thresh = 8);
			void learnImageNoiseModel(const cv::Mat& bg2);

			void updateBackgroundModel(const cv::Mat& bg, const cv::Mat& mask);
		protected:
			double compute(const cv::Mat& src,const cv::Mat& mask, cv::Mat& dest);
			void calcEdgeCapacity(
					const std::vector<cv::Mat>& src,
					const std::vector<cv::Mat>& sobel_x,
					const std::vector<cv::Mat>& sobel_y,
					const std::vector<cv::Mat>& bg_img,
					const std::vector<cv::Mat>& bg_sobel_x,
					const std::vector<cv::Mat>& bg_sobel_y,
					const std::vector<float>& noise_std_dev,
					const std::vector<float>& gh_expectation,
					const std::vector<float>& gh_std_dev,
					float alpha,
					float smoothing_term_weight,
					float thresh_tex_diff,
					cv::Mat& data_term,
					cv::Mat& smoothing_term_x,
					cv::Mat& smoothing_term_y);
			int calcGraphCut(const cv::Mat& data_term,const cv::Mat& smoothing_term_x, const cv::Mat& smoothing_y);
			void setResult(const cv::Mat& src,cv::Mat& dest)const;
			void getSobel(
					const std::vector<cv::Mat>& img,
					std::vector<cv::Mat>* sobel_x,
					std::vector<cv::Mat>* sobel_y);
			std::vector<cv::Mat> bg_img;
			std::vector<cv::Mat> bg_sobel_x;
			std::vector<cv::Mat> bg_sobel_y;
			std::vector<float> noise_std_dev;
			std::vector<float> gh_expectation;
			std::vector<float> gh_std_dev;
			float alpha;
			float smoothing_term_weight;
			float thresh_tex_diff;
			unsigned char over_exposure_thresh;
			unsigned char under_exposure_thresh;

			TexCutGraph* g;
			std::vector<std::vector<TexCutGraph::node_id> > nodes;
		private:
			void setCapacity(
					TexCutGraph* g,std::vector<std::vector<TexCutGraph::node_id> >& nodes, size_t x,size_t y, const cv::Mat& data_term,const cv::Mat& smoothing_term_x, const cv::Mat& smoothing_term_y);
	};

} // namespace skl
#endif // __TEXCUT_H__

