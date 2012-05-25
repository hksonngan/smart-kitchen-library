/*!
 * @file BackgroundCut.h
 * @author a_hasimoto
 * @date Date Created: 2012/Feb/13
 * @date Last Change:2012/Apr/30.
 */
#ifndef __SKL_BACKGROUND_CUT_H__
#define __SKL_BACKGROUND_CUT_H__

// opencv
#include <cv.h>
#include <highgui.h>
#include <ml.h>
#include "skl.h"
#include "BackgroundSubtractAlgorithm.h"

#include "../Core/include/graph.h"

namespace skl{

	/*!
	 * @brief Background Subtraction by Background Cut
	 * @reffer Jian Sun, Weiwei Zhang, Xiaoou Tang, and Heung-Yeung Shum "Background Cut," ECCV 2006
	 */
	class BackgroundCut : public BackgroundSubtractAlgorithm{
		typedef Graph<int,int,int> BackgroundCutGraph;
		public:
			using BackgroundSubtractAlgorithm::compute;
			BackgroundCut(float thresh_bg=2,float thresh_fg=5,float sigma_KL=1,float K=5, float sigma_z=10,float learning_rate=0.2,int bg_cluster_num=15,int fg_cluster_num=5);
			virtual ~BackgroundCut();
			void setParams(float thresh_bg, float thresh_fg, float sigma_KL,float K=5, float sigma_z=10, float learning_rate=0.2, int bg_cluster_num=15, int fg_cluster_num=5);
			void background(const cv::Mat& background);
			cv::Mat background()const{return _background;}
			void updateBackgroundModel(const cv::Mat& img);
		protected:
			void compute(const cv::Mat& src,const cv::Mat& mask,cv::Mat& dest);
			cv::Mat _background;
			CvEM bg_global_model_algo[2];
			CvEM fg_model_algo[2];
			bool model_flag_bg;
			bool model_flag_fg;
			CvEMParams bg_global_model;
			CvEMParams fg_model;
			cv::Mat noise_variance;

			float thresh_bg;
			float thresh_fg;
			float learning_rate;
			float sigma_KL;
			float Ksquared;
			float sigma_z;

			// graph
			BackgroundCutGraph* graph;
			std::vector<std::vector<BackgroundCutGraph::node_id> > nodes;

			cv::Mat data_term_fg;
			cv::Mat data_term_bg;
			cv::Mat hCue;
			cv::Mat vCue;


			void roughSegmentation(
					const cv::Mat& src,
					const cv::Mat& bg,
					const cv::Mat& noise_variance,
					float thresh_bg,
					float thresh_fg,
					cv::Mat& labels);
			bool learnGMM(
					const cv::Mat& src,
					const cv::Mat& mask,
					CvEM* gmm_model_algo,
					CvEMParams* gmm_model,
					float skip_rate = 0.f);
			void calcDataTerm(
					const cv::Mat& src,
					const CvEMParams& fg_model,
					const CvEMParams& bg_global_model,
					const cv::Mat& bg,
					const cv::Mat& noise_variance,
					cv::Mat& data_term_fg,
					cv::Mat& data_term_bg);
			void calcSmoothingTerm(
					const cv::Mat& src,
					const cv::Mat& bg,
					cv::Mat& hCue,
					cv::Mat& vCue);

			BackgroundCutGraph* createGraph(
					const cv::Mat& data_term_fg,
					const cv::Mat& data_term_bg,
					const cv::Mat& hCue,
					const cv::Mat& vCue,
					std::vector<std::vector<BackgroundCutGraph::node_id> >& nodes);
	};

} // skl

#endif // __SKL_BACKGROUND_CUT_H__

