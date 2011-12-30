#ifndef __SKL_TEXCUT_PARALLEL_H__
#define __SKL_TEXCUT_PARALLEL_H__
#include <cv.h>
#include "TexCut_def.h"
#include "sklutils.h"

namespace skl{

	class ParallelNoiseEstimate{
		public:
			ParallelNoiseEstimate(
					std::vector<cv::Mat>* img1,
					std::vector<cv::Mat>* img2,
					std::vector<float>* noise_std_dev,
					std::vector<float>* gh_expectation,
					std::vector<float>* gh_std_dev
					):img1(img1),img2(img2),noise_std_dev(noise_std_dev),gh_expectation(gh_expectation),gh_std_dev(gh_std_dev){}

			void operator()(const cv::BlockedRange& range)const{
				for(int c=range.begin();c!=range.end();c++){
					float nsd, ghe, ghsd;
					noiseEstimate(img1->at(c),img2->at(c),&nsd);
					ghNoiseEstimate(nsd,&ghe,&ghsd);

					noise_std_dev->at(c) = nsd;
					gh_expectation->at(c) = ghe;
					gh_std_dev->at(c) = ghsd;
				}
			}
		protected:
			std::vector<cv::Mat>* img1;
			std::vector<cv::Mat>* img2;
			std::vector<float>* noise_std_dev;
			std::vector<float>* gh_expectation;
			std::vector<float>* gh_std_dev;

		private:
			void ghNoiseEstimate(
					float noise_std_dev,
					float* gh_expectation,
					float* gh_std_dev)const{
				size_t iteration_time = GRADIENT_HETEROGENUITY;
				float moment1(0),moment2(0);
				int elem_num = TEXCUT_BLOCK_SIZE * TEXCUT_BLOCK_SIZE;
				std::vector<float> powers(elem_num,0.0);
				for(size_t i = 0; i < iteration_time; i++){
					for(int e = 0; e < elem_num; e++){
						powers[e] = static_cast<float>(rayleigh_rand(std::sqrt(6.0)*noise_std_dev));
					}
					std::sort(powers.begin(),powers.end(),std::greater<float>());
					float factor = powers[powers.size()/2];
					if(factor==0){
						i--;
						continue;
					}
					moment1 += powers[0]/factor;
					moment2 += std::pow(powers[0]/factor,2);
				}
				moment1 /= iteration_time;
				moment2 /= iteration_time;
				*gh_expectation = moment1;
				*gh_std_dev = 2 * sqrt(moment2 - std::pow(moment1,2));
			}

			void noiseEstimate(
					const cv::Mat& img1,
					const cv::Mat& img2,
					float* noise_std_dev)const{
				float moment1(0),moment2(0);
				float temp1(0),temp2(0);
				for(int y = 0; y < img1.rows; y++){
					temp1 = 0;
					temp2 = 0;
					for(int x = 0; x < img2.cols; x++){
						int diff = 
							static_cast<int>(img1.at<unsigned char>(y,x))
							- img2.at<unsigned char>(y,x);
						temp1 += diff;
						temp2 += diff * diff;
					}
					moment1 += temp1/img1.cols;
					moment2 += temp2/img2.cols;
				}
				moment1 /= img1.rows;
				moment2 /= img2.rows;
				*noise_std_dev = (float)sqrt(moment2 - std::pow(moment1,2));
			}
	};
	class ParallelAddGradientHeterogenuity{
		public:
			ParallelAddGradientHeterogenuity(
					const cv::Mat& data_term,
					const cv::Mat& gradient_heterogenuity,
					cv::Mat& smoothing_term_x,
					cv::Mat& smoothing_term_y
					):
				data_term(data_term), gradient_heterogenuity(gradient_heterogenuity),
				smoothing_term_x(smoothing_term_x),smoothing_term_y(smoothing_term_y){}
			void operator()(const cv::BlockedRange& range)const{
				for(int i = range.begin();i != range.end(); i++){
					int graph_width = data_term.cols;
					int graph_height = data_term.rows;
					int gx = i % graph_width;
					int gy = i / graph_width;
					int data_term_penalty = calcDataTermPenalty(data_term.at<int>(gy,gx));
					float grad_hetero_penalty = gradient_heterogenuity.at<float>(gy,gx);
					int dt_x = data_term_penalty;
					int dt_y = data_term_penalty;

					float gh_x = grad_hetero_penalty;
					float gh_y = grad_hetero_penalty;

					if(gx < graph_width-1){
						int temp = calcDataTermPenalty(data_term.at<int>(gy,gx+1));
						dt_x = dt_x > temp ? dt_x : temp;
						float temp2 = gradient_heterogenuity.at<float>(gy,gx+1);
						gh_x = gh_x > temp2 ? exp(-gh_x) : exp(-temp2);
						dt_x = static_cast<int>(gh_x * QUANTIZATION_LEVEL);
						int s_x = smoothing_term_x.at<int>(gy,gx) + dt_x;
						smoothing_term_x.at<int>(gy,gx) = s_x > 0 ? s_x : 0;

					}
					if(gy < graph_height-1){
						int temp = calcDataTermPenalty(data_term.at<int>(gy+1,gx));
						dt_y = dt_y > temp ? dt_y : temp;
						float temp2 = gradient_heterogenuity.at<float>(gy+1,gx);
						gh_y = gh_y > temp2 ? exp(-gh_y) : exp(-temp2);
						dt_y = static_cast<int>(gh_y * QUANTIZATION_LEVEL);
						int s_y = smoothing_term_y.at<int>(gy,gx) + dt_y;
						smoothing_term_y.at<int>(gy,gx) = s_y > 0 ? s_y : 0;
					}
				}
			}
			int calcDataTermPenalty(int val)const{
				if(val > QUANTIZATION_LEVEL_HARF){
					return 0;
				}
				return QUANTIZATION_LEVEL_HARF - val;
			}
		protected:
			const cv::Mat& data_term;
			const cv::Mat& gradient_heterogenuity;
			cv::Mat& smoothing_term_x;
			cv::Mat& smoothing_term_y;
		private:
	};
	class ParallelCalcEdgeCapacity{
		public:
			ParallelCalcEdgeCapacity(
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
					float thresh_tex_diff,
					unsigned char over_exposure_thresh,
					unsigned char under_exposure_thresh,
					cv::Mat& data_term,
#ifdef DEBUG
					cv::Mat& tex_int,
#endif
					cv::Mat& gradient_heterogenuity,
					cv::Mat& smoothing_term_x,
					cv::Mat& smoothing_term_y
						):src(src),sobel_x(sobel_x),sobel_y(sobel_y),
						bg_img(bg_img),bg_sobel_x(bg_sobel_x),bg_sobel_y(bg_sobel_y),
						noise_std_dev(noise_std_dev),gh_expectation(gh_expectation),
						gh_std_dev(gh_std_dev),alpha(alpha),
						thresh_tex_diff(thresh_tex_diff),
						over_exposure_thresh(over_exposure_thresh),under_exposure_thresh(under_exposure_thresh),
						data_term(data_term),
#ifdef DEBUG
						tex_int(tex_int),
#endif
						gradient_heterogenuity(gradient_heterogenuity),
						smoothing_term_x(smoothing_term_x),
						smoothing_term_y(smoothing_term_y){}
			void operator()(const cv::BlockedRange& range)const;
		protected:
			const std::vector<cv::Mat>& src;
			const std::vector<cv::Mat>& sobel_x;
			const std::vector<cv::Mat>& sobel_y;
			const std::vector<cv::Mat>& bg_img;
			const std::vector<cv::Mat>& bg_sobel_x;
			const std::vector<cv::Mat>& bg_sobel_y;
			const std::vector<float>& noise_std_dev;
			const std::vector<float>& gh_expectation;
			const std::vector<float>& gh_std_dev;
			float alpha;
			float smoothing_term_weight;
			float thresh_tex_diff;
			unsigned char over_exposure_thresh;
			unsigned char under_exposure_thresh;
			cv::Mat& data_term;
#ifdef DEBUG
			cv::Mat& tex_int;
#endif
			cv::Mat& gradient_heterogenuity;
			cv::Mat& smoothing_term_x;
			cv::Mat& smoothing_term_y;

			void calcDataTerm(
					const cv::Mat& sobel_x, const cv::Mat& sobel_y,
					const cv::Mat& bg_sobel_x, const cv::Mat& bg_sobel_y,
					float nsd,float gh_mean,float gh_sd,
					float* tex_int, float* gh, float* tex_diff)const;
			float calcGradHetero(std::vector<float>& power)const;
			void calcSmoothingTerm(
					const cv::Mat& src_left, const cv::Mat& src_right,
					const cv::Mat& bg_left, const cv::Mat& bg_right,
					float* smoothing_term, float nsd)const;
			float normalize(float val, float sigma, float mean = 0)const;
			int isOverUnderExposure(const cv::Mat& block)const;
			bool isOverUnderExposure(const std::vector<cv::Mat>& img_planes,const cv::Rect& roi)const;
	};

}

#endif // __SKL_TEXCUT_PARALLEL_H__
