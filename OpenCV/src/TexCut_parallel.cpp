#include "TexCut_parallel.h"
#include "TexCut_def.h"
using namespace skl;

void ParallelCalcEdgeCapacity::operator()(const cv::BlockedRange& range)const{
	for(int i=range.begin();i<range.end();i++){
		int graph_width = data_term.cols;
		int graph_height = data_term.rows;
		int gx = i % graph_width;
		int gy = i / graph_width;
		cv::Rect roi;
		roi.x = gx * TEXCUT_BLOCK_SIZE;
		roi.y = gy * TEXCUT_BLOCK_SIZE;
		roi.width = TEXCUT_BLOCK_SIZE;
		roi.height = TEXCUT_BLOCK_SIZE;

		bool ignore_data_term = false;
		if(isOverUnderExposure(src,roi) || isOverUnderExposure(bg_img,roi)){
			data_term.at<int>(gy,gx) = QUANTIZATION_LEVEL_HARF;
#ifdef DEBUG
			tex_int.at<float>(gy,gx) = 0.0f;
#endif
			gradient_heterogenuity.at<float>(gy,gx) = 0.0f;
			ignore_data_term = true;
		}

		int channels = src.size();
		std::vector<float> t_i(channels, 0);
		std::vector<float> d_t(channels, 0);
		std::vector<float> s_x(channels, FLT_MAX);
		std::vector<float> s_y(channels, FLT_MAX);
		std::vector<float> g_h(channels,0);
		for(int c=0;c<channels;c++){
			cv::Mat _src = cv::Mat(src[c],roi);
			cv::Mat _sobel_x = cv::Mat(sobel_x[c],roi);
			cv::Mat _sobel_y = cv::Mat(sobel_y[c],roi);
			cv::Mat _bg = cv::Mat(bg_img[c],roi);
			cv::Mat _bg_sobel_x = cv::Mat(bg_sobel_x[c],roi);
			cv::Mat _bg_sobel_y = cv::Mat(bg_sobel_y[c],roi);


			if(!ignore_data_term){
				calcDataTerm(_sobel_x,_sobel_y,_bg_sobel_x,_bg_sobel_y,
						noise_std_dev[c], gh_expectation[c], gh_std_dev[c],
						&t_i[c],&g_h[c],&d_t[c]);
			}
			if(gx < graph_width -1){
				cv::Rect right = roi;
				right.x += TEXCUT_BLOCK_SIZE;
				cv::Mat _src_right = cv::Mat(src[c],right);
				cv::Mat _bg_right = cv::Mat(bg_img[c],right);
				calcSmoothingTerm(_src,_src_right,_bg,_bg_right,&s_x[c], noise_std_dev[c]);
			}

			if(gy < graph_height -1){
				_src = _src.clone();
				_src.t();
				_bg = _bg.clone();
				_bg.t();
				cv::Rect bottom = roi;
				bottom.y += TEXCUT_BLOCK_SIZE;
				cv::Mat _src_bottom = cv::Mat(src[c],bottom).clone();
				_src_bottom.t();
				cv::Mat _bg_bottom = cv::Mat(bg_img[c],bottom).clone();
				_bg_bottom.t();
				calcSmoothingTerm(_src,_src_bottom,_bg,_bg_bottom,&s_y[c],noise_std_dev[c]);
			}
		}
		if(!ignore_data_term){
			int argmax_tex = 0;
			for(int c = 1; c<channels;c++){
				argmax_tex = t_i[argmax_tex] > t_i[c] ? argmax_tex : c;
			}

#ifdef DEBUG
			tex_int.at<float>(gy,gx) = t_i[argmax_tex];
#endif
			gradient_heterogenuity.at<float>(gy,gx) = g_h[argmax_tex];

			assert(0 <= d_t[argmax_tex]);
			if(1.0f < d_t[argmax_tex]){
				d_t[argmax_tex] = 1.0f;
			}
			data_term.at<int>(gy,gx) = static_cast<int>(d_t[argmax_tex] * QUANTIZATION_LEVEL);
			//		std::cerr << d_t[argmax_tex] << ", " << data_term.at<int>(gy,gx) << std::endl;
		}
		if(gx < graph_width-1){
			float s_t = s_x[0];
			for(int c = 1; c<channels;c++){
				s_t = s_t > s_x[c] ? s_t : s_x[c];
			}
			s_t *= QUANTIZATION_LEVEL;
			s_t = s_t < QUANTIZATION_LEVEL ? s_t : QUANTIZATION_LEVEL;
			smoothing_term_x.at<int>(gy,gx) = static_cast<int>(s_t);
		}

		if(gy < graph_height-1){
			float s_t = s_y[0];
			for(int c = 1; c<channels;c++){
				s_t = s_t > s_y[c] ? s_t : s_y[c];
			}
			s_t *= QUANTIZATION_LEVEL;
			s_t = s_t < QUANTIZATION_LEVEL ? s_t : QUANTIZATION_LEVEL;
			smoothing_term_y.at<int>(gy,gx) = static_cast<int>(s_t);
		}
	}
}
int ParallelCalcEdgeCapacity::isOverUnderExposure(const cv::Mat& block)const{
	int exposure_state = 0;
	int temp;
	for(int y=0;y<block.rows;y++){
		for(int x=1;x<block.cols;x++){
			temp = 0;
			unsigned char val = block.at<unsigned char>(y,x);
			if(val > over_exposure_thresh) temp = 1;
			else if(val < under_exposure_thresh) temp = -1;
			if(temp==0) return 0;
			if(exposure_state==0){
				exposure_state = temp;
			}
			else if(exposure_state!=temp){
				return 0;
			}
		}
	}
	return exposure_state;
}

bool ParallelCalcEdgeCapacity::isOverUnderExposure(const std::vector<cv::Mat>& img_planes,const cv::Rect& roi)const{
	int _isOverUnderExposure = 0;
	for(size_t c=0;c<img_planes.size();c++){
		int check = isOverUnderExposure(cv::Mat(img_planes[c],roi));
		if(check == 0) return false;
	}
	return true;
}

void ParallelCalcEdgeCapacity::calcDataTerm(
		const cv::Mat& sobel_x, const cv::Mat& sobel_y,
		const cv::Mat& bg_sobel_x, const cv::Mat& bg_sobel_y,
		float nsd, float gh_mean, float gh_sd,
		float* tex_int, float* gh, float* tex_diff)const{
	float auto_cor_src(0),auto_cor_bg(0);
	float cross_cor(0);
	int square_size = TEXCUT_BLOCK_SIZE * TEXCUT_BLOCK_SIZE;
	std::vector<float> sort_tar_x(square_size,0);
	std::vector<float> sort_tar_y(sort_tar_x),bg_sort_tar_x(sort_tar_x),bg_sort_tar_y(sort_tar_x);
	for(int y=0,i=0;y<TEXCUT_BLOCK_SIZE;y++){
		for(int x=0;x<TEXCUT_BLOCK_SIZE;x++,i++){
			float ssx,ssy,bsx,bsy;
			ssx = static_cast<float>(sobel_x.at<unsigned char>(y,x));
			ssy = static_cast<float>(sobel_y.at<unsigned char>(y,x));
			bsx = static_cast<float>(bg_sobel_x.at<unsigned char>(y,x));
			bsy = static_cast<float>(bg_sobel_y.at<unsigned char>(y,x));
			auto_cor_src += (std::pow(ssx,2) + std::pow(ssy,2));
			auto_cor_bg += (std::pow(bsx,2) + std::pow(bsy,2));
			cross_cor += (ssx * bsx + ssy * bsy);

			sort_tar_x[i] = ssx;
			sort_tar_y[i] = ssy;
			bg_sort_tar_x[i] = bsx;
			bg_sort_tar_y[i] = bsy;
		}
	}
	// calc texture intenxity
	*tex_int = auto_cor_src > auto_cor_bg ? auto_cor_src : auto_cor_bg;
	*tex_int = sqrt(*tex_int/square_size);
	*tex_int = static_cast<float>(
		normalize(*tex_int,sqrt(3.0f) * nsd));

	// calc gradient heterogenuity
	float grad_hetero = calcGradHetero(sort_tar_x);
	float temp = calcGradHetero(sort_tar_y);
	grad_hetero = grad_hetero > temp ? grad_hetero : temp;
	temp = calcGradHetero(bg_sort_tar_x);
	grad_hetero = grad_hetero > temp ? grad_hetero : temp;
	temp = calcGradHetero(bg_sort_tar_y);
	grad_hetero = grad_hetero > temp ? grad_hetero : temp;
	grad_hetero = normalize(grad_hetero, 2*gh_sd, gh_mean);
	*gh = grad_hetero;

	float auto_cor = auto_cor_src + auto_cor_bg;
	if(auto_cor==0){
		*tex_int = 0;
		*tex_diff = 0.0f;
		return;
	}
	float normalized_correlation_dist = 1.0f - (2.0f * cross_cor)/auto_cor;
	assert(0 <= normalized_correlation_dist && normalized_correlation_dist <= 1.0);
	*tex_int = exp((grad_hetero * *tex_int) - 1.0f);
//	*tex_int = grad_hetero * *tex_int;
	*tex_diff = (thresh_tex_diff + *tex_int * ( normalized_correlation_dist - thresh_tex_diff )) / (2 * thresh_tex_diff);
/*
	if(*tex_diff > 0.5){
		std::cerr << *tex_int << ", " << normalized_correlation_dist << std::endl;
		std::cerr << *tex_diff << std::endl;
	}
*/
}

float ParallelCalcEdgeCapacity::normalize(float val,float sigma, float mean)const{
	val -= mean + sigma;
	if(val<0) return 0;
	val /= 2 * alpha * sigma;
	if(val>1) return 1;
	return val;
}

float ParallelCalcEdgeCapacity::calcGradHetero(std::vector<float>& power)const{
	std::sort(power.begin(),power.end(),std::greater<float>());
	float factor = power[power.size()/2];
	if(factor == 0.0f){
		if(power[0]==0.0f){
			return 0;
		}
		else{
			return FLT_MAX;
		}
	}
	return power[0]/factor;
}
void ParallelCalcEdgeCapacity::calcSmoothingTerm(
		const cv::Mat& src_left, const cv::Mat& src_right,
		const cv::Mat& bg_left, const cv::Mat& bg_right,
		float* smoothing_term, float nsd)const{
	float diff_l2r(0),diff_r2l(0);
	for(int i=0;i<TEXCUT_BLOCK_SIZE;i++){
		float s1,s2,b1,b2;
		s1 = src_left.at<unsigned char>(i,TEXCUT_BLOCK_SIZE-1);
		b1 = bg_left.at<unsigned char>(i,TEXCUT_BLOCK_SIZE-1);
		s2 = src_right.at<unsigned char>(i,TEXCUT_BLOCK_SIZE-1);
		b2 = bg_right.at<unsigned char>(i,TEXCUT_BLOCK_SIZE-1);
		diff_l2r += s1 - (b1 * (s2/b2));

		s1 = src_right.at<unsigned char>(i,0);
		b1 = bg_right.at<unsigned char>(i,0);
		s2 = src_left.at<unsigned char>(i,0);
		b2 = bg_left.at<unsigned char>(i,0);
		diff_r2l += s1 - (b1 * (s2/b2));
	}

	float diff = diff_l2r > diff_r2l ? diff_l2r : diff_r2l;
//	std::cerr << diff << " > " << (2*nsd * sqrt(2.0/TEXCUT_BLOCK_SIZE)) << std::endl;
	diff /= TEXCUT_BLOCK_SIZE;
	*smoothing_term = static_cast<float>(normalize(diff, nsd / sqrt(static_cast<float>(TEXCUT_BLOCK_SIZE))));
//	*smoothing_term = 1.0f - *smoothing_term;
	*smoothing_term = 1.0f - exp(*smoothing_term - 1.0f);
}

