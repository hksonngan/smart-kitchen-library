#include "sklcvutils.h"
#include <highgui.h>
#include <algorithm>


#ifdef _DEBUG
#define DEBUG_SKLUTILS
#endif


namespace skl{
	/*!
	 * calc minimum rect which contains points
	 */
	cv::Rect fitRect(const std::vector< cv::Point >& points){
		cv::Rect rect(INT_MAX,INT_MAX,0,0);
		for(size_t i=0;i<points.size();i++){
			int x = points[i].x;
			int y = points[i].y;
			rect.x = rect.x < x ? rect.x : x;
			rect.y = rect.y < y ? rect.y : y;
			rect.width = rect.width > x ? rect.width : x;
			rect.height = rect.height > y ? rect.height : y;
		}
		rect.width -= (rect.x-1);
		rect.height -= (rect.y-1);
		return rect;
	}



	// 7*2*2 valiations of color
	cv::Vec3b assignColor(size_t ID){
		int hue_id = ID % 7;
		int lum_id = (int)ID / 7;
		int sat_id = lum_id / 3;
		lum_id %= 3;
		sat_id %= 2;
		if(hue_id == 6){
			cv::Vec3b bgr;
			int gray_level = 255 - (sat_id + lum_id * 2) * 32;
			for(int c=0;c<3;c++){
				bgr[c] = gray_level;
			}
			return bgr;
		}
		cv::Vec3b hls;
		hls[0] = hue_id * 30; // hue
		hls[1] = 255 - lum_id * 64;
		hls[2] = 255 - sat_id * 128;
		//	std::cerr << (int)hls[0] << ", " << (int)hls[1] << ", " << (int)hls[2] << std::endl;
		return convHLS2BGR(hls);
	}




	template<> void setWeight<unsigned char>(const unsigned char& mask, double* w1, double* w2){
		*w1 = mask;
		*w2 = 255 - mask;
		*w1 /= 255.0;
		*w2 /= 255.0;
	}

	template<> cv::Vec3b blend(const cv::Vec3b& pix1,const cv::Vec3b& pix2, double w1, double w2){
		cv::Vec3b val;
		for(size_t i=0;i<3;i++){
			val[i] = static_cast<unsigned char>(w1 * pix1[i] + w2 * pix2[i] + 0.5);
		}
		return val;
	}


	void edge_difference(const cv::Mat& src1, const cv::Mat& src2, cv::Mat& edge1, cv::Mat& edge2, double canny_thresh1, double canny_thresh2, int aperture_size, int dilate_size){
		assert(src1.size()==src2.size());
		cv::Mat gray1,gray2;
		if(src1.channels()==1){
			gray1 = src1.clone();
		}
		else{
			gray1 = cv::Mat(src1.size(),CV_8UC1);
			cv::cvtColor(src1,gray1,CV_BGR2GRAY);
		}
		if(src2.channels()==1){
			gray2 = src2.clone();
		}
		else{
			gray2 = cv::Mat(src2.size(),CV_8UC1);
			cv::cvtColor(src2,gray2,CV_BGR2GRAY);
		}

		cv::Mat _edge1 = cv::Mat(src1.size(),CV_8UC1);
		cv::Mat _edge2 = cv::Mat(src2.size(),CV_8UC1);
		cv::Canny(gray1,_edge1, canny_thresh1, canny_thresh2, aperture_size);
		cv::Canny(gray2,_edge2, canny_thresh1, canny_thresh2, aperture_size);
#ifdef DEBUG_SKLUTILS
		cv::namedWindow("edge1",0);
		cv::namedWindow("edge2",0);
		cv::imshow("edge1",_edge1);
		cv::imshow("edge2",_edge2);
#endif

		cv::Size kernel_size(dilate_size,dilate_size);
		cv::Mat dick_edge1 = cv::Mat(src1.size(),CV_8UC1);
		cv::Mat dick_edge2 = cv::Mat(src2.size(),CV_8UC1);
		cv::blur(_edge1,dick_edge1,kernel_size);
		cv::threshold(dick_edge1,dick_edge1,0,255,CV_THRESH_BINARY);
		cv::blur(_edge2,dick_edge2,kernel_size);
		cv::threshold(dick_edge2,dick_edge2,0,255,CV_THRESH_BINARY);

		edge1 = _edge1 - dick_edge2;
		edge2 = _edge2 - dick_edge1;
///
/*		cv::imwrite("src.png",src1);
		cv::imwrite("fg_edge.png",edge1);
		cv::imwrite("src_fg_edge.png",_edge1);
		cv::imwrite("bg_edge_dick.png",dick_edge2);

		cv::imwrite("bg.png",src2);
		cv::imwrite("bg_edge.png",edge2);
		cv::imwrite("src_bg_edge.png",_edge2);
		cv::imwrite("fg_edge_dick.png",dick_edge1);
*/
///
	}


template<typename ValType> bool _checkMat(
		const ValType val,
		const ValType condition,
		const ValType skipConditionVal, 
		const std::string& debug_comment){
	if(condition == skipConditionVal) return true;// skip
	if(val == condition) return true;
#ifdef _DEBUG
	std::cerr << "ERROR: " << debug_comment << std::endl;
	std::cerr << "       value '" << val << "' does not equal to '"<<condition<<"'." << std::endl;
#endif
	return false;
}
bool checkMat(const cv::Mat& mat, int depth,int channels,cv::Size size){
	bool result = true;
	if(mat.empty()) return false;
	result &= _checkMat(mat.depth(),depth,-1,"depth does not fit.");
	result &= _checkMat(mat.channels(),channels,0,"channels does not fit.");
	result &= _checkMat(mat.cols,size.width,0,"image width does not fit.");
	result &= _checkMat(mat.rows,size.height,0,"image height does not fit.");
	return result;
}

bool ensureMat(cv::Mat& mat, int depth,int channels,cv::Size size){
	if(checkMat(mat,depth,channels,size)) return true;
	mat = cv::Mat(size, CV_MAKETYPE(depth,channels));
	return !mat.empty();
}




template<typename Iter> cv::Mat fitModel(const cv::Mat& samples, Iter begin, Iter end){
	cv::Mat model(cv::Size(samples.cols,1),samples.type(),cv::Scalar(0));
	
	size_t count = 0;
	cv::Rect roi(0,0,samples.cols,1);
	for(Iter it = begin; it != end; it++){
		size_t idx = *it;
		roi.y = idx;
		model += cv::Mat(samples,roi);
		count++;
	}
	return model / count;
}

template<typename Iter>double calcError(const cv::Mat& samples, const cv::Mat& model, Iter begin, Iter end){
	double error = 0;
	size_t count = 0;
	
	cv::Rect roi(0,0,samples.cols,1);
	for(Iter it = begin; it != end; it++){
		size_t idx = *it;
		roi.y = idx;
		error += cv::norm(model, cv::Mat(samples,roi));
		count++;
	}
	return error / count;
}

cv::Mat ransac(const cv::Mat& samples, cv::TermCriteria termcrit, double thresh_outliar, double sampling_rate, double minimum_inliar_rate){
	assert(0 < sampling_rate && sampling_rate <= 1);
	assert(0 < minimum_inliar_rate && minimum_inliar_rate <= 1);


	cv::Mat best_model(cv::Size(samples.cols,1),samples.type(),cv::Scalar(0));
	size_t dim = samples.cols;
	size_t sample_num = samples.rows;

	size_t iterations = 0;
	size_t sampling_num = sample_num * sampling_rate;
	size_t minimum_inliar_num = sample_num * minimum_inliar_rate;

	double best_error = DBL_MAX;

	std::vector<size_t> sample_index(sample_num);
	for(size_t i=0;i<sample_num;i++) sample_index[i] = i;


	while( ( (termcrit.type & CV_TERMCRIT_ITER) == 0 || iterations < (size_t)termcrit.maxCount) 
		&& ( (termcrit.type & CV_TERMCRIT_EPS) == 0 || best_error > termcrit.epsilon) ){
		// main loop

		// random sampling (use first part as sampled data.)
		std::random_shuffle(sample_index.begin(),sample_index.end());

		std::vector<size_t> consensus_set(sampling_num);
		std::copy(sample_index.begin(),sample_index.begin()+sampling_num,consensus_set.begin());

		cv::Mat this_model = fitModel(samples,sample_index.begin(),sample_index.begin()+sampling_num);

		// do process for not-sampled elements.
		cv::Rect roi(0,0,dim,1);
		for(size_t _i=sampling_num;_i<sample_num;_i++){
			size_t idx = sample_index[_i];
			
			roi.y = idx;
			if(cv::norm(this_model,cv::Mat(samples,roi))<thresh_outliar){
				consensus_set.push_back(idx);
			}
		}

		// increment iteration before checking validity
		iterations++;

		// check validity
		if(consensus_set.size() < minimum_inliar_num) continue;
		if(consensus_set.size() > sampling_num){
			cv::Mat refinement = fitModel(samples,consensus_set.begin()+sampling_num,consensus_set.end());
			this_model = ( sampling_num * this_model + (consensus_set.size()-sampling_num) * refinement ) /consensus_set.size();
		}
		double this_error = calcError(samples, this_model, consensus_set.begin(),consensus_set.end());
		if(this_error < best_error){
			best_error = this_error;
			best_model = this_model;
		}

	}

	return best_model;
}


}// namespace skl
