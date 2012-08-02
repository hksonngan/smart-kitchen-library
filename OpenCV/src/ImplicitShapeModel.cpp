/*!
 * @file ImplicitShapeModel.cpp
 * @author a_hasimoto
 * @date Date Created: 2012/Jul/11
 * @date Last Change: 2012/Jul/31.
 */
#include "ImplicitShapeModel.h"
#include <fstream>
#include <set>
using namespace skl;


/*!
 * @brief デフォルトコンストラクタ
 */
ImplicitShapeModel::ImplicitShapeModel(
		const cv::Mat& __vocaburary,
		float entry_threshold,
		float hypothesis_threshold,
		float kapper1,
		float kapper2,
		float object_kernel_ratio,
		float std_size,
		FeatureType feature_type):
	_entry_threshold(entry_threshold),
	_hypothesis_threshold(hypothesis_threshold),
	_kapper1(kapper1),
	_kapper2(kapper2),
	_object_kernel_ratio(object_kernel_ratio),
	_std_size(std_size),
	_feature_type(feature_type),
	hasVocaburary(false)
	{
	vocaburary(__vocaburary);
}

/*!
 * @brief デストラクタ
 */
ImplicitShapeModel::~ImplicitShapeModel(){

}

/*
 * @brief release data.
 * */
void ImplicitShapeModel::release(){
	_vocaburary = cv::Mat();
	occurrences.clear();
	v_norm.clear();
	hasVocaburary = false;

}

/*
 * @brief set vocaburary
 * */
void ImplicitShapeModel::vocaburary(const cv::Mat& vocaburary){
	release();
	if(vocaburary.empty()) return;
	this->_vocaburary = vocaburary.t();
	occurrences.assign(_vocaburary.cols,ISMEntries());
	getNorm(vocaburary,v_norm);
	hasVocaburary = true;
}


/*
 * @brief calculate relative location from v2 to v1.
 * @param v1 end point
 * @param v2 start point
 * @param scale_x scale for x direction
 * @param scale_y scale for y direction
 * */
cv::Point2f ImplicitShapeModel::getRelativeLocation(
		const cv::KeyPoint& v1,
		const cv::KeyPoint& v2,
		float scale_x,
		float scale_y){
	cv::Point2f pt;
	pt.x = (v1.pt.x/v1.size - v2.pt.x/v2.size) * scale_x;
	pt.y = (v1.pt.y/v1.size - v2.pt.y/v2.size) * scale_y;

	float rad = std::max(v1.angle,0.f) - std::max(v2.angle,0.f);
//	std::cerr << center.angle << ", " << abs_pt.angle << std::endl;
//	std::cerr << rad << "rad." << std::endl;
	// check and confirm this rotation calculation!!
	if(rad!=0){
		cv::Point2f temp(pt);
		pt.x = temp.x * cos(rad) - temp.y * sin(rad);
		pt.y = temp.x * sin(rad) + temp.y * cos(rad);
	}
	return pt;
}

/*
 * @brief calculate similarity between each sample feature and word.
 * @parames features sample feature set. a column must be a samplel feature.
 * @params similarity calculated similarities.
 * @return if it successed or not.
 * */
bool ImplicitShapeModel::getSimilarity(const cv::Mat& features,cv::Mat& similarity)const{
	if(features.cols != _vocaburary.rows) return false;
	similarity = features * _vocaburary;
	if(_feature_type != NORMALIZED){
		std::vector<float> norm;
		getNorm(features,norm);
		for(int i=0;i<similarity.rows;i++){
			for(int w=0; w<similarity.cols;w++){
				similarity.at<float>(i,w) /= 
					std::max(norm[i],v_norm[w]);
			}
		}
	}
	return true;
}

/*
 * @brief draw 'x' mark to dist at pt.
 * */
void drawX(cv::Mat& dist,const cv::Point2f& pt, const cv::Scalar& col,int size){
	cv::Point2f pt1(pt.x-size,pt.y-size);
	cv::Point2f pt2(pt.x+size,pt.y+size);
	cv::line(dist,pt1,pt2,col);

	pt1.y+=2*size;
	pt2.y-=2*size;
	cv::line(dist,pt1,pt2,col);
}


/*
 * @brief draw '+' mark to dist at pt.
 * */
void drawplus(cv::Mat& dist,const cv::Point2f& pt, const cv::Scalar& col,int size){
	cv::Point2f pt1(pt.x,pt.y-size);
	cv::Point2f pt2(pt.x,pt.y+size);
	cv::line(dist,pt1,pt2,col);

	pt1 = cv::Point2f(pt.x-size,pt.y);
	pt2 = cv::Point2f(pt.x+size,pt.y);
	cv::line(dist,pt1,pt2,col);
}

/*
 * @brief visualize occurrence and its votes.
 * */
cv::Mat ImplicitShapeModel::visualize(
		const cv::Mat& base_image,
		const cv::Mat& features,
		const std::vector<cv::KeyPoint>& keypoints,
		const std::vector<std::vector<ISMEntry> >& _occurrences,
		const std::map<int,cv::Scalar>& word_color_map)const{
	if(keypoints.empty()) return base_image;
	cv::Mat dist = base_image.clone();
	cv::Mat similarities;
	if(!getSimilarity(features,similarities)) return dist;

	cv::Scalar nomatch_col(127,0,127);

	std::map<int,cv::Scalar>::const_iterator pwc;
	for(size_t i=0;i<keypoints.size();i++){
		int matched_word_count = 0;
		for(pwc = word_color_map.begin();pwc!=word_color_map.end();pwc++){
			int w = pwc->first;
			cv::Scalar col = pwc->second;
//			std::cerr << w << ", " << i << std::endl;
			if(similarities.at<float>(i,w)<_entry_threshold) continue;
			cv::circle(dist,keypoints[i].pt,matched_word_count+2,col);
			for(size_t e=0; e <_occurrences[w].size();e++){
				cv::Point2f voted_pt(
						keypoints[i].pt.x + _occurrences[w][e].pt().x,
						keypoints[i].pt.y + _occurrences[w][e].pt().y);
				cv::line(dist,keypoints[i].pt,voted_pt,col);
				drawplus(dist,voted_pt,col,3);
			}
			matched_word_count++;
		}
		if(matched_word_count==0){
			drawX(dist,keypoints[i].pt,nomatch_col,5);
		}
	}

	return dist;
}

/*
 * @brief incremental learning for each object/image.
 * @params features feature for each point in the image.
 * @params feature_locations location for each point
 * @params patch around each point. This parameter can be skipped by setting NULL.
 * @params shape_location object location with its size and angle.
 * @params class_response object class ID.
 * @params current_occurrences generated occurrences by this sample. This is used by menber function "visualize", and can be skipped by setting NULL.
 * */
bool ImplicitShapeModel::add(
		const cv::Mat& features,
		const std::vector<cv::KeyPoint>& feature_locations,
		const std::vector<cv::Mat>* patches,
		cv::KeyPoint& shape_location,
		int class_response,
		std::vector<std::vector<ISMEntry> >* current_occurrences){
	assert(hasVocaburary);
	if(_std_size==0) return false;
	if(shape_location.size ==0) return false;
//	std::cerr << _std_size << "/" << shape_location.size << std::endl;
	float scale = _std_size / shape_location.size;

	if(current_occurrences!=NULL){
		current_occurrences->assign(occurrences.size(),std::vector<ISMEntry>());
	}

	// similarity (inner product) calculation
	cv::Mat similarities;
	if(!getSimilarity(features,similarities)) return false;

	std::map<int,std::vector<size_t> >::iterator psn;

	// do for each sample
	for(int i = 0; i < features.rows; i++){
		cv::Point2f pt = getRelativeLocation(shape_location,feature_locations[i],scale);

		// do for each visual word
		for(int w = 0; w < _vocaburary.cols;w++){
			float sim = similarities.at<float>(i,w);
			if(sim < _entry_threshold) continue;
			cv::Mat patch;
			if(patches!=NULL){
				patch = patches->at(i);
			}
			ISMEntry entry(pt,std::vector<float>(2,scale),class_response,patch);
			occurrences[w].insert(entry);
			psn = word_hit_num.find(entry.class_response());
			if(word_hit_num.end()==psn){
				word_hit_num[entry.class_response()] = std::vector<size_t>(wordNum(),0);
				psn = word_hit_num.find(entry.class_response());
			}
			psn->second[w]++;

			// save current muched occurrence if needed.
			if(NULL != current_occurrences){
				current_occurrences->at(w).push_back(entry);
			}
		}
	}
	return true;
}


/*
 * @brief calculate norm for each feature to normalize the correlation. The way of Norm calculation is depend on member variable "_feature_type."
 * @param features sample features, whose norm will be calculated.
 * @param norm calculated norms.
 * */
void ImplicitShapeModel::getNorm(
		const cv::Mat& features,
		std::vector<float>& norm)const{
	norm.resize(features.rows,1);
	cv::Rect roi(0,0,features.cols,1);
	if(_feature_type == POSITIVE){
		for(int i=0;i<features.rows;i++){
			roi.y = i;
			norm[i] = cv::sum(features(roi))[0];
		}
	}
	else if(_feature_type == OTHER){
		cv::Mat sqr = features.mul(features);
		for(int i=0;i<features.rows;i++){
			roi.y = i;
			norm[i] = cv::sum(sqr(roi))[0];
		}
	}
}

/*
 * @brief calculate Euclid distance.
 * */
inline float distance(const cv::Point2f& a,const cv::Point2f& b){
	return sqrt( std::pow(a.x-b.x,2.f) + std::pow(a.y-b.y,2.f) );
}

/*
 * @brief predict existance of the objects and its class.
 * @param features sample point features from current image.
 * @param locations locations of predicted(detected) objects.
 * @param scales scales of predicted(detected) objects.
 * @param response_likelihoods class responses for each object locations.
 * @param voting_images images of voting space for each class.
 * @return if it successed or not.
 * */
bool ImplicitShapeModel::predict(
		const cv::Mat& features,
		const std::vector<cv::KeyPoint>& feature_points,
		std::vector<cv::KeyPoint>& locations,
		std::vector<float>& scales,
		std::vector<std::map<int,float> >& response_likelihoods,
		std::map<int, cv::Mat>* voting_images,
		const cv::Size& size)const{

	// similarity (inner product) calculation
	cv::Mat similarities;
	if(!getSimilarity(features,similarities)) return false;

	// do for each sample

	return true;
}

/*
 * @brief predict object class.
 * @param features sample point features from current image.
 * @param location location of the object.
 * @param scale scale of the object.
 * @param response_likelihood class responses for directed object locations.
 * @param voting_images images of voting space for each class.
 * @return if it successed or not.
*/
bool ImplicitShapeModel::predict(
		const cv::Mat& features,
		const std::vector<cv::KeyPoint>& feature_points,
		const cv::KeyPoint& location,
		float kernel_size,
		std::map<int,float>& response_likelihood,
		std::map<int,cv::Mat>* voters,
		const cv::Size& size)const{
	response_likelihood.clear();

	// similarity (inner product) calculation
	cv::Mat similarities;
	if(!getSimilarity(features,similarities)) return false;

	// do for each object locations
//	float kernel_size = sqrt(location.size);

	bool visualize = false;
	std::map<int,cv::Mat>::iterator pimg;
	if(voters != NULL && size.width > 0 && size.height > 0){
		visualize = true;
		for(pimg = voters->begin();pimg != voters->end();pimg++){
			pimg->second = cv::Scalar(0);
		}
	}
	float scale = _std_size / location.size;

	std::map<int,int> class_index;
	std::vector<int> class_index_inv;
	std::map<int,float>::iterator pvote;
	size_t class_num = 0;
	for(std::map<int,std::vector<size_t> >::const_iterator pp=word_hit_num.begin();
			pp!=word_hit_num.end();pp++,class_num++){
		class_index[pp->first] = class_num;
		class_index_inv.push_back(pp->first);
		response_likelihood[pp->first] = 0;
	}
	std::vector<size_t> whole_word_hit_num(class_num,0);
	size_t total_entry_num = 0;
	for(size_t c=0;c<class_num;c++){
		std::map<int,std::vector<size_t> >::const_iterator pp = word_hit_num.find(class_index_inv[c]);
		assert(word_hit_num.end()!=pp);
		for(size_t w=0;w<wordNum();w++){
			whole_word_hit_num[c] += pp->second[w];
		}
		total_entry_num += whole_word_hit_num[c];
	}
	std::vector<double> class_sample_weight(class_num,0.0);
	for(size_t c=0;c<class_num;c++){
		class_sample_weight[c] = static_cast<double>(whole_word_hit_num[c])/total_entry_num;
	}



	// do for each sample
	for(int i = 0; i < features.rows; i++){

		std::vector<std::vector<std::list<size_t> > > matched_occurrences(
				class_num,
				std::vector<std::list<size_t> >(wordNum()));

		std::vector<size_t> num_match(class_num,0);

		// do for each visual word
		for(int w = 0; w < _vocaburary.cols;w++){
			float sim = similarities.at<float>(i,w);
			if(sim < _entry_threshold) continue;
			cv::Point2f fp2center = getRelativeLocation(
					location,
					feature_points[i],
					scale);
			std::list<size_t> near_candidates = occurrences[w].getIndex(fp2center,kernel_size);
			std::list<size_t>::const_iterator pocc;
			for(pocc = near_candidates.begin();pocc!= near_candidates.end();pocc++){
				float dist = distance(fp2center,occurrences[w][*pocc].pt());
				if(dist > kernel_size) continue;
				int class_response = occurrences[w][*pocc].class_response();
				int ci = class_index[class_response];
				num_match[ci]++;
				matched_occurrences[ci][w].push_back(*pocc);
			}
		}

		std::list<size_t>::iterator pmo;
		for(size_t ci=0;ci<class_num;ci++){
			float match_weight = 1.f / num_match[ci];
			int class_response = class_index_inv[ci];
			pvote = response_likelihood.find(class_response);
			if(response_likelihood.end()==pvote){
				return false;
			}
			for(size_t w=0; w<wordNum(); w++){
				float occ_weight = class_sample_weight[ci] / word_hit_num.find(class_response)->second[w];
				for(pmo = matched_occurrences[ci][w].begin();
					pmo != matched_occurrences[ci][w].end();pmo++){
					const ISMEntry* occurrence = &occurrences[w][*pmo];
					assert(class_response == occurrence->class_response());
					float weight = match_weight * occ_weight;

					pvote->second += weight;
					
					if(visualize){
						pimg = voters->find(class_response);
						if(voters->end()==pimg){
							(*voters)[class_response] = cv::Mat::zeros(size,CV_8UC1);
							pimg = voters->find(class_response);
						}
						int radius = std::max(1,static_cast<int>((weight*20) / match_weight));
						cv::circle(pimg->second,feature_points[i].pt,radius,cv::Scalar(255));
						cv::line(pimg->second,feature_points[i].pt,feature_points[i].pt+occurrence->pt(),cv::Scalar(127));
					}
				}
			}
		}
	}

	return true;
}

bool ImplicitShapeModel::read(const std::string& filename){
	release();
	std::ifstream fin;
	fin.open(filename.c_str(),std::ios::binary);
	if(!fin){
		return false;
	}
	if(!read_header(fin)) return false;
	if(!read_entries(fin)) return false;
	fin.close();
	return true;
}

bool ImplicitShapeModel::read_header(const std::string& filename){
	std::ifstream fin;
	fin.open(filename.c_str(),std::ios::binary);
	if(!fin){
		return false;
	}
	if(!read_header(fin)) return false;
	fin.close();
	return true;
}

bool ImplicitShapeModel::read_entries(const std::string& filename){
	std::ifstream fin;
	fin.open(filename.c_str(),std::ios::binary);
	if(!fin){
		return false;
	}
	if(!read_entries(fin)) return false;
	fin.close();
	return true;
}
bool ImplicitShapeModel::read_header(std::istream& in){
	int word_num;
	int feature_dim;
	in.read((char*)&word_num,sizeof(int));
	in.read((char*)&feature_dim,sizeof(int));
	cv::Mat vtemp(cv::Size(word_num,feature_dim),CV_32FC1);
	for(int i=0;i<feature_dim;i++){
		in.read((char*)vtemp.ptr<float>(i),word_num*sizeof(float));
	}
	vocaburary(vtemp.t());
	float buf[6];
	in.read((char*)buf,6*sizeof(float));
	_entry_threshold = buf[0]; _hypothesis_threshold = buf[1]; _kapper1 = buf[2];
	_kapper2 = buf[3]; _object_kernel_ratio = buf[4]; _std_size = buf[5];
	int temp;
	in.read((char*)&temp,sizeof(int));
	_feature_type = (FeatureType)temp;
	return false;
}

bool ImplicitShapeModel::read_entries(std::istream& in){
	if(wordNum()==0) return false;
	std::map<int,std::vector<size_t> >::iterator psn;
	for(size_t w = 0; w < wordNum(); w++){
		size_t num;
		in.read((char*)&num,sizeof(size_t));
//		std::cerr << num << " entries for word " << w << std::endl;
		for(size_t n=0;n<num;n++){
//			std::cerr << n << "/" << num << std::endl;
			ISMEntry entry;
			entry.read(in);
//			std::cerr << entry.class_response() << std::endl;
//			std::cerr << entry.pt().x << ", " << entry.pt().y << ": " << entry.class_response() << std::endl;
			occurrences[w].insert(entry);

			psn = word_hit_num.find(entry.class_response());
			if(word_hit_num.end()==psn){
				word_hit_num[entry.class_response()] = std::vector<size_t>(wordNum(),0);
				psn = word_hit_num.find(entry.class_response());
			}
			psn->second[w]++;

		}
	}
	return true;
}

bool ImplicitShapeModel::write(const std::string& filename)const{
	std::ofstream fout;
	fout.open(filename.c_str(),std::ios::binary);
	if(!fout){
		return false;
	}
	if(!write_header(fout)) return false;
	if(!write_entries(fout)) return false;
	fout.close();
	return true;
}

bool ImplicitShapeModel::write_header(const std::string& filename)const{
	if(!hasVocaburary) return false;
	std::ofstream fout;
	fout.open(filename.c_str(),std::ios::binary);
	if(!fout){
		return false;
	}
	if(!write_header(fout)) return false;
	fout.close();
	return true;
}

bool ImplicitShapeModel::write_entries(const std::string& filename)const{
	std::ofstream fout;
	fout.open(filename.c_str(),std::ios::binary);
	if(!fout){
		return false;
	}
	if(!write_entries(fout)) return false;
	fout.close();
	return true;
}

bool ImplicitShapeModel::write_header(std::ostream& out)const{
	if(!hasVocaburary) return false;
	out.write((char*)&_vocaburary.cols,sizeof(int));
	out.write((char*)&_vocaburary.rows,sizeof(int));
	for(int i=0;i<_vocaburary.rows;i++){
		out.write((char*)_vocaburary.ptr<float>(i),_vocaburary.cols*sizeof(float));
	}
	float buf[6];
	buf[0] = _entry_threshold; buf[1] = _hypothesis_threshold; buf[2] = _kapper1;
	buf[3] = _kapper2; buf[4] = _object_kernel_ratio; buf[5] = _std_size;
	out.write((char*)buf,6*sizeof(float));
	int temp = (int)_feature_type;
	out.write((char*)&temp,sizeof(int));
	return true;
}

bool ImplicitShapeModel::write_entries(std::ostream& out)const{
	bool hasData = false;
	for(size_t w=0;w<wordNum();w++){
		size_t num = occurrences[w].size();
		out.write((char*)&num,sizeof(size_t));
		std::vector<ISMEntry>::const_iterator pp;
		for(pp=occurrences[w].begin();pp!=occurrences[w].end();pp++){
			pp->write(out);
		}
		if(!occurrences[w].empty()){
			hasData = true;
		}
	}
	return hasData;
}


void ImplicitShapeModel::MDLVerification(){
	// not implemented
}
