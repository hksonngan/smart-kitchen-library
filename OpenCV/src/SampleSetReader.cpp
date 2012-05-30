/*!
 * @file SampleSetReader.cpp
 * @author a_hasimoto
 * @date Date Created: 2012/May/30
 * @date Last Change: 2012/May/30.
 */
#include "SampleSetReader.h"

using namespace skl;


/*!
 * @brief デフォルトコンストラクタ
 */
SampleSetReader::SampleSetReader(){

}

/*!
 * @brief デストラクタ
 */
SampleSetReader::~SampleSetReader(){

}

bool SampleSetReader::read(const std::string& filename, cv::Mat* samples, cv::Mat* responces, cv::Mat* likelihoods,std::vector<skl::Time>* timestamps){
	std::ifstream fin;
	fin.open(filename.c_str());
	if(!fin) return false;
	bool isSuccess = read(fin,samples,responces,likelihoods,timestamps);
	fin.close();
	return isSuccess;
}

bool SampleSetReader::read(std::istream& in, cv::Mat* samples, cv::Mat* responces, cv::Mat* likelihoods,std::vector<skl::Time>* timestamps){

	size_t sample_num;
	size_t sample_dim;
	size_t class_num;
	bool has_responce;
	bool has_timestamp;

	if(!_readHeader(in,sample_num,sample_dim,has_responce,class_num,has_timestamp)) return false;

	// read samplels
	if(samples!=NULL){
		if(!_readSamples(in, sample_num, sample_dim, *samples)) return false;
	}
	else{
		skip(in,sample_dim);
	}

	// read responces
	if(responces!=NULL && has_responce){
		if(responces!=NULL){
			if(!_readResponces(in, sample_num, *responces)) return false;
		}
		else{
			skip(in,1);
		}
	}

	// read likelihoods
	if(likelihoods!=NULL && class_num>0){
		if(!_readLikelihoods(in,sample_num,class_num, *likelihoods)) return false;
	}
	else{
		skip(in,class_num);
	}

	// read timestamps
	if(timestamps!=NULL && has_timestamp){
		if(!_readTimestamps(in,sample_num,*timestamps)) return false;
	}
	else{
		skip(in,1);
	}
	return true;
}




bool SampleSetReader::skip(std::istream& in,size_t num){
	std::string buf;
	for(size_t i=0;i<num;i++){
		if(in.eof()) return false;
		std::getline(in,buf);
	}
	return true;
}

bool SampleSetReader::_readHeader(std::istream& in, size_t& sample_num, size_t& sample_dim, bool& has_responce, size_t& class_num, bool& has_timestamp){
	std::string str;
	if(in.eof()) return false;
	std::getline(in,str);
	std::vector<std::string> buf;
	buf = skl::split_strip(str,",",5);
	if(buf.size()!=5) return false;
	sample_num = atoi(buf[0].c_str());
	sample_dim = atoi(buf[1].c_str());
	has_responce = atoi(buf[2].c_str())>0;
	class_num = atoi(buf[3].c_str());
	has_timestamp = atoi(buf[4].c_str())>0;
	return true;
}

bool SampleSetReader::_readTimestamps(std::istream& in, size_t sample_num, std::vector<skl::Time>& dist){
	std::string str;
	if(in.eof()) return false;
	std::getline(in,str);
	std::vector<std::string> buf = skl::split_strip(str,",",sample_num);
	if(buf.size()!=sample_num) return false;

	dist.resize(sample_num);
	for(size_t i=0;i<sample_num;i++){
		dist[i].parseString(buf[i]);
	}
	return true;
}


