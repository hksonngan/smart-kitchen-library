#include "skl.h"
#include "sklcv.h"

opt_on(std::string, input_file,"","-i","<FILE>","sample.csv");
opt_on(std::string, output_file,"","-o","<FILE>","sample.csv");

opt_on(size_t,sample_num,4,"-n","<UINT>","number of samples synthesized.");
opt_on(size_t,sample_dim,6,"-d","<UINT>","number of feature dimensions synthesized.");

size_t class_num = 2;// synthesized sample set always has only two classes.


void synthesize(cv::Mat& samples,cv::Mat& responces,cv::Mat& likelihoods,std::vector<skl::Time>& timestamps);

int main(int argc,char* argv[]){
	skl::OptParser options;
	std::vector<std::string> args;
	opt_parse(options,argc,argv,args);

	if(options.help()){
		options.usage();
		return EXIT_FAILURE;
	}

	cv::Mat samples,responces,likelihoods;
	std::vector<skl::Time> timestamps;
	if(!input_file.empty()){
		if(!skl::SampleSetReader::read(input_file,&samples,&responces,&likelihoods,&timestamps)){
			std::cerr << "ERROR: failed to read a sample set data from '" << input_file << "'." << std::endl;
		}
	}
	else{
		synthesize(samples,responces,likelihoods,timestamps);
	}


	if(!output_file.empty()){
		if(!skl::SampleSetWriter::write(output_file,&samples,&responces,&likelihoods,&timestamps)){
			std::cerr << "ERROR: failed to write the sample set data into '" << output_file << "'." << std::endl;
		}
	}
	else{
		skl::SampleSetWriter::write(std::cout,&samples,&responces,&likelihoods,&timestamps);
		std::cerr << std::endl;
	}

	return EXIT_SUCCESS;
}

void synthesize(cv::Mat& samples,cv::Mat& responces,cv::Mat& likelihoods,std::vector<skl::Time>& timestamps){
	samples = cv::Mat::zeros(cv::Size(sample_dim,sample_num),CV_32FC1);
	responces = cv::Mat::zeros(cv::Size(1,sample_num),CV_32SC1);
	likelihoods = cv::Mat::zeros(cv::Size(class_num,sample_num),CV_32FC1);
	timestamps.resize(sample_num);

	for(size_t i=0;i<sample_num;i++){
		float likelihood = 0.f;
		if(i%2==0){
			responces.at<int>(1,i) = 1;
			for(size_t d=0;d<sample_dim;d++){
				float feature = 20.f - (12.f * (float)(rand())/RAND_MAX);
				samples.at<float>(i,d) = feature;
				likelihood+= feature/20;
			}
			likelihood /= sample_dim;
			likelihoods.at<float>(i,0) = likelihood;
			likelihoods.at<float>(i,1) = 1.f - likelihood;
		}
		else{
			for(size_t d=0;d<sample_dim;d++){
				float feature = 12.f * (float)(rand())/RAND_MAX;
				samples.at<float>(i,d) = feature;
				likelihood+= feature/20;
			}
			likelihood /= sample_dim;
			likelihoods.at<float>(i,1) = likelihood;
			likelihoods.at<float>(i,0) = 1.f - likelihood;
		}
		timestamps[i] = skl::Time::now();
	}
}
