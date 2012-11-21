#define USE_VIDEO_CAPTURE_OPT_PARSER
#include "skl.h"
#include "sklcv.h"



// NOTICE!!!!!!
// This sample program works just as OpenCV's ConDensation sample.
// see http://opencv.jp/sample/estimators.html



opt_on(std::string, camera_setting, "", "-C","<FILE>","load camera conf parameters.");
opt_on(std::string, input_file,"","-i","<FILE>","load video file");

opt_on(std::string, output_file,"","-o","<FILE>","save video");

opt_on(int,dev,0,"-d","<DEVICE_ID>","direct device id.");
opt_on_bool(trackbar,"","create track bar with the window.");


// parameters for tracking
opt_on(size_t,sample_num,5000,"","<UINT>","set particle num");
opt_on(double,std_dev,15,"","<DOUBLE>","set standard deviation of color for tracking.");
int _target_rgb[3] = {127,127,0};
std::vector<int> target_rgb(_target_rgb,_target_rgb+3);
opt_on_container(std::vector,int,target_rgb,"","<R:G:B>","set tracking target color.",":",3);
opt_on(float,rand_max,50,"","<FLOAT>","set maximum randam work distance");

// you need to write your likelihood calculation functor!
// in this sample code, Observation = cv::Mat and LikelihoodCalcFunctor = LikelihoodByColor
// then skl::ParticleFilter<cv::Mat,LikelihoodByColor> is the algorithm class for tracking.
class LikelihoodByColor{
	public:
		// functor must have this set function.
		void set(const cv::Mat observation,float** samples,float* likelihoods){
			_observation = observation;
			_samples = samples;
			_likelihoods = likelihoods;
		}

		void operator()(const cv::BlockedRange& range)const{
			assert(skl::checkMat(_observation,CV_8U,3));
			for(int i=range.begin();i<range.end();i++){
				// State is 2 dimentional, which are x and y coordinate in the image respectively.
				cv::Point2f point(_samples[i][0],_samples[i][1]);
				_likelihoods[i] = 0.f;
				// likelihood is 0 if they are out of the image.
				if(point.x<0 || _observation.cols<=point.x) continue;
				if(point.y<0 || _observation.rows<=point.y) continue;

				cv::Vec3b col = _observation.at<cv::Vec3b>(point.y,point.x);
				_likelihoods[i] = calcNormalProb(col,_target_bgr,_std_dev);
			}
		}
		cv::Mat _observation;
		float** _samples;
		float* _likelihoods;
		double _std_dev;
		cv::Vec3b _target_bgr;

	private:
		// subfunction for operator()
		float calcNormalProb(const cv::Vec3b& col1,const cv::Vec3b& col2,double std_dev)const{
			double diff=0;
			for(size_t c=0;c<3;c++) diff += std::pow((double)col1[c] - (double)col2[c],2);
			float dist = sqrt(diff);
			return 1.0 / (sqrt(2.0*CV_PI) * _std_dev) * expf(-dist * dist / (2.0*std_dev*std_dev));
		}

};

typedef skl::ParticleFilter<cv::Mat,LikelihoodByColor> MyParticleFilter;
typedef skl::ParticleFilter<cv::Mat,LikelihoodByColor>::State MyState;


int main(int argc,char* argv[]){
	skl::OptParser options;
	std::vector<std::string> args;
	opt_parse(options,argc,argv,args);

	if(options.help()){
		options.usage();
		return EXIT_FAILURE;
	}

	skl::VideoCapture cam;
	skl::VideoCaptureParams params;

	if(!camera_setting.empty()){
		params.load(camera_setting);
	}
	// CAUTION: call opt_parse_cap_prop after opt_parse
	opt_parse_cap_prop(params);



	if(input_file.empty()){
		cam.open(dev);
	}
	else{
		cam.open(input_file);
	}

	// CAUTION: set params after open the camera.
	cam.set(params);


	if(!cam.isOpened()){
		std::cerr << "ERROR: failed to open video." << std::endl;
		std::cerr << "       Maybe, camera is not connected to the PC??" << std::endl;
		return EXIT_FAILURE;
	}

	cv::namedWindow("image",0);
	cv::Mat image;

	cv::Vec3b target_bgr;
	for(size_t c=0;c<3;c++){
		target_bgr[c] = target_rgb[2-c];
	}
	LikelihoodByColor tracker_functor;
	tracker_functor._std_dev = std_dev;
	tracker_functor._target_bgr = target_bgr;

	MyParticleFilter tracker(sample_num);
	tracker.functor(&tracker_functor);
	// or skl::ParticleFilter<cv::Mat,LikelihoodByColor> tracker(sample_num,tracker_functor);

	// set parameters for time increment P(X_t|X_{t-1})
	MyState upperBound(4),lowerBound(4);
	cam >> image;
	upperBound.data[0] = image.cols;
	upperBound.data[1] = image.rows;
	upperBound.data[2] = 50; // max moving distance per frame (in pixel) for x direction
	upperBound.data[3] = 50; // for y direction
	lowerBound.data[0] = 0;
	lowerBound.data[1] = 0;
	lowerBound.data[2] = 0; // min moving distance per frame for x direction
	lowerBound.data[3] = 0; // for y direction

	// dynamMat for motion model (linear)
	cv::Mat dynamMat = cv::Mat::eye(cv::Size(4,4),CV_32FC1);
	dynamMat.at<float>(0,2)=1; // motion calc (MyState[2]=velocity_x)
	dynamMat.at<float>(1,3)=1; // motion calc (MyState[3]=velocity_y)
	tracker.initialize(lowerBound,upperBound, dynamMat);
	// tracker.initialize(lowerBound,upperBound); for ignoring v_{x,y} component.

	// set random walk parameters
	for(size_t d=0;d<2;d++){
		tracker.setRandStateParam(d, rand_max,-rand_max,0,CV_RAND_NORMAL);
	}



	while('q'!=cv::waitKey(10)){
		cam >> image;
		if(image.empty()) break;
		tracker.compute(image);

		for(size_t n=0;n<sample_num;n++){
			MyState st = tracker.state(n);
//			float prob = tracker.confidence(n);
			cv::Point2f pt(st.data[0],st.data[1]);
			cv::circle(image,pt,2,CV_RGB(target_rgb[0],target_rgb[1],target_rgb[2]));
		}
		cv::imshow("image",image);
	}
	cv::destroyWindow("image");

	return EXIT_SUCCESS;
}
