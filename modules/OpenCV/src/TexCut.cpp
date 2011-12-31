#include "TexCut.h"
#include "TexCut_parallel.h"
#include <iostream>

using namespace skl;
TexCut::TexCut():g(NULL){
}

TexCut::TexCut(const cv::Mat& bg1, const cv::Mat& bg2, float alpha, float smoothing_term_weight,float thresh_tex_diff,unsigned char over_exposure_thresh,unsigned char under_exposure_thresh):g(NULL){
	setBackground(bg1);
	learnImageNoiseModel(bg2);
	setParams(alpha, smoothing_term_weight,thresh_tex_diff,over_exposure_thresh,under_exposure_thresh);
#ifdef DEBUG
	cv::namedWindow("texcut_data_term",0);
	cv::namedWindow("texcut_tex_int",0);
	cv::namedWindow("texcut_gradient_heterogenuity",0);
	cv::namedWindow("texcut_smoothing_term_x",0);
	cv::namedWindow("texcut_smoothing_term_y",0);
#endif
}

TexCut::~TexCut(){
#ifdef DEBUG
	cv::destroyWindow("texcut_data_term");
	cv::destroyWindow("texcut_tex_int");
	cv::destroyWindow("texcut_gradient_heterogenuity");
	cv::destroyWindow("texcut_smoothing_term_x");
	cv::destroyWindow("texcut_smoothing_term_y");
#endif

}

void TexCut::setParams(float alpha, float smoothing_term_weight, float thresh_tex_diff, unsigned char over_exposure_thresh, unsigned char under_exposure_thresh){
	this->alpha = alpha;
	this->smoothing_term_weight = smoothing_term_weight;
	this->thresh_tex_diff = thresh_tex_diff;
	this->over_exposure_thresh = over_exposure_thresh;
	this->under_exposure_thresh = under_exposure_thresh;
}

int TexCut::compute(const cv::Mat& _src,const cv::Mat& mask,cv::Mat& dest){
	return compute(_src,dest);
}

int TexCut::compute(const cv::Mat& _src,cv::Mat& dest){
	// compute edge capacity and construct graph model
	std::vector<cv::Mat> src;
	if(_src.channels() == 3){
		cv::split(_src, src);
	}
	else if(_src.channels() == 1){
		src.push_back(_src);
	}
	else{
		bool isValidImageChannel = false;
		assert(isValidImageChannel);
	}
	assert(!bg_img.empty());
	assert(!nodes.empty());

	std::vector<cv::Mat> sobel_x,sobel_y;
	getSobel(src,&sobel_x,&sobel_y);

	cv::Mat data_term,smoothing_term_x,smoothing_term_y;
	calcEdgeCapacity(
			src,sobel_x,sobel_y,
			bg_img,bg_sobel_x,bg_sobel_y,
			noise_std_dev,
			gh_expectation,
			gh_std_dev,
			alpha,
			smoothing_term_weight,
			thresh_tex_diff,
			data_term,
			smoothing_term_x,
			smoothing_term_y);

	int flow = calcGraphCut(data_term,smoothing_term_x,smoothing_term_y);

	setResult(_src,dest);
	return flow;
}

void TexCut::setBackground(const cv::Mat& bg){
	if(bg.channels()==3){
		cv::split(bg,this->bg_img);
	}
	else{
		this->bg_img.push_back(bg);
	}


	getSobel(bg_img,&bg_sobel_x,&bg_sobel_y);

	int graph_width = bg.cols / TEXCUT_BLOCK_SIZE;
	int graph_height = bg.rows / TEXCUT_BLOCK_SIZE;
	nodes.assign(
			graph_height, std::vector<TexCutGraph::node_id>(graph_width,0));
}

void TexCut::learnImageNoiseModel(const cv::Mat& bg2){
	assert(!bg_img.empty());
	size_t channels = bg_img.size();
	std::vector<cv::Mat> bg_img2;
	if(bg2.channels()==3){
		cv::split(bg2,bg_img2);
	}
	else{
		bg_img2.push_back(bg2);
	}


	assert(bg_img.size()==bg_img2.size());
	assert(bg_img[0].rows == bg_img2[0].rows);
	assert(bg_img[0].cols == bg_img2[0].cols);
	assert(bg_img[0].type() == bg_img2[0].type());

	noise_std_dev.assign(channels,0);
	gh_expectation.assign(channels,0);
	gh_std_dev.assign(channels,0);
	cv::parallel_for(
			cv::BlockedRange(0,channels),
			ParallelNoiseEstimate(
				&bg_img,
				&bg_img2,
				&noise_std_dev,
				&gh_expectation,
				&gh_std_dev
				)
			);

// debug
#ifdef DEBUG
	std::cerr << "Noise SD: ";
	for(size_t i=0;i<channels;i++){
		if(i!=0) std::cerr << ",";
		std::cerr << noise_std_dev[i];
	}
	std::cerr << std::endl;

	std::cerr << "GH Expec: ";
	for(size_t i=0;i<channels;i++){
		if(i!=0) std::cerr << ",";
		std::cerr << gh_expectation[i];
	}
	std::cerr << std::endl;

	std::cerr << "GH SD   : ";
	for(size_t i=0;i<channels;i++){
		if(i!=0) std::cerr << ",";
		std::cerr << gh_std_dev[i];
	}
	std::cerr << std::endl;
#endif //DEBUG

}

void TexCut::getSobel(
		const std::vector<cv::Mat>& img,
		std::vector<cv::Mat>* sobel_x,
		std::vector<cv::Mat>* sobel_y){
	size_t channels = img.size();
	sobel_x->resize(channels);
	sobel_y->resize(channels);
	for(size_t c=0;c<channels;c++){
		cv::Mat temp = cv::Mat::zeros(img[c].rows,img[c].cols,CV_16SC1);
		sobel_x->at(c) = cv::Mat::zeros(img[c].rows,img[c].cols,CV_8UC1);
		sobel_y->at(c) = sobel_x->at(c).clone();

		CvMat bg = img[c];
		CvMat stemp = temp;

		cvSobel(&bg,&stemp,1,0,3);
		CvMat sobel = sobel_x->at(c);
		cvConvertScaleAbs(&stemp,&sobel,1,0);

		cvSobel(&bg,&stemp,0,1,3);
		sobel = sobel_y->at(c);
		cvConvertScaleAbs(&stemp,&sobel,1,0);

/*
		cv::imshow("bgsobelx",bg_sobel_x[c]);
		cv::imshow("bgsobely",bg_sobel_y[c]);
		cv::waitKey(-1);
*/
	}
}

void TexCut::setResult(const cv::Mat& src,cv::Mat& dest)const{
	int graph_width = src.cols/TEXCUT_BLOCK_SIZE;
	int graph_height = src.rows/TEXCUT_BLOCK_SIZE;
	dest = cv::Mat::zeros(graph_height,graph_width,CV_8UC1);
	for(int y=0;y<dest.rows;y++){
		for(int x=0;x<dest.cols;x++){
			if(g->what_segment(nodes[y][x],TexCutGraph::SOURCE)){
				dest.at<unsigned char>(y,x) = 255;
			}
		}
	}
}

void TexCut::calcEdgeCapacity(
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
		cv::Mat& smoothing_term_y){

	// * smoothing_term_weight
	int channels(src.size());
	int graph_width = src[0].cols / TEXCUT_BLOCK_SIZE;
	int graph_height = src[0].rows / TEXCUT_BLOCK_SIZE;
	int graph_size = graph_width * graph_height;

	data_term = cv::Mat::zeros(graph_height,graph_width,CV_32SC1);
	smoothing_term_x = cv::Mat::zeros(graph_height,graph_width-1,CV_32SC1);
	smoothing_term_y = cv::Mat::zeros(graph_height-1,graph_width,CV_32SC1);
#ifdef DEBUG
	cv::Mat tex_int = cv::Mat::zeros(graph_height,graph_width,CV_32FC1);
#endif
	cv::Mat gradient_heterogenuity = cv::Mat::zeros(graph_height,graph_width,CV_32FC1);

	cv::parallel_for(
			cv::BlockedRange(0,graph_size),
			ParallelCalcEdgeCapacity(
				src,sobel_x,sobel_y,
				bg_img,bg_sobel_x,bg_sobel_y,
				noise_std_dev,
				gh_expectation,
				gh_std_dev,
				alpha,
				thresh_tex_diff,
				over_exposure_thresh,
				under_exposure_thresh,
				data_term,
#ifdef DEBUG
				tex_int,
#endif
				gradient_heterogenuity,
				smoothing_term_x,
				smoothing_term_y
				)
			);

	cv::parallel_for(
			cv::BlockedRange(0,graph_size),
			ParallelAddGradientHeterogenuity(
					data_term,
					gradient_heterogenuity,
					smoothing_term_x,
					smoothing_term_y
				)
			);

#ifdef DEBUG
	cv::imshow("texcut_data_term",data_term);
	cv::imshow("texcut_smoothing_term_x",smoothing_term_x);
	cv::imshow("texcut_smoothing_term_y",smoothing_term_y);
	cv::imshow("texcut_tex_int",tex_int);
	cv::imshow("texcut_gradient_heterogenuity",gradient_heterogenuity);
//	cv::waitKey(-1);
#endif


}

int TexCut::calcGraphCut(const cv::Mat& data_term,const cv::Mat& smoothing_term_x, const cv::Mat& smoothing_term_y){
	if(g!=NULL){
		delete g;
	}
	size_t graph_width = nodes[0].size();
	size_t graph_height = nodes.size();
	size_t graph_size = graph_width * graph_height;
	g = new TexCutGraph(graph_size,graph_size * 2 - graph_width - graph_height );
	for(size_t x=0;x<graph_width;x++){
		nodes[0][x] = g->add_node();
	}
	
	for(size_t y=0;y<graph_height-1;y++){
		for(size_t x=0;x<graph_width;x++){
			nodes[y+1][x] = g->add_node();
			setCapacity(g,nodes,x,y,
					data_term,smoothing_term_x,smoothing_term_y);
		}
	}
	for(size_t x=0;x<graph_width;x++){
		setCapacity(g,nodes,x,graph_height-1,
				data_term,smoothing_term_x,smoothing_term_y);
	}

	return g->maxflow();
}

void TexCut::setCapacity(
		TexCutGraph* g,std::vector<std::vector<TexCutGraph::node_id> >& nodes, size_t x,size_t y, const cv::Mat& data_term,const cv::Mat& smoothing_term_x, const cv::Mat& smoothing_term_y){
	int _data_term = data_term.at<int>(y,x);
	g->add_tweights(
			nodes[y][x],
			QUANTIZATION_LEVEL - _data_term,
			_data_term);
	if(x!=nodes[0].size()-1){
		int _smoothing_term_x = static_cast<int>(smoothing_term_weight * smoothing_term_x.at<int>(y,x));
		if(_smoothing_term_x < 0){
			std::cerr << smoothing_term_x.at<int>(y,x) << std::endl;
			std::cerr << smoothing_term_weight << std::endl;
			std::cerr << _smoothing_term_x << std::endl;
			std::cerr << QUANTIZATION_LEVEL << std::endl;
		}
//		std::cerr << x << ", " << y << ": " << _smoothing_term_x << std::endl;
		g->add_edge(
				nodes[y][x],
				nodes[y][x+1],//QUANTIZATION_LEVEL*2, QUANTIZATION_LEVEL*2);
				_smoothing_term_x,
				_smoothing_term_x);
	}
	if(y!=nodes.size()-1){
		int _smoothing_term_y = static_cast<int>(smoothing_term_weight * smoothing_term_y.at<int>(y,x));
		g->add_edge(
				nodes[y][x],
				nodes[y+1][x],//QUANTIZATION_LEVEL*2,QUANTIZATION_LEVEL*2);
				_smoothing_term_y,
				_smoothing_term_y);
	}
}

void TexCut::updateBackgroundModel(const cv::Mat& bg, const cv::Mat& mask){
	std::vector<cv::Mat> new_bg;
	cv::split(bg,new_bg);
	for(size_t c=0;c<bg_img.size();c++){
		blending<unsigned char,float>(bg_img[c],new_bg[c],mask,bg_img[c]);
	}
}
