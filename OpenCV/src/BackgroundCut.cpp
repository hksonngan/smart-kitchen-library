/*!
 * @file BackgroundCut.cpp
 * @author a_hasimoto
 * @date Date Created: 2012/Feb/13
 * @date Last Change: 2012/Feb/15.
 */
#include "BackgroundCut.h"
#define QUANTIZATION_LEVEL SHRT_MAX//32767.0f

#ifdef DEBUG
#define DEBUG_BACKGROUND_CUT
#endif
#define BGCUT_SAMPLE_NUM 7680.f /*320*240*/
using namespace skl;

float sqrt2pi = sqrt(2*CV_PI);

float norm(unsigned char val){
	return static_cast<float>(val);
}
float norm(const cv::Vec3b& vec){
	float val = 0;
	for(int c=0;c<3;c++){
		val += std::pow(static_cast<float>(vec[c]),2);
	}
	return sqrt(val);
}
float norm(const cv::Vec3f& vec){
	float val = 0;
	for(int c=0;c<3;c++){
		val += std::pow(vec[c],2);
	}
	return sqrt(val);
}

float gaussian_prob(float val, float mean, float var){
	return exp(-std::pow(val-mean,2)/(2*var))/ (sqrt2pi*sqrt(var));
}

float gaussian_prob(const cv::Vec3b& _val, const cv::Vec3b& _mean, const cv::Vec3f& var){
	float a = 0;
	for(int c=0;c<3;c++){
		float temp =  (float)_val[c] - (float)_mean[c];
		assert(var[c]>0);
		a += temp * temp / var[c];
	}
	a /= 2;
//	std::cerr << sqrt2pi << ", " << cv::norm(var) << ", " << (int)var[0] << ", " << (int)var[1] << ", " << (int)var[2] << std::endl;
	return exp(-a)/(std::pow(sqrt2pi,3)*sqrt(norm(var)));
}
template<class ElemType> float gaussian_prob(const ElemType& val, const CvEMParams& param){
	cv::Mat means(param.means);
	cv::Mat weights(param.weights);
	float prob = 0.f;
	for(int n=0;n<param.nclusters;n++){
		cv::Mat covs(param.covs[n]);
		cv::Vec3b mean;
		cv::Vec3f var;
		for(int c=0;c<3;c++){
			mean[c] = means.at<float>(n,c);
			var[c] = std::max(FLT_MIN,covs.at<float>(c,c));
		}
		prob+=weights.at<float>(0,n) * gaussian_prob(val,mean,var);
	}
	return prob;
}
float gaussian_prob(float val,const CvEMParams& param,int n){
	cv::Mat means(param.means);
	cv::Mat weights(param.weights);
	float prob = 0.f;
	for(int n=0;n<param.nclusters;n++){
		cv::Mat covs(param.covs[n]);
		prob += weights.at<float>(0,n) * gaussian_prob(val,means.at<float>(n,0),covs.at<float>(0,0));
	}
	return prob;
}

/*!
 * @brief デフォルトコンストラクタ
 */
BackgroundCut::BackgroundCut(float thresh_bg,float thresh_fg,float sigma_KL,float K, float sigma_z, float learning_rate,int bg_cluster_num,int fg_cluster_num):graph(NULL){
	model_flag_bg = false;
	model_flag_fg = false;
	setParams(thresh_bg,thresh_fg,sigma_KL,K,sigma_z,learning_rate,bg_cluster_num,fg_cluster_num);
#ifdef DEBUG_BACKGROUND_CUT
	cv::namedWindow("labels",0);
	cv::namedWindow("data_term_fg",0);
	cv::namedWindow("data_term_bg",0);
	cv::namedWindow("hCue",0);
	cv::namedWindow("vCue",0);
#endif
}

/*!
 * @brief デストラクタ
 */
BackgroundCut::~BackgroundCut(){
	if(graph!=NULL){
		delete graph;
	}
}

void BackgroundCut::setParams(float thresh_bg,float thresh_fg, float sigma_KL, float K, float sigma_z, float learning_rate, int bg_cluster_num,int fg_cluster_num){
	this->thresh_bg = thresh_bg;
	this->thresh_fg = thresh_fg;
	this->sigma_KL = sigma_KL;
	this->Ksquared = K*K;
	this->sigma_z = sigma_z;
	this->learning_rate = learning_rate;
	if(bg_cluster_num != bg_global_model.nclusters){
		bg_global_model.nclusters = bg_cluster_num;
		bg_global_model.weights = NULL;
		bg_global_model.means = NULL;
		bg_global_model.covs = NULL;
		bg_global_model.probs = NULL;

		bg_global_model_algo[0].clear();
		bg_global_model_algo[1].clear();
		if(!_background.empty()){
			cv::Mat temp_mask = cv::Mat(_background.size(),CV_8UC1,255);
			if(learnGMM(_background,temp_mask,&bg_global_model_algo[model_flag_bg], &bg_global_model,std::min(1.f, 1.f - BGCUT_SAMPLE_NUM/(_background.cols*_background.rows) ))){
				model_flag_bg = !model_flag_bg;
			}
		}
	}
	if(fg_cluster_num != fg_model.nclusters){
		fg_model.nclusters = fg_cluster_num;
		fg_model.weights = NULL;
		fg_model.means = NULL;
		fg_model.covs = NULL;
		fg_model.probs = NULL;

		fg_model_algo[0].clear();
		fg_model_algo[1].clear();
	}
}

void BackgroundCut::background(const cv::Mat& background){
	background.copyTo(_background);
	nodes.assign(
			background.rows, std::vector<BackgroundCutGraph::node_id>(background.cols,0));
	data_term_fg = cv::Mat(background.size(),CV_32SC1,cv::Scalar(0));
	data_term_bg = cv::Mat(background.size(),CV_32SC1,cv::Scalar(0));
	hCue = cv::Mat(background.size(),CV_32SC1,cv::Scalar(0));
	vCue = cv::Mat(background.size(),CV_32SC1,cv::Scalar(0));
	cv::Scalar default_noise;
	if(background.channels()==1){
		default_noise = cv::Scalar(10.f);
	}
	else{
		default_noise = cv::Scalar(10.f,10.f,10.f);
	}
	noise_variance = cv::Mat(background.size(),CV_MAKETYPE(CV_32F,background.channels()),default_noise);

	// learning background global model
	cv::Mat temp_mask = cv::Mat(background.size(),CV_8UC1,255);
	if(learnGMM(background,temp_mask,&(bg_global_model_algo[model_flag_bg]), &bg_global_model,std::min(1.f, 1.f - BGCUT_SAMPLE_NUM/(background.cols*background.rows) ))){
		model_flag_bg = !model_flag_bg;
	}
}

/*!
 * @brief calcrate background subtraction
 * */
void BackgroundCut::compute(const cv::Mat& src,const cv::Mat& mask,cv::Mat& dest){
	assert(src.size() == _background.size());
	assert(src.type() == _background.type());
	// do rough segmentation
	cv::Mat labels( src.size(), CV_8UC1, cv::Scalar(0));
	// bg->label=0, fg->label=255, unknown->label = 127
	roughSegmentation(src,_background,noise_variance,thresh_bg,thresh_fg,labels);
#ifdef DEBUG_BACKGROUND_CUT
	cv::imshow("labels", labels);
#endif
	if(dest.size()!=src.size()){
		dest = cv::Mat(src.size(),CV_8UC1,cv::Scalar(0));
	}

	// learn fg_model
	if(!learnGMM(src,labels==127, &fg_model_algo[model_flag_fg], &fg_model,std::min(1.f, 1.f - BGCUT_SAMPLE_NUM/(src.cols*src.rows)))){
		// no foreground objects;
		// rough Segmentation was enough;
		return;
	}
	model_flag_fg = !model_flag_fg;

	// calc DataTerm
	calcDataTerm(src,fg_model,bg_global_model,_background,noise_variance, data_term_fg,data_term_bg);
#ifdef DEBUG_BACKGROUND_CUT
	cv::imshow("data_term_fg", data_term_fg);
	cv::imshow("data_term_bg", data_term_bg);
#endif

	// calc SmoothingTerm
	calcSmoothingTerm(src,_background,hCue,vCue);
#ifdef DEBUG_BACKGROUND_CUT
	cv::imshow("hCue", hCue);
	cv::imshow("vCue", vCue);
#endif
	// do graph cut
	if(graph!=NULL) delete graph;
	graph = createGraph(data_term_fg,data_term_bg,hCue,vCue,nodes);
	graph->maxflow();
	for(int y=0;y<src.rows;y++){
		for(int x=0;x<src.cols;x++){
			if(graph->what_segment(nodes[y][x],BackgroundCutGraph::SOURCE)){
				dest.at<unsigned char>(y,x) = 255;
			}
		}
	}
}


bool BackgroundCut::learnGMM(
		const cv::Mat& src,
		const cv::Mat& mask,
		CvEM* algo,
		CvEMParams* params,
		float skip_rate){

	int sample_num = src.rows * src.cols;
	int channels = src.channels();

	assert(src.size()==mask.size());
	assert(mask.type()==CV_8UC1);
	assert(src.depth()==CV_8U);
	assert(channels == 1 || channels == 3);

	cv::Mat samples(sample_num,channels,CV_32FC1);
//	std::cerr << samples.cols << ", " << samples.rows << ", " << sample_num << std::endl;

	int count = 0;
	if(channels == 1){
		for(int y = 0; y < src.rows; y++){
			const unsigned char* psrc = src.ptr<const unsigned char>(y);
			for(int x = 0; x < src.cols; x++){
				if(rand()<skip_rate*RAND_MAX) continue;
				if(mask.at<unsigned char>(y,x)!=255) continue;
				float* psamples = samples.ptr<float>(count);
				psamples[0] = psrc[x];
				count++;
			}
		}
	}
	else{
		for(int y = 0; y < src.rows; y++){
			const cv::Vec3b* psrc = src.ptr<const cv::Vec3b>(y);
			for(int x = 0; x < src.cols; x++){
				if(rand()<skip_rate*RAND_MAX) continue;
				if(mask.at<unsigned char>(y,x)!=255) continue;
				for(int c=0;c<channels;c++){
					samples.at<float>(count,c) = psrc[x][c];
				}
				count++;
			}
		}
	}
	if(count == 0) return false;
	samples.resize(count);

	cv::Mat labels(count,1,CV_32SC1);

	assert(params->nclusters >0);
	params->cov_mat_type = CvEM::COV_MAT_DIAGONAL;
	params->term_crit.max_iter = 10;//0;
	params->term_crit.epsilon = 0.1;//FLT_EPSILON;
	params->term_crit.type = CV_TERMCRIT_ITER | CV_TERMCRIT_EPS;

	CvMat _samples = CvMat(samples);
	CvMat _labels = CvMat(labels);
	std::cerr << "learning GMM sample num: " << count << std::endl;
	algo->clear();
	algo->train(&_samples,0,*params,&_labels);
	params->weights = algo->get_weights();
	params->means = algo->get_means();
	params->covs = algo->get_covs();
	params->probs = NULL;

	std::cerr << "learning end." << std::endl;
	return true;
}

template<class ElemType,class NoiseType> void _roughSegmentation(
		const cv::Mat& src,
		const cv::Mat& bg,
		const cv::Mat& noise_variance,
		float thresh_bg,
		float thresh_fg,
		cv::Mat& labels){
	assert(src.size() == labels.size());
	assert(CV_8UC1 == labels.type());

	float min_prob(FLT_MAX),max_prob(-1);

	for(int y=0;y<src.rows;y++){
		const ElemType* psrc = src.ptr<const ElemType>(y);
		const ElemType* pbg = bg.ptr<const ElemType>(y);
		const NoiseType* pvar = noise_variance.ptr<const NoiseType>(y);
		unsigned char* plabel = labels.ptr<unsigned char>(y);


		for(int x=0;x<src.cols;x++){
			float prob = gaussian_prob(psrc[x],pbg[x],pvar[x]);
//			std::cerr << prob << std::endl;
			min_prob = std::min(min_prob,prob);
			max_prob = std::max(max_prob,prob);
			float temp = norm(pvar[x]);
			float _thresh_bg = gaussian_prob(thresh_bg * sqrt(temp),0,temp);
			float _thresh_fg = gaussian_prob(thresh_fg * sqrt(temp),0,temp);
			if(prob>_thresh_bg){
				plabel[x] = 0;
			}
			else if(prob<_thresh_fg){
				plabel[x] = 255;
			}
			else{
				plabel[x] = 127;
			}
		}
	}
	std::cerr << "max: " << max_prob << std::endl;
	std::cerr << "min: " << min_prob << std::endl;

}
void BackgroundCut::roughSegmentation(
		const cv::Mat& src,
		const cv::Mat& bg,
		const cv::Mat& noise_variance,
		float thresh_bg,
		float thresh_fg,
		cv::Mat& labels){
	assert(src.type()==CV_8UC3 || src.type()==CV_8UC1);
	
	if(src.type() == CV_8UC3){
		_roughSegmentation<cv::Vec3b,cv::Vec3f>(src,bg,noise_variance,thresh_bg,thresh_fg,labels);
	}
	else if(src.type() == CV_8UC1){
		_roughSegmentation<unsigned char,float>(src,bg,noise_variance,thresh_bg,thresh_fg,labels);
	}
}

float calcKLDivergence_(const CvEMParams& param,int n){
	cv::Mat covs(param.covs[n]);
	cv::Mat means(param.means);
	int channels = covs.rows;
	float var_sum = 0;
	float mean_sq_sum = 1;
	for(int c=0;c<channels;c++){
		var_sum += covs.at<float>(c,c);
		mean_sq_sum += std::pow(means.at<float>(n,c),2);
	}
	return -mean_sq_sum/(2.f*var_sum)-log(2.f*CV_PI*var_sum)/2;
}
float calcKLDivergence(const CvEMParams& param_fg,const CvEMParams& param_bg){
	float val = 0;
	assert(param_fg.weights!=NULL);
	cv::Mat fg_weights(param_fg.weights);
	cv::Mat bg_weights(param_bg.weights);
	for(int i=0;i<param_fg.nclusters;i++){
		float min_kl = FLT_MAX;
		float base = calcKLDivergence_(param_fg,i);
		for(int j=0;j<param_bg.nclusters;j++){
			float kl = base-calcKLDivergence_(param_bg,j)+std::log(fg_weights.at<float>(0,i)/bg_weights.at<float>(0,j));
			min_kl = std::min(min_kl,kl);
		}
		val += fg_weights.at<float>(0,i) * min_kl;
	}
	return val;
}

template<class ElemType,class NoiseType> void _calcDataTerm(
		const cv::Mat& src,
		const CvEMParams& fg_model,
		const CvEMParams& bg_global_model,
		const cv::Mat& bg,
		const cv::Mat& noise_variance,
		float sigma_KL,
		cv::Mat& data_term_fg,
		cv::Mat& data_term_bg){
	float kldivergence = calcKLDivergence(fg_model,bg_global_model);
	float alpha = 1.f - exp(-kldivergence/sigma_KL)/2.f;

	for(int y=0;y<src.rows;y++){
		int* pdfg = data_term_fg.ptr<int>(y);
		int* pdbg = data_term_bg.ptr<int>(y);
		const ElemType* psrc = src.ptr<const ElemType>(y);
		const ElemType* pbg = bg.ptr<const ElemType>(y);
		const NoiseType* pvar = noise_variance.ptr<const NoiseType>(y);
		for(int x=0;x<src.cols;x++){
			pdfg[x] = QUANTIZATION_LEVEL * gaussian_prob<ElemType>(psrc[x],fg_model);
			pdbg[x] = QUANTIZATION_LEVEL * (alpha*gaussian_prob<ElemType>(psrc[x],bg_global_model)+(1.f-alpha)*gaussian_prob(psrc[x],pbg[x],pvar[x]));
		}
	}
}
void BackgroundCut::calcDataTerm(
		const cv::Mat& src,
		const CvEMParams& fg_model,
		const CvEMParams& bg_global_model,
		const cv::Mat& bg,
		const cv::Mat& noise_variance,
		cv::Mat& data_term_fg,
		cv::Mat& data_term_bg){
	assert(src.type() == CV_8UC3 || src.type() == CV_8UC1);
	if(src.type()==CV_8UC3){
		_calcDataTerm<cv::Vec3b,cv::Vec3f>(src,fg_model,bg_global_model,bg,noise_variance,sigma_KL,data_term_fg,data_term_bg);
	}
	else{
		_calcDataTerm<unsigned char,float>(src,fg_model,bg_global_model,bg,noise_variance,sigma_KL,data_term_fg,data_term_bg);
	}
}



template<class ElemType> int calcSmoothingTerm_(
		const ElemType& src1,
		const ElemType& src2,
		float Ksquared,
		float sigma_z,
		const ElemType& bg1,
		const ElemType& bg2){
	ElemType vec1 = src1-bg1;
	ElemType vec2 = src2-bg2;
	float z = norm(vec1-vec2);
	return ( norm(src1-src2)/(1+( norm(bg1-bg2)/Ksquared )*exp(-(z*z)/sigma_z) ) ) * QUANTIZATION_LEVEL;
}
template<> int calcSmoothingTerm_(
		const cv::Vec3b& src1,
		const cv::Vec3b& src2,
		float Ksquared,
		float sigma_z,
		const cv::Vec3b& bg1,
		const cv::Vec3b& bg2){
	cv::Vec3f vec1,vec2,src_diff,bg_diff;
	for(int c=0;c<3;c++){
		vec1[c] = (float)src1[c]-(float)bg1[c];
		vec2[c] = (float)src2[c]-(float)bg2[c];
		src_diff[c] = (float)src1[c]-(float)src2[c];
		bg_diff[c] = (float)bg1[c]-(float)bg2[c];
	}
	float z = norm(vec1-vec2);

	return ( norm(src_diff)/(1+( norm(bg_diff)/Ksquared )*exp(-(z*z)/sigma_z) ) ) * QUANTIZATION_LEVEL;
}


template<class ElemType> void calcSmoothingTerm_(
		const cv::Mat& src,
		const cv::Mat& bg,
		float Ksquared,
		float sigma_z,
		cv::Mat& hCue,
		cv::Mat& vCue){
	const ElemType* psrc,*psrc_b,*pbg,*pbg_b;
	int* phCue,*pvCue;
	for(int y=0;y<src.rows;y++){
		psrc = src.ptr<const ElemType>(y);
		pbg = bg.ptr<const ElemType>(y);
		if(y+1<src.rows){
			psrc_b = src.ptr<const ElemType>(y+1);
			pbg_b = bg.ptr<const ElemType>(y+1);
		}
		phCue = hCue.ptr<int>(y);
		pvCue = vCue.ptr<int>(y);

		for(int x=0;x<src.cols;x++){
			if(x+1<src.cols){
				phCue[x] = calcSmoothingTerm_(psrc[x],psrc[x+1],Ksquared,sigma_z,pbg[x],pbg[x+1]);
			}
			if(y+1<src.rows){
				pvCue[x] = calcSmoothingTerm_(psrc[x],psrc_b[x],Ksquared,sigma_z,pbg[x],pbg_b[x]);
			}
		}
	}
}

void BackgroundCut::calcSmoothingTerm(
		const cv::Mat& src,
		const cv::Mat& bg,
		cv::Mat& hCue,
		cv::Mat& vCue){
	if(src.channels()==3){
		calcSmoothingTerm_<cv::Vec3b>(src,bg,Ksquared,sigma_z,hCue,vCue);
	}
	else{
		calcSmoothingTerm_<unsigned char>(src,bg,Ksquared,sigma_z,hCue,vCue);
	}
}

BackgroundCut::BackgroundCutGraph* BackgroundCut::createGraph(
		const cv::Mat& data_term_fg,
		const cv::Mat& data_term_bg,
		const cv::Mat& hCue,
		const cv::Mat& vCue,
		std::vector<std::vector<BackgroundCutGraph::node_id> >& nodes){
	cv::Size graph_size(data_term_fg.size());
	int node_num = data_term_fg.rows * data_term_fg.cols;
	BackgroundCutGraph* graph = new BackgroundCutGraph(node_num,node_num * 2 - graph_size.width - graph_size.height);
	for(int x = 0; x < graph_size.width; x++){
		nodes[0][x] = graph->add_node();
	}

	for(int y = 0; y < graph_size.height-1; y++){
		const int* pdfg = data_term_fg.ptr<const int>(y);
		const int* pdbg = data_term_bg.ptr<const int>(y);
		const int* phCue = hCue.ptr<const int>(y);
		const int* pvCue = vCue.ptr<const int>(y);
		for(int x = 0; x < graph_size.width; x++){
			nodes[y+1][x] = graph->add_node();
			graph->add_tweights(
					nodes[y][x],
					pdbg[x],
					pdfg[x]);
			if(x+1<graph_size.width){
				graph->add_edge(
						nodes[y][x],
						nodes[y][x+1],
						phCue[x]/2.f,
						phCue[x]/2.f);
			}
			graph->add_edge(
					nodes[y][x],
					nodes[y+1][x],
					pvCue[x]/2.f,
					pvCue[x]/2.f);
		}
	}
	const int* phCue = hCue.ptr<const int>(graph_size.height-1);
	const int* pdfg = data_term_fg.ptr<const int>(graph_size.height-1);
	const int* pdbg = data_term_bg.ptr<const int>(graph_size.height-1);
	for(int x = 0; x < graph_size.width-1; x++){
		graph->add_tweights(
				nodes[graph_size.height-1][x],
				pdbg[x],
				pdfg[x]);
		graph->add_edge(
				nodes[graph_size.height-1][x],
				nodes[graph_size.height-1][x+1],
				phCue[x]/2.f,
				phCue[x]/2.f);
	}
	return graph;
}

void BackgroundCut::updateBackgroundModel(const cv::Mat& img){
	
}

