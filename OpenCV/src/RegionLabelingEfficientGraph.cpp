/*!
 * @file RegionLabelingEfficientGraph.cpp
 * @author a_hasimoto
 * @date Date Created: 2012/Oct/07
 * @date Last Change: 2013/Mar/21.
 */
#include "RegionLabelingEfficientGraph.h"
#include "RegionLabelingEfficientGraph/segment-image.h"
#include "sklcvutils.h"

using namespace skl;


/*!
 * @brief デフォルトコンストラクタ
 */
RegionLabelingEfficientGraph::RegionLabelingEfficientGraph(float __sigma,float __k, int __min_size):_sigma(__sigma),_k(__k),_min_size(__min_size){

}

/*!
 * @brief デストラクタ
 */
RegionLabelingEfficientGraph::~RegionLabelingEfficientGraph(){

}

size_t RegionLabelingEfficientGraph::compute(const cv::Mat& col_image, const cv::Mat& mask,cv::Mat& region_labels){
	int width = col_image.cols;
	int height = col_image.rows;
	assert(!col_image.empty());
	assert(CV_8UC3 == col_image.type());

/*	skl::ensureMat(_temp,CV_32F,col_image.channels(), col_image.size());
	col_image.convertTo(_temp,CV_32F);
*/	image<float> *r = new image<float>(width, height);
	image<float> *g = new image<float>(width, height);
	image<float> *b = new image<float>(width, height);
/*
	_single_channels.push_back(cv::Mat(height,width,CV_32FC1,r));
	_single_channels.push_back(cv::Mat(height,width,CV_32FC1,g));
	_single_channels.push_back(cv::Mat(height,width,CV_32FC1,b));

	cv::split(_temp,_single_channels);
*/
	// smooth each color channel  
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			cv::Vec3b col = col_image.at<cv::Vec3b>(y,x);
			imRef(r, x, y) = col[2];
			imRef(g, x, y) = col[1];
			imRef(b, x, y) = col[0];
		}
	}

	image<float> *smooth_r = smooth(r, _sigma);
	image<float> *smooth_g = smooth(g, _sigma);
	image<float> *smooth_b = smooth(b, _sigma);

//	_single_channels.clear();
	delete r;
	delete g;
	delete b;

	// build graph
	edge *edges = new edge[width*height*4];
	int num = 0;
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			if (x < width-1) {
				edges[num].a = y * width + x;
				edges[num].b = y * width + (x+1);
				edges[num].w = diff(smooth_r, smooth_g, smooth_b, x, y, x+1, y);
				num++;
			}

			if (y < height-1) {
				edges[num].a = y * width + x;
				edges[num].b = (y+1) * width + x;
				edges[num].w = diff(smooth_r, smooth_g, smooth_b, x, y, x, y+1);
				num++;
			}

			if ((x < width-1) && (y < height-1)) {
				edges[num].a = y * width + x;
				edges[num].b = (y+1) * width + (x+1);
				edges[num].w = diff(smooth_r, smooth_g, smooth_b, x, y, x+1, y+1);
				num++;
			}

			if ((x < width-1) && (y > 0)) {
				edges[num].a = y * width + x;
				edges[num].b = (y-1) * width + (x+1);
				edges[num].w = diff(smooth_r, smooth_g, smooth_b, x, y, x+1, y-1);
				num++;
			}
		}
	}
	delete smooth_r;
	delete smooth_g;
	delete smooth_b;

	// segment
	universe *u = segment_graph(width*height, num, edges, _k);

	// post process small components
	for (int i = 0; i < num; i++) {
		int a = u->find(edges[i].a);
		int b = u->find(edges[i].b);
		if ((a != b) && ((u->size(a) < _min_size) || (u->size(b) < _min_size)))
			u->join(a, b);
	}
	delete [] edges;
	
	size_t num_ccs;
	num_ccs = u->num_sets();

	if(col_image.size()!=region_labels.size()
		|| CV_16SC1 != region_labels.type()){
		region_labels = cv::Mat(col_image.size(),CV_16SC1);
	}
	if(num_ccs >= SHRT_MAX) return num_ccs;

	short *label_map = new short[width*height];
	memset(label_map,0,sizeof(short)*width*height);
	short count = 1;
	if(!mask.empty()){
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				int comp = u->find(y * width + x);
				if(label_map[comp]==0){
					label_map[comp] = count;
					count++;
				}
				if(mask.at<unsigned char>(y,x)==0){
					region_labels.at<short>(y,x) = 0;
					continue;
				}
				region_labels.at<short>(y,x) = label_map[comp];
			}
		}
	}
	else{
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				int comp = u->find(y * width + x);
				if(label_map[comp] == 0){
					label_map[comp] = count;
					count++;
				}
				region_labels.at<short>(y,x) = label_map[comp];
			}
		}
	}
	assert(count-1 == (short)num_ccs);
	delete label_map;

	delete u;

	return num_ccs;

}
