/*!
 * @file RegionLabelingEfficientGraph.h
 * @author a_hasimoto
 * @date Date Created: 2012/Oct/07
 * @date Last Change:2012/Oct/22.
 */
#ifndef __SKL_REGION_LABELING_EFFICIENT_GRAPH_H__
#define __SKL_REGION_LABELING_EFFICIENT_GRAPH_H__


#include "FilterMat2Mat.h"


namespace skl{

/*!
 * @class "Efficient Graph-Based Image Segmentation"に基づく領域分割
 * @note "Efficient Graph-Based Image Segmentation," Pedro F. Felzenszwalb and Daniel P. Huttenlocher, International Journal of Computer Vision, 59(2) September 2004.
 * @example ../samples/Segmentation/sample_segmentation_efficient_graph.cpp
 */
 class RegionLabelingEfficientGraph: public FilterMat2Mat<size_t>{

	public:
		RegionLabelingEfficientGraph(float sigma = 0.5, float k = 500, int min_size=30);
		virtual ~RegionLabelingEfficientGraph();
		size_t compute(const cv::Mat& col_image, const cv::Mat& mask,cv::Mat& segment_labels);
		inline size_t compute(const cv::Mat& col_image, cv::Mat& segment_labels){
			return compute(col_image,cv::Mat(),segment_labels);
		};
		
		inline float sigma()const{return _sigma;}
		inline void sigma(float __sigma){_sigma = __sigma;}
		inline float k()const{return _k;}
		inline void k(float __k){_k = __k;}
		inline int min_size()const{return _min_size;}
		inline void min_size(int __min_size){_min_size = __min_size;}
	protected:
		float _sigma;
		float _k;
		int _min_size;
	private:
		
};

} // skl

#endif // __SKL_REGION_LABELING_EFFICIENT_GRAPH_H__

