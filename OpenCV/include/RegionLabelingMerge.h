/*!
 * @file RegionLabelingMerge.h
 * @author a_hasimoto
 * @date Date Created: 2013/Mar/21
 * @date Last Change:2013/Mar/22.
 */
#ifndef __SKL_REGION_LABELING_MERGE_H__
#define __SKL_REGION_LABELING_MERGE_H__

#include "sklcvutils.h"

namespace skl{

/*!
 * @class RegionLabelingMerge
 * @brief 二つのラベル画像をマージする
 */
template<class INPUT=short,int cv_type_input=CV_16S, class OUTPUT=int, int cv_type_output=CV_32S>
class RegionLabelingMerge{
	public:
		typedef std::pair<size_t,size_t> tie;

	public:
		RegionLabelingMerge();
		virtual ~RegionLabelingMerge();
		OUTPUT compute(
				size_t region_num1,
				const cv::Mat& src1,
				size_t region_num2,
				const cv::Mat& src2,
				cv::Mat& dst);
	protected:
		cv::Mat _local_index;

		std::vector<std::set<int> > checkConnection(const std::set<tie>& ties, size_t max_id)const;
	private:


};

template<class INPUT,int cv_type_input, class OUTPUT, int cv_type_output>
RegionLabelingMerge<INPUT,cv_type_input,OUTPUT,cv_type_output>::RegionLabelingMerge(){
}
template<class INPUT,int cv_type_input, class OUTPUT, int cv_type_output>
RegionLabelingMerge<INPUT,cv_type_input,OUTPUT,cv_type_output>::~RegionLabelingMerge(){
}

template<class INPUT,int cv_type_input, class OUTPUT, int cv_type_output>
std::vector<std::set<int> > RegionLabelingMerge<INPUT,cv_type_input,OUTPUT,cv_type_output>::checkConnection(const std::set<tie>& ties,size_t max_id)const{

	std::vector<std::set<int> > components;

	std::vector<bool> isTied(max_id,false);

	for(std::set<tie>::const_iterator iter = ties.begin();
			iter != ties.end(); iter++){
		size_t index1 = iter->first;
		size_t index2 = iter->second;
		isTied[index1] = true;
		isTied[index2] = true;

		int belonging_component1(-1);
		int belonging_component2(-1);

		for(size_t i=0;i<components.size();i++){
			if(components[i].end()!=components[i].find(index1)){
				assert(belonging_component1==-1);
				belonging_component1 = i;
			}
			else if(components[i].end()!=components[i].find(index2)){
				assert(belonging_component2==-1);
				belonging_component2 = i;
			}
		}
		if(belonging_component1==-1
			&&belonging_component2==-1){
			std::set<int> new_component;
			new_component.insert(index1);
			new_component.insert(index2);
			components.push_back(new_component);
		}
		else if(belonging_component1==-1){
			components[belonging_component2].insert(index1);
		}
		else if(belonging_component2==-1){
			components[belonging_component1].insert(index2);
		}
		else{
			// merge two components
			components[belonging_component1].insert(
					components[belonging_component2].begin(),
					components[belonging_component2].end());
			components.erase(components.begin()+belonging_component2);
		}
	}

	for(size_t id=0;id<max_id;id++){
		if(isTied[id]) continue;
		std::set<int> new_component;
		new_component.insert(id);
		components.push_back(new_component);
	}

	return components;
}

// DEBUG用のコード
template<class HOGE>
cv::Vec3b rand_color(){
	cv::Vec3b col;
	for(int c=0;c<3;c++){
		col[c] = rand() % UCHAR_MAX;
	}
	return col;
}


template<class LABEL>
cv::Mat printRegion(const cv::Mat& labels, const cv::Mat& local_index, int target_label){
	srand(time(NULL));
	skl::checkMat(local_index,CV_32S,1,labels.size());
	cv::Mat canvas(labels.size(),CV_8UC3,cv::Scalar(0));

	std::vector<cv::Vec3b> colors;
	for(int y=0;y<labels.rows;y++){
		const LABEL* pLabel = labels.ptr<LABEL>(y);
		const unsigned int* pIndex = local_index.ptr<unsigned int>(y);
		cv::Vec3b* pDst = canvas.ptr<cv::Vec3b>(y);
		for(int x=0;x<labels.cols;x++){
			if(pLabel[x]-1!=target_label)continue;

			unsigned int index = pIndex[x];
			// make a new color
			if(colors.size()<=index){
				int prev_size = colors.size();
				colors.resize(index+1);
				for(size_t i=prev_size;i<colors.size();i++){
					colors[i] = rand_color<unsigned char>();
				}
			}

			pDst[x] = colors[index];
		}
	}

	return canvas;
}

template<class INPUT,int cv_type_input, class OUTPUT, int cv_type_output>
OUTPUT RegionLabelingMerge<INPUT,cv_type_input,OUTPUT,cv_type_output>::compute(
		size_t region_num1,
		const cv::Mat& src1,
		size_t region_num2,
		const cv::Mat& src2,
		cv::Mat& dst){

	assert(skl::checkMat(src1,cv_type_input,1));
	assert(src1.cols>0 && src1.rows>0);
	assert(skl::checkMat(src2,cv_type_input,1,src1.size()));
	skl::ensureMat(dst,cv_type_output,1,src1.size());
	skl::ensureMat(_local_index,CV_32S,1,src1.size());


	std::vector<std::set<tie> > ties;
	std::vector<int> max_local_ids;
	std::vector<std::map<size_t,size_t> > index(region_num1);
	OUTPUT merged_region_num(0);

	unsigned int local_id;


	for(int y=0;y<src1.rows;y++){
		const INPUT* pLabel1 = src1.ptr<INPUT>(y);
		const INPUT* pLabel2 = src2.ptr<INPUT>(y);
		OUTPUT* pMerged = dst.ptr<OUTPUT>(y);
		unsigned int* pLocalIndex = _local_index.ptr<unsigned int>(y);

		OUTPUT* pMerged_up;
		unsigned int* pLocalIndex_up;
		if(y>0){
			pMerged_up = dst.ptr<OUTPUT>(y-1);
			pLocalIndex_up = _local_index.ptr<unsigned int>(y-1);
		}

		for(int x=0;x<src1.cols;x++){
			INPUT label1 = pLabel1[x] - 1;
			INPUT label2 = pLabel2[x] - 1;
			if(label1<0 || label2<0){
				pMerged[x] = 0;
				continue;
			}
			std::map<size_t,size_t>::iterator pIdx = index[label1].find(label2);
			if(index[label1].end()==pIdx){
				index[label1][label2] = merged_region_num;
				pIdx = index[label1].find(label2);

				merged_region_num++;
				ties.resize(merged_region_num);
				max_local_ids.resize(merged_region_num,0);
			}
			OUTPUT mlabel = pIdx->second;
			pMerged[x] = mlabel+1;

			local_id = max_local_ids[mlabel];
			bool hasUp = false;
			if(y>0 && pMerged[x] == pMerged_up[x]){
				local_id = pLocalIndex_up[x];
				hasUp = true;
			}
			if(x>0 && pMerged[x] == pMerged[x-1]){
				if(!hasUp){
					local_id = pLocalIndex[x-1];
				}
				else if(local_id!=(unsigned int)pLocalIndex[x-1]){
					ties[mlabel].insert(tie(local_id,pLocalIndex[x-1]));
				}
			}
			else if(!hasUp){
				max_local_ids[mlabel]++;
			}

			pLocalIndex[x] = local_id;
		}
	}

	// post process (check ties for disconnected components.)
	size_t pre_merged_region_num = merged_region_num;
	std::vector<std::map<int,size_t> > splitting_index(pre_merged_region_num);


	for(size_t i=0;i<pre_merged_region_num;i++){
		if(ties[i].empty()) continue;
		std::vector<std::set<int> > 
			connected_component = checkConnection(ties[i],max_local_ids[i]);
		assert(connected_component.size()>0);

//		std::cerr << i << ": " << connected_component.size() <<  std::endl;
//		cv::imshow("label", printRegion<OUTPUT>(dst,_local_index,i-1));
//		cv::waitKey(500);


		for(size_t n=1;n<connected_component.size();n++){
			for(std::set<int>::iterator it=connected_component[n].begin();
					it != connected_component[n].end(); it++){
				splitting_index[i][*it] = merged_region_num;
//				std::cerr << merged_region_num << std::endl;
			}
			merged_region_num++;
		}
	}

	for(int y=0;y<src1.rows;y++){
		OUTPUT* pMerged = dst.ptr<OUTPUT>(y);
		unsigned int* pLocalIndex = _local_index.ptr<unsigned int>(y);
		for(int x=0;x<src1.cols;x++){
			OUTPUT label = pMerged[x]-1;
			if(splitting_index[label].empty()) continue;
			std::map<int,size_t>::iterator pMap;
			pMap = splitting_index[label].find(pLocalIndex[x]);
			if(splitting_index[label].end() == pMap) continue;
			pMerged[x] = pMap->second+1;

		}
	}

	return merged_region_num;
}

} // skl

#endif // __SKL_REGION_LABELING_MERGE_H__

