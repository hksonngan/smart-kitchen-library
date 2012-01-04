/*!
 * @file PatchModel.cpp
 * @author a_hasimoto
 * @date Last Change:2012/Jan/04.
 */

#include "PatchModel.h"
#include <fstream>
#include <iomanip>
#include <highgui.h>

using namespace skl;

PatchModel::PatchModel():layer(&patches),max_id(0){}
PatchModel::~PatchModel(){}

PatchModel::PatchModel(const cv::Mat& base_bg):layer(&patches),max_id(0){
	base(base_bg);
}

/*
 * @brief 背景画像をセットし、それ以外は全てリセットする
 * */
void PatchModel::base(const cv::Mat& __bg){
	this->_base = __bg.clone();
	_latest_bg = __bg.clone();
	hidden_image = cv::Mat::zeros(__bg.size(),CV_8UC3);
	hidden_mask = cv::Mat::zeros(__bg.size(),CV_32FC1);

	patches.clear();
	changed_bg = false;
	changed_fg = false;
}


/*
 * @brief 完全に新しいパッチを登録する
 * */
size_t PatchModel::putPatch(const cv::Mat& img, const cv::Mat& fg_edge, const cv::Mat& mask, const cv::Rect& roi){

	size_t ID = max_id;
	patches[ID] = Patch(mask, img, _latest_bg, fg_edge, roi);
	std::map<size_t,Patch>::iterator ppatch = patches.find(ID);

	// 重なったパッチのvisivility maskを更新する
	// パッチの上下関係を更新する
	layer.push(ID);
	std::vector<size_t> lower_patches = layer.getAllBeneathPatch(ID,Patch::original);
	for(size_t i=0;i<lower_patches.size();i++){
		patches[lower_patches[i]].setCoveredState(roi,cv::Mat(mask,roi),true);
	}


	put_list.push_back(ID);

	max_id++;
	changed_fg = true;
	return ID;
}


/*
 * @brief パッチをモデルから取り除く
 * */
void PatchModel::takePatch(size_t ID,std::vector<size_t>* taken_patch_ids){
	std::map<size_t,Patch>::iterator ppatch;
	ppatch = patches.find(ID);
	assert(patches.end() != ppatch);

	// 自分より下にある物体のcoveredStateをfalseに戻す
	std::vector<size_t> lower_patches = layer.getAllBeneathPatch(ID,Patch::original);
	for(size_t i=0;i<lower_patches.size();i++){
		patches[lower_patches[i]].setCoveredState(
				ppatch->second.roi(Patch::original),
				ppatch->second.mask(Patch::original),
				false);
	}

	// IDより上にあるパッチを先に取り去る
	size_t upper_patch_id;
	while(UINT_MAX != (upper_patch_id = layer.getUpperPatch(ID,Patch::original))){
		takePatch(upper_patch_id,taken_patch_ids);
	}

	if(taken_patch_ids!=NULL){
		taken_patch_ids->push_back(ID);
	}

	// 自分の後ろの画像を更新用に作成
	cv::Mat hidden_roi = cv::Mat(hidden_image, ppatch->second.roi(Patch::dilate));
	cv::Mat hidden_mask_roi = cv::Mat(hidden_mask, ppatch->second.roi(Patch::dilate));
	// 既にhidden_maskがセットされている領域に対して、
	// ppatchとのブレンディングマスクを作る
	cv::Mat common_mask = ppatch->second.mask(Patch::dilate).clone();
	cv::Mat temp = cv::Mat(hidden_mask_roi.size(), hidden_mask_roi.type());
	cv::threshold(hidden_mask_roi, temp, 0.0, 1.0, CV_THRESH_BINARY);
	common_mask -= cv::Scalar(1.0);
	common_mask *= temp;

	std::cerr << __FILE__ << ": " << __LINE__ << std::endl;
	cv::imshow("debug",common_mask);
	cv::waitKey(-1);

	hidden_roi = blending<cv::Vec3b,float>(
			ppatch->second.background(Patch::dilate),
			hidden_roi,
			common_mask);
	std::cerr << __FILE__ << ": " << __LINE__ << std::endl;
	cv::imshow("debug",hidden_roi);
	cv::waitKey(-1);

	// 自分の後ろの画像に対する更新率を設定
	hidden_mask_roi = cv::max(hidden_mask_roi,ppatch->second.mask(Patch::dilate));

	//  自分のIDをlayerから取り除く
	layer.erase(ID);
	// 自分のIDを_hidden_objects,reappeared_objectsなどから取り除く
	_hidden_objects.remove(ID);
	_newly_hidden_objects.remove(ID);
	_reappeared_objects.remove(ID);

	changed_bg = true;
}

void PatchModel::setObjectLabels(
		const cv::Mat& img,
		const cv::Mat& human,
		const cv::Mat& object_cand_labels,
		size_t object_cand_num,
		std::vector<size_t>* put_object_ids,
		std::vector<size_t>* taken_object_ids){
	assert(NULL != put_object_ids);
	assert(NULL != taken_object_ids);
	put_object_ids->clear();
	taken_object_ids->clear();

	// update hidden state
	std::list<size_t> hidden_patches;
	getHiddenPatches(human,&hidden_patches);
	updateHiddenState(hidden_patches);

	cv::Mat fg_edge;
	cv::Mat bg_edge;
	edge_difference(img,_latest_bg,fg_edge,bg_edge);

	// check whether reappeared objects are still there.
	for(std::list<size_t>::iterator iter = _reappeared_objects.begin();
			iter != _reappeared_objects.end(); iter++){
		if(checkTakenObject(*iter,bg_edge)){
			takePatch(*iter,taken_object_ids);
		}
	}

	std::vector<cv::Mat> cand_masks(object_cand_num,cv::Mat::zeros(object_cand_labels.size(),CV_8UC1));
	std::vector<cv::Rect> cand_rois(object_cand_num,cv::Rect(INT_MAX,INT_MAX,0,0));

	for(int y = 0; y < object_cand_labels.rows; y++){
		for(int x = 0; x < object_cand_labels.cols; x++){
			short label = object_cand_labels.at<short>(y,x);
			if(0==label) continue;
			label--;
			cand_masks[label].at<unsigned char>(y,x) = 255;
			cand_rois[label].x = cand_rois[label].x < x ? cand_rois[label].x : x;
			cand_rois[label].y = cand_rois[label].y < y ? cand_rois[label].y : y;
			cand_rois[label].width = cand_rois[label].width > x ? cand_rois[label].width : x;
			cand_rois[label].height= cand_rois[label].height> y ? cand_rois[label].height: y;
		}
	}
	for(size_t l=0;l < object_cand_num;l++){
		cand_rois[l].width -= cand_rois[l].x - 1;
		cand_rois[l].height-= cand_rois[l].y - 1;
	}

	std::vector<bool> was_taken_object(object_cand_num,false);
	for(size_t l = 0; l < object_cand_num; l++){
		for(int y = cand_rois[l].y;
				y < cand_rois[l].height + cand_rois[l].y; y++){
			for(int x = cand_rois[l].x;
					x < cand_rois[l].width + cand_rois[l].x; x++){
				if(cand_masks[l].at<unsigned char>(y,x)==0) continue;
				if(hidden_mask.at<float>(y,x)==0.0f) continue;
				was_taken_object[l] = true;
				break;
			}
			if(was_taken_object[l]) break;
		}
	}

	// put the rest candidates;
	for(size_t l=0;l<object_cand_num;l++){
		if(was_taken_object[l]) continue;
		std::vector<cv::Point> edge_points;
		for(int y = cand_rois[l].y;
				y < cand_rois[l].height + cand_rois[l].y; y++){
			for(int x = cand_rois[l].x;
					x < cand_rois[l].width + cand_rois[l].x; x++){
				if(fg_edge.at<unsigned char>(y,x)==0) continue;
				edge_points.push_back(cv::Point(x,y));
			}
		}
		if(edge_points.empty()) continue;

		cand_rois[l] = fitRect(edge_points);
		size_t new_id = putPatch(img, fg_edge, cand_masks[l], cand_rois[l]);
		put_object_ids->push_back(new_id);
	}
}

/*
 * @brief 変更をlatest_bgに反映させる
 * */
void PatchModel::update(const cv::Mat& newest_img, const cv::Mat& object_mask, float learning_rate){
	// latest_bgを更新
	cv::Mat _mask = blur_mask(object_mask,PATCH_DILATE);
	cv::Mat mask = cv::Mat::zeros(object_mask.size(), CV_32FC1);
	object_mask.convertTo(mask, CV_32FC1,0,1.0f/255.0f);
	mask = blur_mask(mask,PATCH_DILATE);

	// taken patch に対する更新
	if(changed_bg){
		changed_bg = false;
		blending<cv::Vec3b,float>(
				hidden_image,
				_latest_bg,
				hidden_mask,
				_latest_bg);
		hidden_image = cv::Scalar(0);
		hidden_mask = cv::Scalar(0);
	}

	// put_patch に対する更新
	if(!changed_fg) return;
	changed_fg = false;
	for(size_t i=0;i<put_list.size();i++){
		std::map<size_t,Patch>::iterator ppatch;
		ppatch = patches.find(put_list[i]);
		assert(patches.end()!=ppatch);
		cv::Mat latest_bg_roi = cv::Mat(_latest_bg,ppatch->second.roi(Patch::dilate));
		cv::Mat newest_img_roi = cv::Mat(newest_img,ppatch->second.roi(Patch::dilate));
		latest_bg_roi = blending<cv::Vec3b,float>(
				newest_img_roi,
				latest_bg_roi,
				ppatch->second.mask(Patch::dilate));
	}
	put_list.clear();
}

void PatchModel::latest_bg(const cv::Mat& bg){
	assert(bg.size() == _latest_bg.size());
	assert(bg.type() == _latest_bg.type());
	_latest_bg = bg;
}

const cv::Mat& PatchModel::latest_bg()const{
	return _latest_bg;
}

void PatchModel::save(
		const std::string& file_head,
		const std::string& ext)const{

	std::stringstream ss;
	int width = _base.cols;
	int height = _base.rows;

	std::ofstream fout;
	fout.open((file_head+"_state.txt").c_str());
	assert(fout);

	// パッチの状態をあらわすファイルを作成
	std::map<size_t,Patch>::const_iterator ppatch;
	for(ppatch = patches.begin();
			ppatch != patches.end();ppatch++){
		ss.str("");
		ss << file_head << "_p" << std::setw(2) << std::setfill('0') << ppatch->first;
		ppatch->second.save(ss.str()+"original"+ext,Patch::original);
		ppatch->second.save(ss.str()+"dilate"+ext,Patch::dilate);
		fout << ppatch->first << ": ";
		fout << layer.getOrder(ppatch->first) << "th patch" << std::endl;
	}

	fout.close();
}

const cv::Mat& PatchModel::base()const{
	return _base;
}

const Patch& PatchModel::operator[](size_t ID)const{
	std::map<size_t,Patch>::const_iterator pp;
	pp = patches.find(ID);
	assert(pp!=patches.end());
	return pp->second;
}


void PatchModel::getHiddenPatches(const cv::Mat& mask,std::list<size_t>* hidden_patch_ids){
	assert(1 == mask.channels());
	assert(NULL!=hidden_patch_ids);
	assert(mask.size() == _latest_bg.size());
	hidden_patch_ids->clear();

	// maskの中の4x4のブロックの数をカウントする
	cv::Rect mrect(INT_MAX, INT_MAX, 0, 0);

	for(int y = 0; y < mask.rows; y += PATCH_MODEL_BLOCK_SIZE){
		for(int x = 0; x < mask.cols; x += PATCH_MODEL_BLOCK_SIZE){
			if(0 == mask.at<unsigned char>(y,x)) continue;
			mrect.x = mrect.x < x ? mrect.x:x;
			mrect.y = mrect.y<y?mrect.y:y;
			mrect.width = mrect.width > x?mrect.width:x;
			mrect.height = mrect.height > y?mrect.height:y;
		}
	}
	mrect.width = mrect.x - 1;
	mrect.height = mrect.y - 1;

	std::vector<size_t> current_hidden_patches;
	for(std::map<size_t,Patch>::const_iterator ppatch = patches.begin();
			ppatch != patches.end(); ppatch++){
		// 長方形が重なっていなければスキップ
		if( !( mrect && ppatch->second.roi(Patch::original)) ) continue;

		// パッチの面積(点の数)
		size_t patch_area = ppatch->second.points().size();
		size_t count = 0;
		for(size_t i=0;i<patch_area;i++){
			cv::Point pt = ppatch->second.points()[i];
//			std::cerr << ppatch->first << ": (" << pt.x << ", " << pt.y << ")" << std::endl;
			if(0 == mask.at<unsigned char>(pt.y,pt.x))continue;
			count++;
		}

		double recall_rate = static_cast<double>(count)/patch_area;
		double min_recall_rate = HIDDEN_OBJECT_MIN_RECALL_RATE;

		if(recall_rate > min_recall_rate){
			hidden_patch_ids->push_back(ppatch->first);
		}
	}
}

double PatchModel::calcCommonEdge(
		const CvRect& common_rect,
		const Patch& base,
		const Patch& patch){
//	Image _base_edge(base.getEdgeImage());
//	Image _patch_edge(patch.getEdgeImage());

	cv::Rect crect = common_rect;
	crect.width += crect.x;
	crect.height += crect.y;

	size_t common_edge_count = 0;

	cv::Rect brect = base.roi(Patch::original);
	cv::Rect prect = patch.roi(Patch::original);

	for(int y = crect.y; y < crect.height; y++){
		size_t by = y - brect.y;
		size_t ry = y - prect.y;

		for(int x = crect.x;x < crect.width;x++){
			size_t bx = x - brect.x;
			size_t rx = x - prect.x;
			if(base.edge().at<unsigned char>(by,bx) <= 0) continue;
			if(patch.edge().at<unsigned char>(ry,rx) <= 0) continue;
//			cvCircle(_base_edge.getIplImage(),cvPoint(x,y),3,cvScalar(127,0,0));
//			cvCircle(_patch_edge.getIplImage(),cvPoint(rx,ry),3,cvScalar(127,0,0));
			common_edge_count++;
		}
	}

//	cvShowImage("patch_edge",_patch_edge.getIplImage());
//	cvShowImage("base_edge",_base_edge.getIplImage());
	assert(patch.edge_count()!=0);
	return static_cast<double>(common_edge_count) / patch.edge_count();
}

void PatchModel::getObjectSamplePoints(const cv::Mat& mask, std::vector<cv::Point>* points){
	points->clear();
	for(int y=0;y<mask.rows;y+=PATCH_MODEL_BLOCK_SIZE){
		for(int x=0;x<mask.cols;x+=PATCH_MODEL_BLOCK_SIZE){
			if(0==mask.at<unsigned char>(y,x)) continue;
			points->push_back(cv::Point(x,y));
		}
	}
}

void PatchModel::updateHiddenState(
		std::list<size_t>& __hidden_objects){
	__hidden_objects.sort();
	_newly_hidden_objects.clear();
	std::set_difference(
			__hidden_objects.begin(),__hidden_objects.end(),
			_hidden_objects.begin(),_hidden_objects.end(),
			_newly_hidden_objects.begin());
	std::set_difference(
			_hidden_objects.begin(),_hidden_objects.end(),
			__hidden_objects.begin(),__hidden_objects.end(),
			_reappeared_objects.begin());
	_hidden_objects = __hidden_objects;
}

bool PatchModel::checkTakenObject(size_t id, const cv::Mat& bg_edge)const{
	std::map<size_t,Patch>::const_iterator ppatch = patches.find(id);
	assert(patches.end() != ppatch);
	cv::Rect roi = ppatch->second.roi(Patch::original);

	cv::Mat dick_bg_edge;
	cv::Size kernel_size(PATCH_MODLE_EDGE_DILATE,PATCH_MODLE_EDGE_DILATE);
	cv::blur(cv::Mat(bg_edge,roi),dick_bg_edge,kernel_size);
	cv::threshold(dick_bg_edge,dick_bg_edge,0,255,CV_THRESH_BINARY);

	size_t count_common_edge(0);
	size_t count_original_edge(0);
	for(int y=0;y<roi.height;y++){
		for(int x=0;x<roi.width;x++){
			if(0==ppatch->second.edge().at<unsigned char>(y,x)) continue;
			count_original_edge++;
			if(0==dick_bg_edge.at<unsigned char>(y,x)) continue;
			count_common_edge++;
		}
	}
	double score = static_cast<double>(count_common_edge) / count_original_edge;
	std::cerr << __FILE__ << ": " << __LINE__ << std::endl;
	std::cerr << score << std::endl;
	if(score > TAKEN_OBJECT_EDGE_CORELATION){
		return true;
	}
	return false;
}
