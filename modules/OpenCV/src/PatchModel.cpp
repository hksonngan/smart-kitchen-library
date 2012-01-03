/*!
 * @file PatchModel.cpp
 * @author a_hasimoto
 * @date Last Change:2011/Nov/01.
 */

#include "PatchModel.h"
#include <fstream>
#include <iomanip>
using namespace skl;

PatchModel::PatchModel():layer(&patches),max_id(0){}
PatchModel::~PatchModel(){}

PatchModel::PatchModel(const cv::Mat& base_bg):layer(&patches),max_id(0){
	base(base_bg);
}

/*
 * @brief 繝後Μ繧ュ繝上う髴悶・訒懊し繝サ繝・・繝阪キ。「、ス、繝上え繝シ縲√・繝√お縲√ル繝サ繝サ繧オ繝サ繝・・繝阪ケ、 * */
void PatchModel::base(const cv::Mat& __bg){
	this->_base = __bg.clone();
	_newest_bg = __bg.clone();
	hidden_image = cv::Mat::zeros(__bg.size(),CV_8UC3);
	hidden_mask = cv::Mat::zeros(__bg.size(),CV_32FC1);

	patches.clear();
	changed_bg = false;
	changed_fg = false;
}


/*
 * @brief 繧ィ繝シ繝√お縲√ヲ繧ス繧ュ縲√く縲√ム・チE・繝√Eッ丈スサ辯ュ蝙「繝サ * */
size_t PatchModel::putPatch(const cv::Mat& mask, const cv::Mat& newest_image){

/*	// 縲√・縲√さ縲√・縲ゅ、ウ、mask繝帙・繝シ髫阪Ε繧ッ繧ケ繧ウ繧懊・繝ッ繝九・繝・ョサラ、・・ム・チE・繝√繝後繝上縲√繧ゥ繧ヲ繝帙レ繧。縲√こ縲√・	std::vector<size_t> matched_ids;
	Image mask(_mask);
	bool hasRest = true;
std::cerr << __FILE__ << ": " << __LINE__ << std::endl;

	for(size_t i=0;i<trackers.size();i++){
		if(!trackers[i].findTrackingTarget(mask,newest_image,newest_bg,&matched_ids,&mask)){
std::cerr << __FILE__ << ": " << __LINE__ << std::endl;
			hasRest = false;
			break;
		}
	}
std::cerr << __FILE__ << ": " << __LINE__ << std::endl;
	// 縲∫ュ・Υ繝・Α繧「蝙「繝サ繧ァ繝・・縲√Ε縲√ワ縲√トノタラエ・訒ア繝サ繝サ	for(size_t i=0;i<trackers.size();i++){
		if(!trackers[i].hasTargets()){
			trackers.erase(trackers.begin()+i);
			i--;
		}
	}
	std::cerr << "num of tracker: " << trackers.size() << std::endl;

	for(size_t i=0;i<matched_ids.size();i++){
		isMoving[matched_ids[i]] = false;
		layer.push(matched_ids[i]);
		// 繝後Μ繧ュ繝上う髴悶・魄・エェ髱エ蝙「繝サ		// (繝輔シチEメ繝サ
	}
std::cerr << __FILE__ << ": " << __LINE__ << std::endl;

	if(!hasRest){
		// 繧オ繝医テ、ソmaskホ繝シ髫阪Ε縲√ワ縲√縲√ワ縲・ャ倥繧ス繧ュ縲√た縲√ワ繝サ繝璢繝・・繝√ナミマソ、キ、ハ、、Estd::cerr << __FILE__ << ": " << __LINE__ << std::endl;
		return UINT_MAX;
	}
*/

	size_t ID = max_id;
	std::vector<cv::Point> points;
	getObjectSamplePoints(mask, &points);
	patches[ID] = Patch(mask,newest_image,_newest_bg,points);
	std::map<size_t,Patch>::iterator ppatch = patches.find(ID);
	if( ppatch->second.edge_count() == 0){
		patches.erase(ppatch);
		return UINT_MAX;
	}

	// スナ、ハ、テ、ソ・ム・チE・繝√縺Evisivility mask縲・ョ・エェ髱エ蝙「	// 繝代ャ繝√・荳贋ク矩未菫ゅｒ譖エ譁ー縺僞	layer.push(ID);
	std::vector<size_t> lower_patches = layer.getAllBeneathPatch(ID,Patch::original);
	for(size_t i=0;i<lower_patches.size();i++){
		patches[lower_patches[i]].setCoveredState(
				ppatch->second.roi(Patch::original),
				ppatch->second.mask(Patch::original),
				true);
	}


	put_list.push_back(ID);

	max_id++;
	isMoving.resize(max_id,false);
	changed_fg = true;
	return ID;
}



/*
 * @brief 繝サ繝璢繝・・繝√Eャ溽噫螯翫・繧ゥ縲・ョ琺嚶縺E繝サ繝・ * */
void PatchModel::takePatch(size_t ID,std::vector<size_t>* taken_patch_ids){
	std::map<size_t,Patch>::iterator ppatch;
	ppatch = patches.find(ID);
	assert(patches.end() != ppatch);

	// 繧ィ險キ繝偵す髫阪・繝サ繧ゥ縲√・繧ス繝サ繝チE・繝√繝上Eョィ繧ァ繝帙し
	if(isMoving[ppatch->first]) return;
	isMoving[ppatch->first] = true;

	// 繧キ繧ゥ繝上Ε縲・嚶繝サ繧キ縲√ヲ縲√縲√・繧ァ繝・・縲√・visibility縲・カッrue縺ォ謌サ縺・	
	std::vector<size_t> lower_patches = layer.getAllBeneathPatch(ID,Patch::original);
	for(size_t i=0;i<lower_patches.size();i++){
		patches[lower_patches[i]].setCoveredState(
				ppatch->second.roi(Patch::original),
				ppatch->second.mask(Patch::original),
				false);
	}

	// ID繧医陞溘ヲ縲√、ム・チE・繝√Eョエ闌イ謌ソ闌イ繝サ驥・繝サ	
	size_t upper_patch_id;
	while(UINT_MAX != (upper_patch_id = layer.getUpperPatch(ID,Patch::original))){
		takePatch(upper_patch_id,taken_patch_ids);
	}

	if(taken_patch_ids!=NULL){
		taken_patch_ids->push_back(ID);
	}

	// 繧キ繧ゥ繝上Ε縲√・繧ッ陞滓ヵ繝帙う髴悶・魄・エェ逡ヲ蜒ょソ倩争繝ァ
	cv::Mat _hidden = cv::Mat(hidden_image, ppatch->second.roi(Patch::dilate));
	cv::Mat _hidden_mask = cv::Mat(hidden_mask, ppatch->second.roi(Patch::dilate));
	// 繧ィ險キ繝檀idden_image縲√Ε繝サ繧オ繝サ繝・・繝阪オ、繝九縲√縺E繝帙・髫阪ヲ繝・Α縲√く縲√ル縲ゅE	// hidden_image縲√ロppatch縲√レ繧ォ繝イ繝医ヵ繝寂劼笙「笆ウ繝サ繝サ驕峨・繝サ繝ィ繝サ繝サ魴仙ヲ企オ仙凍豕ェ螂ス魄醍。コ繝サ	
	cv::Mat common_mask = ppatch->second.mask(Patch::dilate).clone();
	cv::Mat temp = common_mask.clone();
	cvThreshold(&CvMat(_hidden_mask),&CvMat(temp),0.0,1.0,CV_THRESH_BINARY);
	common_mask -= cv::Scalar(1.0);
	common_mask *= temp;
	
//	cvShowImage("hoge",common_mask.getIplImage());

	blending<cv::Vec3b,float>(
			_hidden,
			ppatch->second.background(Patch::dilate),
			common_mask,
			_hidden);
//	cvShowImage("huga",hidden_image.getIplImage());

	// 繧キ繧ゥ繝上Ε縲√・繧ッ陞滓ヵ繝帙う髴悶・繝偵ヤ繝溘ケ、繧ア繧ス繧ュ繝帙ぅ縲・ョエ轢・
	cvMax(
			&CvMat(_hidden_mask),
			&CvMat(ppatch->second.mask(Patch::dilate)),
			&CvMat(_hidden_mask)
			);
//	cvShowImage("hage",hidden_mask.getIplImage());
//	cvWaitKey(-1);

	// 閾ェ蛻・・ID繧値ayer縺九ｉ蜿悶繝サ繝・	layer.erase(ID);
	// 繧キ繧ゥ繝上Ε縲√・ID縲・エヂtches_in_hidden縺九ｉ蜿悶繝サ繝・	patches_in_hidden.erase(ID);
	changed_bg = true;
}

/*
 * @brief 繝上Β繧ア繧ア縲・エ・west_bg縺ォ蜿肴丐縺輔○縲√・ * */
void PatchModel::updateNewestBG(){
	// newest_bg縲・ョ・エェ
	if(changed_bg){
		changed_bg = false;
		blending<cv::Vec3b,float>(
				hidden_image,
				_newest_bg,
				hidden_mask,
				_newest_bg);
		hidden_image = cv::Scalar(0);
		hidden_mask = cv::Scalar(0);
	}

	if(!changed_fg) return;
	changed_fg = false;
	for(size_t i=0;i<put_list.size();i++){
		std::map<size_t,Patch>::iterator ppatch;
		ppatch = patches.find(put_list[i]);
		assert(patches.end()!=ppatch);
		cv::Mat __newest_bg = cv::Mat(_newest_bg,ppatch->second.roi(Patch::dilate));
		blending<cv::Vec3b,float>(
				ppatch->second.image(Patch::dilate),
				__newest_bg,
				ppatch->second.mask(Patch::dilate),
				__newest_bg);
	}
	put_list.clear();
}

void PatchModel::newest_bg(const cv::Mat& bg){
	assert(bg.size() == _newest_bg.size());
	assert(bg.type() == _newest_bg.type());
	_newest_bg = bg;
}

const cv::Mat& PatchModel::newest_bg()const{
	return _newest_bg;
}


size_t PatchModel::checkLostPatches(const cv::Mat& mask,const cv::Mat& newest_image,const cv::Mat& newest_bg)const{
	assert(mask.size() == _base.size());
	assert(mask.type() == CV_8UC1);

	// mask縺ォ蛹・性縺輔縺E繝サ繝帙・髫埼ャ溯飴ミ泌・ェ髱エ闡」螻ョ螂ス轣ー笆ウ逋シ縺・匸驥碁エ頴ving_patch_ids縺ォ霑ス蜉	// 匁E性縺輔ニ、、、繧ゥ縲√ヮ縲√Υ縲√か縲√・縲√く縲√Η縲√テチE0.9
	double min_region_rate = REMOVED_OBJECT_MIN_REGION_RATE;
	double thresh_min_score = REMOVED_OBJECT_MIN_SCORE;

	std::vector<cv::Point> points;
	getObjectSamplePoints(mask, &points);
	Patch test_patch(mask,newest_bg,newest_image,points);
	if(test_patch.edge_count() == 0) return UINT_MAX;
	cv::Mat edge = test_patch.edge();
	cvSmooth(&CvMat(edge),&CvMat(edge),CV_BLUR,4,4);
	cvThreshold(&CvMat(edge),&CvMat(edge),0,255,CV_THRESH_BINARY);
	test_patch.edge(edge);

	cv::Rect mrect = test_patch.roi(Patch::original);
	size_t mask_count = points.size();

	for(size_t i=0;i<layer.size();i++){

		std::map<size_t,Patch>::const_iterator ppatch =
			patches.find(layer.getLayer(i,false));
//		std::cerr << "patch id: " << layer.getLayer(i,false) << "/" << layer.size() << std::endl;
		assert(ppatch != patches.end());

		// if tracking moving patch, this works. Otherwise, meaningless.
		if(isMoving[ppatch->first]) continue;

		cv::Rect patch_rect = ppatch->second.roi(Patch::original);
		cv::Rect common_rect = patch_rect & mrect;
		// トケハ鮠手ヲ・セー險弱＞隕・渊繝サ繝溘・繧ア繝サ繝・繝サ繝・・繝ゥ
		if(common_rect.width <= 0 || common_rect.height <= 0) continue;


//ppatch->second.save("temp.png",Patch::original,mask.getWidth(),mask.getHeight());
//Image temp("temp.png");

		// 繝サ繝チE・繝√フフタム(ナタ、繧ス繝サ
		size_t patch_area = ppatch->second.points().size();
		size_t count = 0;
		for(size_t i=0;i<patch_area;i++){
			cv::Point pt = ppatch->second.points()[i];
//			cvCircle(temp.getIplImage(),pt,3,CV_RGB(255,0,0));
			if(0 == mask.at<unsigned char>(pt.y,pt.x)) continue;
			count++;
		}

		// 繝城、樊ウ呎ー励・繝九、、、キ、ニ、、、繧ゥ縲√ヮ縲√Υ縲√か縲√・繝サ繝√・繧。繝サ繝・・繝・
		double recall_rate = static_cast<double>(count)/patch_area;
		double precision_rate = static_cast<double>(count)/mask_count;
		if(recall_rate < min_region_rate
				&& precision_rate < min_region_rate) continue;


		// 繝シ繝サ繝ゥ繝翫Ν縲√・繝サ繝√・繧。繝サ繝・・繝・
		double score = calcCommonEdge(common_rect,test_patch,ppatch->second);
		if(score > thresh_min_score){
			return ppatch->first;
		}
	}

	// No vanished objects are found.
	return UINT_MAX;
}

bool PatchModel::isThere(size_t ID, const cv::Mat& newest_image)const{
	std::map<size_t,Patch>::const_iterator ppatch;
	ppatch = patches.find(ID);
	assert(patches.end() != ppatch);
	cv::Mat mask = cv::Mat::zeros(newest_image.size(),CV_8UC1);
	cv::Mat mask_roi = cv::Mat(mask,ppatch->second.roi(Patch::original));
	ppatch->second.mask(Patch::original).convertTo(mask_roi,CV_8U,255);

	std::vector<cv::Point> points;
	getObjectSamplePoints(mask, &points);
	Patch test_patch(mask,newest_image,_base,points);
	if(test_patch.edge_count() == 0) return false;

	cv::Mat edge = test_patch.edge();
	cvSmooth(&CvMat(edge),&CvMat(edge),CV_BLUR,4,4);
	cvThreshold(&CvMat(edge),&CvMat(edge),0,255,CV_THRESH_BINARY);
	test_patch.edge(edge);

	double score = calcCommonEdge(test_patch.roi(Patch::original), test_patch, ppatch->second);
//	std::cerr << "isThere: score for " << ID << " = " << score << std::endl;
	if(score < IS_THERE_OBJECT_MIN_RECALL_RATE){
		return false;
	}
	return true;
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

	// 繝サ繝チE・繝√縺E繧サ螻ャ鬯壺無阨輔・繧ア繝サ繝ヲ繝サ縲ゅ・縲√・繝サ魄題争
	std::map<size_t,Patch>::const_iterator ppatch;
	for(ppatch = patches.begin();
			ppatch != patches.end();ppatch++){
		ss.str("");
		ss << file_head << "_p" << std::setw(2) << std::setfill('0') << ppatch->first;
		ppatch->second.save(ss.str()+"original"+ext,Patch::original);
		ppatch->second.save(ss.str()+"dilate"+ext,Patch::dilate);
		fout << ppatch->first << ": ";
		if(isMoving[ppatch->first]){
			fout << "Not on BasePatch" << std::endl;
		}
		else{
			fout << layer.getOrder(ppatch->first) << "th patch" << std::endl;
		}
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


void PatchModel::getHiddenPatches(const cv::Mat& mask,std::vector<size_t>* newly_hidden_patch_ids,std::vector<size_t>* reappeared_patch_ids){
	assert(1 == mask.channels());
	assert(NULL!=newly_hidden_patch_ids);
	assert(NULL!=reappeared_patch_ids);
	assert(mask.size() == _newest_bg.size());


	// mask縺ョ荳ュ縺ョ4x4縺ョ繝也、弱け縺ョ謨ー繧偵き繧ヲ繝ウ繝医☆E	size_t mask_count = 0;
	cv::Rect mrect(INT_MAX, INT_MAX, 0, 0);

	size_t mask_count = 0;
	for(int y = 0; y < mask.rows; y += PATCH_MODEL_BLOCK_SIZE){
		for(int x = 0; x < mask.cols; x += PATCH_MODEL_BLOCK_SIZE){
			if(0 == mask.at<unsigned char>(y,x)) continue;
			mrect.x = mrect.x<x?mrect.x:x;
			mrect.y = mrect.y<y?mrect.y:y;
			mrect.width = mrect.width > x?mrect.width:x;
			mrect.height = mrect.height > y?mrect.height:y;
			mask_count++;
		}
	}


	std::vector<size_t> current_hidden_patches;
	for(std::map<size_t,Patch>::const_iterator ppatch = patches.begin();
			ppatch != patches.end(); ppatch++){
//		if(isMoving[ppatch->first]) continue;

		CvRect patch_rect = ppatch->second.roi(Patch::original);
		// 髟キ譁ケ蠖「縺碁㍾縺ェ縺」縺ヲ縺・↑縺代縺E繝溘・繧ア繝サ繝・繝サ繝・・繝ゥ
		if(patch_rect.x > mrect.width
				|| patch_rect.y > mrect.height
				|| patch_rect.x + patch_rect.width < mrect.x
				|| patch_rect.y + patch_rect.height < mrect.y) continue;

		// 繝サ繝チE・繝√フフタム(ナタ、繧ス繝サ
		size_t patch_area = ppatch->second.points().size();
		size_t count = 0;
		for(size_t i=0;i<patch_area;i++){
			cv::Point pt = ppatch->second.points()[i];
//			std::cerr << ppatch->first << ": (" << pt.x << ", " << pt.y << ")" << std::endl;
			if(0== mask.at<unsigned char>(pt.y,pt.x))continue;
			count++;
		}

		double recall_rate = static_cast<double>(count)/patch_area;
		double min_recall_rate = HIDDEN_OBJECT_MIN_RECALL_RATE;

		if(recall_rate > min_recall_rate){
			current_hidden_patches.push_back(ppatch->first);
//			std::cerr << ppatch->first << " is now hidden." << std::endl;
		}
	}

	newly_hidden_patch_ids->clear();
	reappeared_patch_ids->clear();
	std::set<size_t>::iterator pp;
	std::set<size_t> temp(patches_in_hidden);
	std::set<size_t> next_in_hidden;
	for(size_t i=0;i<current_hidden_patches.size();i++){
		next_in_hidden.insert(current_hidden_patches[i]);
		pp = temp.find(current_hidden_patches[i]);
		if(pp == temp.end()){
			newly_hidden_patch_ids->push_back(current_hidden_patches[i]);
		}
		else{
			temp.erase(pp);
		}
	}
/*
	std::cerr << "prev_hidden_patches   :";
	for(pp = patches_in_hidden.begin();pp != patches_in_hidden.end();pp++){
		std::cerr << *pp << ", ";
	}
	std::cerr << std::endl;

	std::cerr << "current_hidden_patches:";
	for(size_t i=0;i<current_hidden_patches.size();i++){
		std::cerr << current_hidden_patches[i] << ", ";
	}
	std::cerr << std::endl;
	std::cerr << "newly_hidden_patch    :";
	for(size_t i=0;i<newly_hidden_patch_ids->size();i++){
		std::cerr << newly_hidden_patch_ids->at(i) << ", ";
	}
	std::cerr << std::endl;
*/
	for(pp=temp.begin();pp!=temp.end();pp++){
		reappeared_patch_ids->push_back(*pp);
	}
/*	std::cerr << "reappeared_patch      :";
	for(size_t i=0;i<reappeared_patch_ids->size();i++){
		std::cerr << reappeared_patch_ids->at(i) << ", ";
	}
	std::cerr << std::endl;
*/

	patches_in_hidden = next_in_hidden;
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
