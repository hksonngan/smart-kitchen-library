/*!
 * @file ObjectManager.cpp
 * @author a_hasimoto
 * @date Last Change:2011/Sep/12.
 * */

#include "ObjectManager.h"
#include <sstream>

using namespace mmpl;
using namespace mmpl::image;

ObjectManager::ObjectManager():max_id(0){}

ObjectManager::~ObjectManager(){}


bool ObjectManager::updateHiddenStatus(const Image& human_mask,double thresh){
	iterator obj;
	if(_getHidden.size() != size()){
		_getHidden.resize(size(),false);
		_getRevealed.resize(size(),false);
	}

	size_t idx = 0;
	for(obj = begin(); obj != end(); obj++,idx++){
		size_t count = 0;
		std::vector<CvPoint2D32f>::iterator point;
		for(point = obj->getPoints().begin(); point != obj->getPoints().end(); point++){
			CvPoint pt = cvPointFrom32f(*point);
			size_t offset = pt.x + pt.y * human_mask.getIplImage()->widthStep;
			if(0 != (unsigned char)(human_mask.getIplImage()->imageData[offset])){
				count++;
			}
		}

		obj->prob_hidden = count == obj->getPoints().empty() ? 0 : (double)count/obj->getPoints().size();
//		std::cerr << obj->prob_hidden << std::endl;
		if(obj->prob_hidden > thresh){
			if(!obj->isHidden){
				_getHidden[idx] = true;
				obj->timestamp_isHidden = timestamp;
			}
			else{
				_getHidden[idx] = false;
			}
			obj->isHidden = true;
			_getRevealed[idx] = false;
		}
		else{
			if(obj->isHidden){
				_getRevealed[idx] = true;
			}
			else{
				_getRevealed[idx] = false;
			}
			obj->isHidden = false;
			_getHidden[idx] = false;
		}
	}
	return true;
}

bool ObjectManager::updateMovedStatus(const Image& object_mask,double thresh){
	iterator obj;
	_getMoved.assign(size(),false);
	_getStoped.assign(size(),false);

	size_t idx = 0;
	for(obj = begin(); obj != end(); obj++,idx++){
		size_t count = 0;
		std::vector<CvPoint2D32f>::iterator point;
		for(point = obj->getPoints().begin(); point != obj->getPoints().end(); point++){
			CvPoint pt = cvPointFrom32f(*point);
			size_t offset = pt.x + pt.y * object_mask.getIplImage()->widthStep;
			if(0 != (unsigned char)(object_mask.getIplImage()->imageData[offset])){
				count++;
			}
		}
		if(count > thresh * obj->getPoints().size()){
			if(_isMoved[idx]){
				_getMoved[idx] = false;
			}
			else{
				_getMoved[idx] = true;
			}
			_isMoved[idx] = true;
			_getStoped[idx] = false;
		}
		else{
			if(_isMoved[idx]){
				_getStoped[idx] = true;
			}
			else{
				_getStoped[idx] = false;
			}
			_isMoved[idx] = false;
			_getMoved[idx] = false;
		}
	}
	return true;
}

bool ObjectManager::updateStaticStatus(const Image& whole_mask,double threshold){
	iterator obj;
	_isStatic.assign(size(),true);

	size_t idx = 0;
	for(obj = begin(); obj != end(); obj++,idx++){
		// $B0\F0Cf(B?
		if(this->isMoved(idx)){
			_isStatic[idx] = false;
			continue;
		}

		// $B<WJCCf(B?
		if(this->isHidden(idx)){
			_isStatic[idx] = false;
			continue;
		}

		// $BGD;}Cf(B?
		if(obj->getProbGrabbed() > 1.0-threshold){
			_isStatic[idx] = false;
			continue;
		}

		// $B0\F0Cf(B?
		std::vector<CvPoint2D32f>::iterator point;
		for(point = obj->getPoints().begin(); point != obj->getPoints().end(); point++){
			CvPoint pt = cvPointFrom32f(*point);
			size_t offset = pt.x + pt.y * whole_mask.getIplImage()->widthStep;
			if(0 != (unsigned char)(whole_mask.getIplImage()->imageData[offset])){
				_isStatic[idx] = false;
				break;
			}
		}
	}
	return true;
}

/*
 * @brief $B>r7o$,(Btrue$B$H$J$C$F$$$kJ*BN$NFCD'E@$H%$%s%G%C%/%9$rJV$9(B
 * @param condition $BFCD'E@$r$H$k$+$I$&$+7hDj$9$k%U%i%0(B
 * @param points $BFCD'E@$N3JG<@h(B
 * @param point_indice $B8e$GJ*BNKh$rJ,3d$9$k:]$K;H$&%$%s%G%C%/%9(B
 * */
std::vector<std::vector<CvPoint2D32f> > ObjectManager::getObjectPointsIf(const std::vector<bool>& condition)const{
	assert(condition.size()==size());

	std::vector<std::vector<CvPoint2D32f> > point_vectors(size());

	size_t idx = 0;
	ObjectManager::const_iterator pp;
	for(pp = begin(); pp!=end(); pp++,idx++){
		if(condition[idx]){
			point_vectors[idx] = pp->getPoints();
		}
	}
	return point_vectors;
}

/*
 * @brief $BJ*BN$N0LCV$dGD;}$K4X$9$k>uBV$r99?7$9$k(B
 * */
size_t ObjectManager::updateObjectStatus(
		const FlowManager& flow_manager,
		const std::vector<std::vector<CvPoint2D32f> >& _obj_candidates,
		double threshold_point_distance,
		size_t threshold_min_point_num,
		double threshold_grab,
		double threshold_release){
#ifdef DEBUG
	std::cerr << "Not implemented updateObjectStatus." << std::endl;
#endif // DEBUG

	_getGrabbed.assign(size(),false);

	// $BA07JJ*BN$,$J$K$b$J$$$J$iJ*BN$N0LCV!"GD;}$5$l$?$+$I$&$+!"(B
	// $B?75,J*BN$N=P8=$J$I$N>uBV$OJQ$o$i$J$$(B
	if(_obj_candidates.empty()){
		return 0;
	}

	// const $B$r=q$-49$($i$l$k$h$&$K%3%T!<$r$9$k(B
	std::vector<std::vector<CvPoint2D32f> > obj_candidates = _obj_candidates;

	// $BJ*BN$N0\F07k2L$H!"$=$NCf?4!"5Z$S6k7A$r5a$a$k(B
	updateObjectLocation(flow_manager,&obj_candidates,threshold_point_distance);


	// $B$b$7==J,$J?t$NFCD'E@$r;}$DNN0h$,;D$C$FF~$l$P!"?7$?$JJ*BN$H$9$k(B
	size_t count_new_obj = updateNewObjects(obj_candidates, threshold_min_point_num);

	// $B4{$KJ*BN$,GD;}$5$l$F$$$k$+!"5Z$SGD;}8e$K$J$/$J$C$F$$$k$+%A%'%C%/$7!"$J$1$l$P>C5n$9$k(B
	size_t idx = 0;
	for(iterator obj=begin();obj!=end();){
		std::cerr << "grab judge: " << threshold_grab << "?" << obj->getProbGrabbed() << std::endl;
		if(threshold_grab > obj->getProbGrabbed()){
			// $BGD;}$5$l$F$$$J$$(B
			obj++;
			idx++;
			continue;
		}
		if(obj->getProbInHand() < threshold_release){
			// $B4{$KGD;}8e$KCV$+$l$F$$$k(B
			obj = this->erase(obj);
			_getHidden.erase(_getHidden.begin()+idx);
			_getRevealed.erase(_getRevealed.begin()+idx);
			_isMoved.erase(_isMoved.begin()+idx);
			_getMoved.erase(_getMoved.begin()+idx);
			_getStoped.erase(_getStoped.begin()+idx);
			_isStatic.erase(_isStatic.begin()+idx);
			_getGrabbed.resize(size());
		}
		else{
			// $BGD;}$5$l$F$$$k$,!"CV$+$l$F$$$J$$(B($BGD;}Cf(B)
			if(!obj->isGrabbed){
				// $B$3$N%U%l!<%`$G;O$a$FGD;}$5$l$?$HH=Dj$5$l$?(B
				_getGrabbed[idx] = true;
				obj->isGrabbed = true;
			}
			obj++;
			idx++;
		}
	}

	return count_new_obj;
}

void ObjectManager::updateObjectLocation(
		const FlowManager& flow_manager,
		std::vector<std::vector<CvPoint2D32f> >* object_candidates,
		double threshold_point_distance){
	assert(object_candidates!=NULL);
	assert(!object_candidates->empty());

	// $BJ*BN$N0\F0@h$r0l;~E*$K3JG<$9$k$?$a$NJQ?t(B
	std::vector<CvPoint2D32f> new_points;

	// $B7W;;$N8zN(2=$N$?$a!"A07J$NJ*BN8uJdNN0h$r0O$`6k7A$r7W;;$7$F$*$/(B
	std::vector<CvBox2D> candidate_rects(object_candidates->size());
	ObjectInstance temp_obj(UINT_MAX);
	for(size_t i=0;i<candidate_rects.size();i++){
		temp_obj.setPoints((*object_candidates)[i]);
		candidate_rects[i] = temp_obj.getBox();
	}

	size_t obj_id = 0;
	for(iterator obj = begin(); obj != end(); obj++, obj_id++){
		if(flow_manager.getObjectFlow(obj_id).start_points.empty()){
			continue;
		}
		getValidPoints(
						flow_manager.getObjectFlow(obj_id),
						threshold_point_distance,
						candidate_rects,
						object_candidates,
						&new_points);
		obj->setPoints(new_points);
	}
}

size_t ObjectManager::updateNewObjects(
		const std::vector<std::vector<CvPoint2D32f> >& obj_candidates,
		size_t threshold_min_point_num){


	size_t count_new_obj = 0;
	for(size_t i=0;i<obj_candidates.size();i++){
		if(obj_candidates[i].size() > threshold_min_point_num){
			this->push_back(ObjectInstance(max_id,obj_candidates[i]));
			max_id++;
			count_new_obj++;
			_getHidden.resize(_getHidden.size()+1,false);
			_getRevealed.resize(_getRevealed.size()+1,false);
			_getGrabbed.resize(_getGrabbed.size()+1,false);
			_isMoved.resize(_isMoved.size()+1,false);
			_getMoved.resize(_getMoved.size()+1,false);
			_getStoped.resize(_getStoped.size()+1,true);
			_isStatic.resize(_isStatic.size()+1,true);
		}
	}
	return count_new_obj;
}

void ObjectManager::divideFlow(
		const Flow& flow,
		const std::map<size_t,std::pair<size_t,size_t> >& indice,
		std::map<size_t,Flow>* dist){

	std::map<size_t,std::vector<CvPoint2D32f> > temp_start_points,temp_end_points;
	divideObjectPoints<CvPoint2D32f>(
			flow.start_points,
			indice,
			&temp_start_points
			);
	divideObjectPoints<CvPoint2D32f>(
			flow.end_points,
			indice,
			&temp_end_points);
	std::map<size_t,std::vector<char> > temp_status;
	divideObjectPoints<char>(
			flow.status,
			indice,
			&temp_status);
	std::map<size_t,std::pair<size_t,size_t> >::const_iterator index;
	for(index = indice.begin(); index != indice.end(); index++){
		Flow temp_flow;
		temp_flow.start_points = temp_start_points[index->first];
		temp_flow.end_points = temp_end_points[index->first];
		temp_flow.status = temp_status[index->first];
		(*dist)[index->first] = temp_flow;
	}
}

template <class T> void ObjectManager::divideObjectPoints(
		const std::vector<T>& points,
		const std::map<size_t,std::pair<size_t,size_t> >&  indice,
		std::map<size_t,std::vector<T> >* objects)
{
	assert(objects!=NULL);

	std::map<size_t,std::pair<size_t,size_t> >::const_iterator pp;
	std::vector<T> temp;
	for(pp = indice.begin();pp != indice.end();pp++){
		temp.clear();
		size_t index = pp->second.first;
		size_t size = pp->second.second;
		temp.clear();
		temp.insert(temp.end(),points.begin()+index,points.begin()+index+size);
		(*objects)[pp->first] = temp;
	}
}

void ObjectManager::getValidPoints(
		const Flow& flow,
		double threshold,
		const std::vector<CvBox2D>& candidate_rects,
		std::vector<std::vector<CvPoint2D32f> >* obj_candidates,
		std::vector<CvPoint2D32f>* new_points){

	const size_t region_num = obj_candidates->size();
	const size_t flow_size = flow.start_points.size();
	assert(candidate_rects.size() == region_num);
	assert(candidate_rects.size() > 0);

	std::vector<size_t> counts_ep(region_num,0);// $BNN0hKh!"4^$^$l$k(Bend_point$B$N?t(B
	std::vector<size_t> counts_sp(region_num,0);// $BNN0hKh!"4^$^$l$k(Bstart point$B$N?t(B


	std::vector<std::vector<CvPoint2D32f> > new_point_candidates(region_num);

	size_t vanished_point_num = 0;

	// $B%*%W%F%#%+%k%U%m!<$NKh$K=hM}(B
	for(size_t pt_id=0;pt_id < flow_size;pt_id++){
		// $B%U%m!<$N>uBV$K4p$E$/%+%&%s%?!<$r?J$a$k(B
		if(flow.status[pt_id] == OpticalFlowPyramidalLK::INVALID){
			vanished_point_num++;
		}

		for(size_t region_id=0;region_id<region_num;region_id++){

			if(inBox(candidate_rects[region_id], flow.start_points[pt_id])){
				counts_sp[region_id]++;
			}

			if(flow.status[pt_id] == OpticalFlowPyramidalLK::INVALID){
				continue;
			}

			if(inBox(candidate_rects[region_id], flow.end_points[pt_id])){
				new_point_candidates[region_id].push_back(flow.end_points[pt_id]);
				counts_ep[region_id]++;
			}

		}
	}

	size_t ep_region = UINT_MAX;
	size_t sp_region = UINT_MAX;
	size_t max_count_ep = 0;
	size_t max_count_sp = 0;
	for(size_t region_id = 0;region_id < region_num; region_id++){
		if(counts_ep[region_id] > max_count_ep){
			max_count_ep = counts_ep[region_id];
			ep_region = region_id;
		}
		if(counts_sp[region_id] > max_count_sp){
			max_count_sp = counts_sp[region_id];
			sp_region = region_id;
		}
	}

	std::cerr << "sp_region: ep_region = " << sp_region << ":" << ep_region << std::endl;


	if(ep_region < region_num
			&& !new_point_candidates[ep_region].empty()){
		*new_points = new_point_candidates[ep_region];
	}
	else{
		// $BBP1~@h$,8+$D$+$i$J$$$N$G>C<:$H$$$&$3$H$K$9$k(B
		new_points->clear();
	}


	// $B%U%m!<$N;OE@<~JU$NBP1~E@$r=hM}(B
	if(!obj_candidates->at(sp_region).empty()){
		for(size_t pt_id=0;pt_id < flow_size;pt_id++){
			// $BJ*BN$,>C<:$7$F$*$i$:!"(Bflow$B$,(BINVALID$B$J$i%9%-%C%W(B
			if(!new_points->empty() && flow.status[pt_id] == OpticalFlowPyramidalLK::INVALID){
				continue;
			}


			// $BE@$,A07JNN0h$rJq4^$9$k6k7A$K4^$^$l$J$1$l$PBP1~E@$,$J$$$N$G%9%-%C%W(B
			if(!inBox(candidate_rects[sp_region], flow.start_points[pt_id])){
				continue;
			}

			std::vector<std::vector<CvPoint2D32f> >::iterator cand = obj_candidates->begin() + sp_region;
			std::vector<CvPoint2D32f>::iterator pcand;

			for(pcand = cand->begin();pcand!=cand->end();){
				// $B%U%m!<$N;OE@B&$N<~JU$NE@$r%4!<%9%H$H$7$F=hM}(B
				double dist = calcDistance(*pcand,flow.start_points[pt_id]);
				//					std::cerr << "dist = " << dist << "(";
				//					std::cerr << pcand->x << "," << pcand->y << ") : (";
				//					std::cerr << flow.start_points[pt_id].x << "," <<
				//						flow.start_points[pt_id].y << ")" << std::endl;
				if(dist < threshold){
					pcand = cand->erase(pcand);
				}
				else{
					pcand++;
				}
			}
		}
	}

	// $B8uJdFCD'E@$N$&$A(Bend_point$B$K6a$$$b$N$r(Bnew_points$B$KDI2C!"(B
	// candidate$B$+$i>C5n(B
	if(!new_points->empty()){
		size_t np_size = new_points->size();
		for(size_t pt_id=0;pt_id < np_size;pt_id++){
			if(!inBox(candidate_rects[ep_region], new_points->at(pt_id))){
				continue;
			}

			// $BA07JNN0hFb$NFCD'E@$G5wN%$,6a$$$b$N$r(Bnew_point$B$K2C$($F>C5n(B
			std::vector<std::vector<CvPoint2D32f> >::iterator cand = obj_candidates->begin() + ep_region;
			std::vector<CvPoint2D32f>::iterator pcand;

			for(pcand = cand->begin();pcand!=cand->end();){
				double dist = calcDistance(*pcand,new_points->at(pt_id));
				// $B0\F0@h$NE@$N<~JU$NE@$r(Bnew_point$B$K2C$($F$+$i>C5n(B
				if(dist < threshold){
					new_points->push_back(*pcand);
					pcand = cand->erase(pcand);
				}
				else{
					pcand++;
				}
			}
		}
	}

	return;
}

/*
 * @brief $B3FJ*BN$N>uBV$r2hA|$K=q$-2C$($k(B
 * @param img $B=q$-2C$($kA0$N852hA|(B
 * */
Image ObjectManager::visualize(const Image& src)const{
	Image tar(src);

	CvFont font;
	cvInitFont(&font,CV_FONT_HERSHEY_TRIPLEX,1,1);

	assert(size()==_getGrabbed.size());
	assert(size()==_getRevealed.size());
	assert(size()==_isMoved.size());
	assert(size()==_getMoved.size());
	assert(size()==_getStoped.size());
	assert(size()==_getHidden.size());
	assert(size()==_isStatic.size());

	size_t obj_idx=0;
	for(const_iterator obj = begin(); obj != end(); obj++,obj_idx++){
		// $BJ*BN$,4{$KGD;}$5$l$?8e$J$iIA2h$7$J$$(B
		if(obj->isGrabbed){
			continue;
		}

		// $B$=$NJ*BNMQ$N?'$r7h$a$k(B
		CvScalar col = col_manager.getColor(obj->getID());
		std::vector<CvPoint2D32f>::const_iterator pt;
		std::vector<CvPoint> points(obj->getPoints().size());
		size_t idx  = 0;
		for(pt = obj->getPoints().begin();
				pt != obj->getPoints().end(); pt++,idx++){
			points[idx] = cvPointFrom32f(*pt);
		}
		// $BFCD'E@$rBG$D(B
		for(idx = 0;idx < points.size(); idx++){
			cvCircle(tar.getIplImage(),points[idx],4,col);
		}

		// $B0O$`B?3Q7A$rI=<((B
		if(points.size()>2){
			CvMat point_mat = cvMat(1,points.size(), CV_32SC2, &points[0]);
			std::vector<int> hull(points.size());
			CvMat hull_mat = cvMat(1,points.size(),CV_32SC1, &hull[0]);

			cvConvexHull2(&point_mat,&hull_mat,CV_CLOCKWISE,0);
			hull.resize(hull_mat.cols);

			cvLine(tar.getIplImage(),points[hull[hull.size()-1]],points[0],col);
			for(idx = 0; idx < hull.size() - 1; idx++){
				cvLine(tar.getIplImage(),points[hull[idx]],points[hull[idx+1]],col);
			}
		}


		// $BJ*BN$N(BID$B$d>uBV$rI=<((B
		std::stringstream ss;
		ss << obj->getID();
		if(_getHidden[obj_idx]){
			ss << "(get hidden)";
		}
		else if(this->at(obj_idx).isHidden){
			ss << "(hidden)";
		}
		else if(_getRevealed[obj_idx]){
			ss << "(get revealed)";
		}
		else if(_isMoved[obj_idx]){
			ss << "(moving)";
		}
		else if(!_isStatic[obj_idx]){
			ss << "(moved?)";
		}

		cvPutText(tar.getIplImage(),ss.str().c_str(),cvPointFrom32f(obj->getBox().center),&font,col);

	}

	for(size_t i=0;i<_getGrabbed.size();i++){
		if(_getGrabbed[i]){
			std::cerr << "GRAB " << this->at(i).getID() << std::endl;
		}
	}

	return tar;
}


/*
 * @brief 2$B$D$NE@$N%f!<%/%j%C%I5wN%$r7W;;$9$k(B
 * @param a 1$B$DL\$NE@(B
 * @param b 2$B$DL\$NE@(B
 *
 * */
double ObjectManager::calcDistance(const CvPoint2D32f& a, const CvPoint2D32f& b){
	return sqrt( std::pow( a.x - b.x, 2) + std::pow( a.y - b.y, 2 ) );
}

/*
 * @brief $BF~NO$5$l$?E@$,6k7A$NCf$KF~$C$F$$$k$+$I$&$+H=Dj$9$k(B
 * */
bool ObjectManager::inBox(const CvBox2D& box, const CvPoint2D32f& pt){
	CvPoint2D32f relative_pt;
	relative_pt.x = pt.x - box.center.x;
	relative_pt.y = pt.y - box.center.y;
	double len = calcDistance(box.center,pt);
	double theta = acos( relative_pt.x / len ) * 180.0 / CV_PI;
	if( relative_pt.y < 0 ){
		theta = 360 - theta;
	}

	// $B3QEY$H:BI8$N79$-$r(Bbox$B$K9g$o$;$k(B
	theta -= box.angle;
	if(theta<0){
		theta += 360;
	}

	relative_pt.x = len * cos(theta);
	relative_pt.y = len * sin(theta);
	if( abs(relative_pt.x) < box.size.width / 2 
			&& abs(relative_pt.y) < box.size.height / 2 ){
		return true;
	}
	return false;
}


bool ObjectManager::isHidden(size_t idx)const{
	assert(idx < size());
	return (begin()+idx)->isHidden;
}

bool ObjectManager::getHidden(size_t idx)const{
	assert(idx < _getHidden.size());
	return _getHidden[idx];
}

bool ObjectManager::getRevealed(size_t idx)const{
	assert(idx < _getRevealed.size());
	return _getRevealed[idx];
}

bool ObjectManager::isGrabbed(size_t idx)const{
	assert(idx < size());
	return (begin()+idx)->isGrabbed;
}

bool ObjectManager::getGrabbed(size_t idx)const{
	assert(idx < _getGrabbed.size());
	return _getGrabbed[idx];
}

bool ObjectManager::isMoved(size_t idx)const{
	assert(idx < _isMoved.size());
	return _isMoved[idx];
}

bool ObjectManager::isStatic(size_t idx)const{
	assert(idx < _isStatic.size());
	return _isStatic[idx];
}
bool ObjectManager::getMoved(size_t idx)const{
	assert(idx < _getMoved.size());
	return _getMoved[idx];
}
bool ObjectManager::getStoped(size_t idx)const{
	assert(idx < _getStoped.size());
	return _getStoped[idx];
}



ColorManager ObjectManager::col_manager;
