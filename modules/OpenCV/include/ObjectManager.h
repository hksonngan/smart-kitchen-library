/*!
 * @file ObjectManager.h
 * @author a_hasimoto
 * @date Last Change:2011/Sep/12.
 * */

#ifndef __OBJECT_MANAGER_H__
#define __OBJECT_MANAGER_H__

#include <vector>
#include <map>
#include "MmplTime.h"
#include "cv.h"
#include "ObjectInstance.h"
#include "ImageLabeledImage.h"
#include "FlowManager.h"
#include "ColorManager.h"

namespace mmpl{
	namespace image{
		/*
		 * @class ObjectManager
		 * @brief 台上の物体の出現・把持を管理するクラス
		 * */
		class ObjectManager:public std::vector<ObjectInstance>{
			public:
				typedef std::vector<ObjectInstance>::iterator iterator;
				typedef std::vector<ObjectInstance>::const_iterator const_iterator;

				ObjectManager();
				~ObjectManager();
				bool updateHiddenStatus(const Image& human_mask,double threshold);
				bool updateMovedStatus(const Image& object_mask,double threshold);
				bool updateStaticStatus(const Image& whole_mask,double threshold);

				std::vector<std::vector<CvPoint2D32f> > getObjectPointsIf(const std::vector<bool>& condition)const;

				bool isHidden(size_t idx)const;
				bool getHidden(size_t idx)const;
				bool getRevealed(size_t idx)const;
				bool isGrabbed(size_t idx)const;
				bool getGrabbed(size_t idx)const;
				bool isMoved(size_t idx)const;
				bool getMoved(size_t idx)const;
				bool getStoped(size_t idx)const;
				bool isStatic(size_t idx)const;

				size_t updateObjectStatus(
						const FlowManager& flow_manager,
						const std::vector<std::vector<CvPoint2D32f> >& _obj_candidates,
						double threshold_point_distance,
						size_t threshold_min_point_num,
						double threshold_grab,
						double threshold_release);
				Image visualize(const Image& img)const;

				mmpl::Time timestamp;
			protected:
				//! このフレームで始めて隠されたかどうか
				//  (updateHiddenStatusで更新)
				std::vector<bool> _getHidden;
				
				//! このフレームで隠されていたのが戻ったかどうか
				//  (updateHiddenStatusで更新)
				std::vector<bool> _getRevealed;

				//! このフレームで物体の把持が確認されたのか
				//  (updateObjectStatusの中で更新)
				std::vector<bool> _getGrabbed;

				//! このフレームで動いたかどうか
				std::vector<bool> _isMoved;

				//! このフレームから動き始めたかどうか
				std::vector<bool> _getMoved;

				//! このフレームで動きを止めたかどうか
				std::vector<bool> _getStoped;


				//! このフレームで止まっているかどうか
				std::vector<bool> _isStatic;


				static void divideFlow(
						const Flow& src,
						const std::map<size_t,std::pair<size_t,size_t> >& indice,
						std::map<size_t,Flow>* dist);
				template<class T> static void divideObjectPoints(
						const std::vector<T>& points,
						const std::map<size_t,std::pair<size_t,size_t> >&  indice,
						std::map<size_t,std::vector<T> >* objects);

				static void getValidPoints(
						const Flow& flow,
						double threshold,
						const std::vector<CvBox2D>& candidate_rects,
						std::vector<std::vector<CvPoint2D32f> >* obj_candidates,
						std::vector<CvPoint2D32f>* new_points);
			private:
				size_t max_id;
				static ColorManager col_manager;

				static double calcDistance(const CvPoint2D32f& a, const CvPoint2D32f& b);
				static bool inBox(const CvBox2D& box, const CvPoint2D32f& pt);

				void updateObjectLocation(
						const FlowManager& flow_manager,
						std::vector<std::vector<CvPoint2D32f> >* object_candidates,
						double threshold_point_distance);

				size_t updateNewObjects(
						const std::vector<std::vector<CvPoint2D32f> >& obj_candidates,
						size_t threshold_min_point_num);
				//				Image visualizeBox(const Image& src,const CvBox2D& box,const CvScalar& col);
		};
	} // namespace image
} // namespace mmpl

#endif // __OBJECT_MANAGER_H__
