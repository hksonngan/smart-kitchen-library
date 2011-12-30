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
		 * @brief ����ʪ�Τνи����Ļ���������륯�饹
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
				//! ���Υե졼��ǻϤ�Ʊ����줿���ɤ���
				//  (updateHiddenStatus�ǹ���)
				std::vector<bool> _getHidden;
				
				//! ���Υե졼��Ǳ�����Ƥ����Τ���ä����ɤ���
				//  (updateHiddenStatus�ǹ���)
				std::vector<bool> _getRevealed;

				//! ���Υե졼���ʪ�Τ��Ļ�����ǧ���줿�Τ�
				//  (updateObjectStatus����ǹ���)
				std::vector<bool> _getGrabbed;

				//! ���Υե졼���ư�������ɤ���
				std::vector<bool> _isMoved;

				//! ���Υե졼�फ��ư���Ϥ᤿���ɤ���
				std::vector<bool> _getMoved;

				//! ���Υե졼���ư����ߤ᤿���ɤ���
				std::vector<bool> _getStoped;


				//! ���Υե졼��ǻߤޤäƤ��뤫�ɤ���
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
