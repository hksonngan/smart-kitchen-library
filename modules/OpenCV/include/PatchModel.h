/*!
 * @file PatchModel.h
 * @author a_hasimoto
 * @date Last Change:2011/Nov/01.
 */
#ifndef __PATCH_MODEL_H__
#define __PATCH_MODEL_H__

#include <list>
#include <set>
#include <map>
#include "sklcvutils.h"

#define PATCH_MODEL_BLOCK_SIZE 4

#define PATCH_EDGE_CANNY_THRESH1 16
#define PATCH_EDGE_CANNY_THRESH2 32

#define REMOVED_OBJECT_MIN_REGION_RATE 0.85
#define REMOVED_OBJECT_MIN_SCORE 0.4
#define HIDDEN_OBJECT_MIN_RECALL_RATE 0.0
#define IS_THERE_OBJECT_MIN_RECALL_RATE 0.4

namespace skl{

#define PATCH_DILATE 32

		class Patch{
			public:
			enum Type{
				original=0,
				dilate=1
			};
			public:
				Patch();
				Patch(const cv::Mat& mask, const cv::Mat& newest_image, const cv::Mat& current_bg,const std::vector<cv::Point>& points);
				~Patch();

				void set(const cv::Mat& mask, const cv::Mat& newest_image, const cv::Mat& current_bg,const std::vector<cv::Point>& points);

				void setCoveredState(const cv::Rect& rect,const cv::Mat& mask,bool isCovered);
				void setCoveredState(int x,int y,bool isCovered);


				void save(const std::string& filename, Type type, const std::string& edge_filename="")const;

				/*** Accessor ***/
				float maskValue(int x,int y, Type type=original)const;
				const unsigned char* operator()(int x,int y, Type type=original)const;
				unsigned char* operator()(int x,int y,Type type=original);

				const cv::Rect& roi(Type type=original)const{return _roi[type];}
				cv::Mat& image(Type type=original){return _image[type];}
				const cv::Mat& image(Type type=original)const{return _image[type];}
				cv::Mat& background(Type type=original){return _background[type];}
				const cv::Mat& background(Type type=original)const{return _background[type];}
				const cv::Mat& mask(Type type=original)const{return _mask[type];}

				const cv::Mat& edge()const{return _edge;}
				void edge(const cv::Mat& __edge){_edge = __edge.clone();}
				size_t edge_count()const{return _edge_count;};

				const std::vector<cv::Point>& points()const{return _points;}

		protected:
				cv::Mat _mask[2];
				cv::Mat _image[2];
				cv::Mat _background[2];
				cv::Mat _edge;
				cv::Rect _roi[2];
				cv::Point center;
				cv::Mat _covered_state;
				size_t _edge_count;
				cv::Size base_size;
				std::vector<cv::Point> _points;
			private:
				void cvtG2L(int* x,int* y, Type type)const;
				bool isIn(int local_x,int local_y, Type type)const;
				cv::Rect extractEdges(cv::Mat& edge, std::vector<cv::Point>* edge_points)const;
		};

		class PatchLayer{
			public:
				PatchLayer(std::map<size_t,Patch>* patches);
				~PatchLayer();
				void push(size_t ID);
				void erase(size_t ID);

				// Ω≈§ §√§∆§§§ÅE—•√•¡§Œ§¶§¡°¢1§ƒ§¿§±æÂ§Œ§‚§Œ§Ú ÷§π
				// § §±§ÅE–UINT_MAX§Ú ÷§π
				size_t getUpperPatch(size_t ID, Patch::Type)const;
				std::vector<size_t> getAllBeneathPatch(size_t ID,Patch::Type type);
				size_t getOrder(size_t ID)const;
				size_t getTopLayer()const;
				size_t size()const;
				size_t getLayer(size_t i,bool from_top = true)const;
			protected:
				std::list<size_t> layer_order;
				std::map<size_t,Patch>* patches;
			private:
				static bool isOverlayed(const cv::Rect& common_rect,const Patch& p1,const Patch& p2, Patch::Type type, cv::Mat& mask=cv::Mat());
		};

		class PatchModel{
			friend class PatchLayer;
			friend class PatchTracker;
			public:
				PatchModel();
				PatchModel(const cv::Mat& base);
				~PatchModel();
				Patch& operator()(size_t ID);
				void base(const cv::Mat& __bg);
				const cv::Mat& base()const;
				size_t putPatch(const cv::Mat& mask,const cv::Mat& newest_image);
				void takePatch(size_t ID,std::vector<size_t>* taken_patch_ids);
				void updateNewestBG();
				void newest_bg(const cv::Mat& bg);
				const cv::Mat& newest_bg()const;
				size_t checkLostPatches(const cv::Mat& __mask,const cv::Mat& __newest_image,const cv::Mat& __newest_bg)const;
				bool isThere(size_t ID,const cv::Mat& newest_image)const;

				void getHiddenPatches(const cv::Mat& human_mask, std::vector<size_t>* newly_hidden_patch_ids, std::vector<size_t>* reappeared_patch_ids);

				void save(const std::string& file_head,const std::string& ext)const;
				const Patch& operator[](size_t ID)const;
			protected:
				std::map<size_t,Patch> patches;
				cv::Mat _newest_bg;
				cv::Mat _base;
				PatchLayer layer;

				static double calcCommonEdge(
						const CvRect& common_rect,
						const Patch& patch_1,
						const Patch& patch_2);
				static void getObjectSamplePoints(const cv::Mat& mask, std::vector<cv::Point>* points);

			private:
				size_t max_id;
				std::vector<size_t> put_list;
				std::vector<bool> isMoving;
				cv::Mat hidden_image;
				cv::Mat hidden_mask;
				bool changed_bg;
				bool changed_fg;
				std::set<size_t> patches_in_hidden;
		};

}

#endif // __PATCH_MODEL_H__
