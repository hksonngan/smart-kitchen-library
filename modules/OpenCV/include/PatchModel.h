/*!
 * @file PatchModel.h
 * @author 橋本敦史
 * @date Last Change:2011/Nov/01.
 */
#ifndef __PATCH_MODEL_H__
#define __PATCH_MODEL_H__

#include <set>
#include <map>
#include "ImageImage.h"

#define PATCH_EDGE_CANNY_THRESH1 16
#define PATCH_EDGE_CANNY_THRESH2 32

#define REMOVED_OBJECT_MIN_REGION_RATE 0.85
#define REMOVED_OBJECT_MIN_SCORE 0.4
#define HIDDEN_OBJECT_MIN_RECALL_RATE 0.0
#define IS_THERE_OBJECT_MIN_RECALL_RATE 0.4

namespace mmpl{
	namespace image{

#define PATCH_DILATE 32
#define BLOCK_SIZE 4

		class Patch{
			public:
			enum Type{
				original=0,
				dilate=1
			};
			public:
				Patch();
				Patch(const Image& mask, const Image& newest_image, const Image& current_bg);
				~Patch();

				void set(const Image& mask, const Image& newest_image, const Image& current_bg);

				void setVisibility(const CvRect& rect,const Image& mask,bool visibility);

				// noch nicht
				// homographyに従ってmask[original],局所特徴の位置や値を変換
				// newest_imageと局所特徴が不一致ならmask[original]を0.0に。
				// 最後にmaskに基づいて他の画像を取得
//				void move(const cv::mat& homography,const Image& newest_image);
//
				void save(const std::string& filename,Type type,int width,int height,const std::string& edge_filename="")const;

				/*** Accessor ***/
				float maskValue(int x,int y, Type type=original)const;
				const unsigned char* operator()(int x,int y, Type type=original)const;
				unsigned char* operator()(int x,int y,Type type=original);
				const CvRect& getRect(Type type=original)const;
				bool isVisible(int x,int y);
				Image& getImage(Type type=original);
				const Image& getImage(Type type=original)const;
				Image& getBG(Type type=original);
				const Image& getBG(Type type=original)const;
				const Image& getMask(Type type=original)const;
				static CvRect getRect(const Image& mask,size_t* pix_num=NULL);

				const Image& getEdgeImage()const;
				void setEdgeImage(const Image& _edge);
				size_t getEdgePixelNum()const;

				void setVisibility(int x,int y,bool isVisible);

				static CvRect getCommonRectanglarRegion(const CvRect& r1,const CvRect& r2);
				const std::vector<CvPoint>& getPoints()const;
			protected:
				Image mask[2];
				Image image[2];
				Image hidden[2];
				Image edge;
				CvRect rect[2];
				CvPoint center;
				Image visibility;
				std::vector<CvPoint> points;
				size_t edge_pixel_num;
			private:
				void convXY_G2L(int* x,int* y, Type type)const;
				bool isIn(int local_x,int local_y, Type type)const;
				void setPoints();
				void setDilate(
						const CvRect& rect,
						const Image& mask,
						const Image& src,
						const Image& bg);
				static void crop(Image* dist,const Image& src,const CvRect& rect,const IplImage* mask=NULL);
				CvRect extractEdges(Image* edge,const Image& mask,const Image& src,const Image& bg,size_t* edge_pix_num=NULL)const;
		};

		class PatchLayer{
			public:
				PatchLayer(std::map<size_t,Patch>* patches);
//				PatchLayer(const PatchLayer& other);
				~PatchLayer();
				void push(size_t ID);
				void erase(size_t ID);

				// 重なっているパッチのうち、1つだけ上のものを返す
				// なければUINT_MAXを返す
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
				static bool isOverlayed(const CvRect& common_rect,const Patch& p1,const Patch& p2, Patch::Type type, Image* mask=NULL);
		};

		class PatchModel{
			friend class PatchLayer;
			friend class PatchTracker;
			public:
				PatchModel();
				PatchModel(const Image& base_bg);
				~PatchModel();
				Patch& operator()(size_t ID);
				void setBaseBG(const Image& base_bg);
				const Image& getBaseBG()const;
				size_t putPatch(const Image& mask,const Image& newest_image);
//				bool movePatch(size_t ID,const cv::Mat& homography,const Image& newest_image);
				void takePatch(size_t ID,std::vector<size_t>* taken_patch_ids);
				void updateNewestBG();
				void setNewestBG(const Image& bg);
				const Image& getNewestBG()const;
				size_t checkMovingPatches(const Image& mask,const Image& newest_image,const Image& newest_bg)const;
				bool isThere(size_t ID,const Image& newest_image)const;

				void getHiddenPatches(const Image& human_mask,std::vector<size_t>* newly_hidden_patch_ids,std::vector<size_t>* reappeared_patch_ids);

				void save(const std::string& file_head,const std::string& ext)const;
				const Patch& operator[](size_t ID)const;
			protected:
				std::map<size_t,Patch> patches;
				Image newest_bg;
				Image base_bg;
				PatchLayer layer;

				static double calcCommonEdge(
						const CvRect& common_rect,
						const Patch& patch_1,
						const Patch& patch_2);

			private:
				size_t max_id;
				std::vector<size_t> put_list;
				std::vector<bool> isMoving;
				Image hidden_image;
				Image hidden_mask;
				bool changed_bg;
				bool changed_fg;
				std::set<size_t> patches_in_hidden;
		};
	}
}

#endif // __PATCH_MODEL_H__
