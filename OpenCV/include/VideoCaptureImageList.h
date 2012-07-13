/*!
 * @file VideoCaptureImageList.h
 * @author a_hasimoto
 * @date Date Created: 2012/Jan/18
 * @date Last Change:2012/Jul/06.
 */
#ifndef __SKL_VIDEO_CAPTURE_IMAGE_LIST_H__
#define __SKL_VIDEO_CAPTURE_IMAGE_LIST_H__

// STL
#include <vector>


#include "sklcv.h"

namespace skl{

	/*!
	 * @brief 画像列が記述されたファイルリストからファイルを読み込む
	 */
	class VideoCaptureImageList: public VideoCaptureInterface<VideoCaptureImageList>{

		public:
			using _VideoCaptureInterface::get;
			using _VideoCaptureInterface::set;
			VideoCaptureImageList();
			virtual ~VideoCaptureImageList();
			virtual bool open(const std::string& filename);
			bool isOpened()const{return !img_list.empty();}
			void release();
			bool grab();
			bool retrieve(cv::Mat& image, int channel=0);

			bool set(capture_property_t prop_id,double val);
			double get(capture_property_t prop_id);
		protected:
			int index;
			std::vector<std::string> img_list;
			int bayer_change;
			int imread_option;
			int checked_frame_pos;
		private:
			// 画像列が記されたファイルが専門で、
			// カメラなどのデバイスは読み込まない
			bool open(int device){return false;}
	};

} // skl

#endif // __SKL_VIDEO_CAPTURE_IMAGE_LIST_H__

