/*!
 * @file VideoCapture.h
 * @author a_hasimoto
 * @date Date Created: 2012/Jan/12
 * @date Last Change:2012/Jan/13.
 */
#ifndef __SKL_VIDEO_CAPTURE_H__
#define __SKL_VIDEO_CAPTURE_H__

#include "skl.h"
#include <cv.h>
#include <highgui.h>


namespace skl{
	class VideoCapture;

	/*!
	 * @class VideoCaptureがサポートしているパラメタを格納するクラス
	 * */
	typedef std::map<int,double>::const_iterator VideoParamIter;
	class VideoParams: public Printable<VideoParams>{
			friend class VideoCapture;
		public:
			VideoParams();
			virtual ~VideoParams();
			VideoParams(const std::string& filename);
			VideoParams(const VideoParams& other);

			std::string print()const;
			bool scan(const std::string& buf);

			bool load(const std::string& filename);
			void save(const std::string& filename)const;

			bool set(const std::string& prop_name,double val);
			bool set(int prop_id,double val);
			VideoParamIter begin()const;
			VideoParamIter end()const;
		protected:
			std::map<std::string,int> property_name_id_map;
			std::map<int,double> property_id_value_map;
	};

	/*!
	 * @class パラメタの読み込み機能などを強化したVideoCapture
	 */
	class VideoCapture: public cv::VideoCapture{

		public:
			VideoCapture();
			virtual ~VideoCapture();
			virtual bool set(const VideoParams& params);
			virtual bool set(int prop_id,double val);
			virtual VideoParams get();
		protected:
			VideoParams params;
		private:

	};

} // skl

#endif // __SKL_VIDEO_CAPTURE_H__

