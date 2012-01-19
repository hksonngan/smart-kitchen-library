/*!
 * @file VideoCapture.h
 * @author a_hasimoto
 * @date Date Created: 2012/Jan/12
 * @date Last Change:2012/Jan/18.
 */
#ifndef __SKL_VIDEO_CAPTURE_H__
#define __SKL_VIDEO_CAPTURE_H__

#include "skl.h"
#include "cvtypes.h"
#include <cv.h>
#include <vector>
#include "VideoCaptureInterface.h"

namespace skl{

	/*!
	 * @class パラメタの読み込み機能などを強化したVideoCapture
	 */
	class VideoCapture: public VideoCaptureInterface<VideoCapture>{
		public:
			using _VideoCaptureInterface::set;
			using _VideoCaptureInterface::get;
			VideoCapture();
			virtual ~VideoCapture();

			bool open(const std::string& filename){
				release();
				return push_back(filename);
			}
			bool open(int device){
				release();
				return push_back(device);
			}
			template<class Iter> bool open(Iter first,Iter last);

			bool isOpened()const;
			void release();
			bool grab();
			virtual bool retrieve(cv::Mat& image, int channel=0);

			bool set(capture_property_t prop_id,double val);
			double get(capture_property_t prop_id);

			_VideoCaptureInterface& operator[](int device);
			size_t size()const{return cam_interface.size();}

			virtual bool push_back(const std::string& filename);
			virtual bool push_back(int device);

			VideoCapture& operator>>(std::vector<cv::Mat>& mat_vec);

		protected:
			std::vector<cv::Ptr<_VideoCaptureInterface> > cam_interface;
		private:
	};


	template <class Iter> bool VideoCapture::open(Iter first, Iter last){
		size_t id = 0;
		release();
		for(Iter iter = first; iter != last; iter++,id++){
			if(!push_back(*iter)) return false;
		}
		return true;
	}

} // skl

#endif // __SKL_VIDEO_CAPTURE_H__

