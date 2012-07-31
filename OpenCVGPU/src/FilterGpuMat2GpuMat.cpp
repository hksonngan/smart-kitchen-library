/*!
 * @file FilterGpuMat2GpuMat.cpp
 * @author a_hasimoto
 * @date Date Created: 2012/Feb/10
 * @date Last Change:2012/Feb/10.
 */
#include "FilterGpuMat2GpuMat.h"

namespace skl{
	namespace gpu{
		// specialization for void
/*		template<> void FilterGpuMat2GpuMat<void>::compute(const cv::gpu::GpuMat& src,cv::gpu::GpuMat& dest){
			compute(src,dest,cv::gpu::Stream::Null());
		}

		template<> void FilterGpuMat2GpuMat<void>::compute(const cv::gpu::GpuMat& src,cv::Mat& dest,cv::gpu::Stream& s){
			compute(src,dest_buf,s);
			dest_buf.download(dest);
		}

		template<> void FilterGpuMat2GpuMat<void>::compute(const cv::Mat& src,cv::gpu::GpuMat& dest,cv::gpu::Stream& s){
			cv::gpu::ensureSizeIsEnough(src.size(),src.type(),src_buf);
			src_buf.upload(src);
			compute(src_buf,dest,s);
		}

		template<> void FilterGpuMat2GpuMat<void>::compute(const cv::Mat& src,cv::Mat& dest,cv::gpu::Stream& s){
			cv::gpu::ensureSizeIsEnough(src.size(),src.type(),src_buf);
			src_buf.upload(src);
			compute(src_buf,dest_buf,s);
			dest_buf.download(dest);
		}
		*/
	}
}
