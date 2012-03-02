/*!
 * @file sklcvgpu_utils.h
 * @author a_hasimoto
 * @date Date Created: 2012/Jan/25
 * @date Last Change:2012/Feb/20.
 */
#ifndef __SKL_CV_GPU_UTILS_H__
#define __SKL_CV_GPU_UTILS_H__

#ifdef DEBUG
#define DEBUG_CV_GPU_UTILS
#endif

#include <cv.h>
#include <highgui.h>
#include <opencv2/gpu/gpu.hpp>
namespace skl{
	namespace gpu{

		void meanStdDev(const cv::gpu::GpuMat& mtx,cv::Scalar& mean, cv::Scalar& stddev);
	} // namespace gpu
} // namespace skl

#endif // __SKL_CV_GPU_UTILS_H__
