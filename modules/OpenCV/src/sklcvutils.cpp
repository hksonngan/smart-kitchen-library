#include "sklcvutils.h"

cv::Rect operator&(const cv::Rect& left, const cv::Rect& right){
	cv::Rect rect(left.x,left.y,left.x+left.width,left.y+left.height);
	rect.x = rect.x > right.x ? rect.x : right.x;
	rect.y = rect.y > right.y ? rect.y : right.y;
	rect.width = rect.width < right.x + right.width ? rect.width : right.x + right.width;
	rect.height = rect.height < right.y + right.height ? rect.height : right.y + right.height;
	rect.width -= rect.x;
	rect.height -= rect.y;
	return rect;
}
bool operator&&(const cv::Rect& left, const cv::Rect& right){
	cv::Rect rect = left & right;
	return (rect.width > 0) && (rect.height > 0);
}

cv::Rect operator|(const cv::Rect& left, const cv::Rect& right){
	cv::Rect rect(left.x,left.y,left.x+left.width,left.y+left.height);
	rect.x = rect.x < right.x ? rect.x : right.x;
	rect.y = rect.y < right.y ? rect.y : right.y;
	rect.width = rect.width > right.x + right.width ? rect.width : right.x + right.width;
	rect.height = rect.height > right.y + right.height ? rect.height : right.y + right.height;
	rect.width -= rect.x;
	rect.height -= rect.y;
	return rect;
}



namespace skl{
	/*!
	 * calc minimum rect which contains points
	 */
	cv::Rect fitRect(const std::vector< cv::Point >& points){
		cv::Rect rect(INT_MAX,INT_MAX,0,0);
		for(size_t i=0;i<points.size();i++){
			int x = points[i].x;
			int y = points[i].y;
			rect.x = rect.x < x ? rect.x : x;
			rect.y = rect.y < y ? rect.y : y;
			rect.width = rect.width > x ? rect.width : x;
			rect.height = rect.height > y ? rect.height : y;
		}
		return rect;
	}

	/*!
	 * @brief calc HLS value
	 * */
	unsigned char getHLSvalue(int min,int max,unsigned char hue){
		if(max>255)max=255;if(min>255)min=255;
		if(hue<=30) return (unsigned char)(min+(max-min)*hue/30);
		else if(hue<=90)return (unsigned char)max;
		else if(hue<=120) return (unsigned char)(min+(max-min)*(120-hue)/30);
		else return (unsigned char)min;
	}

	cv::Vec3b convHLS2BGR(const cv::Vec3b& hls){
		unsigned char r,g,b,h,l,s;
		h=hls[0];
		l=hls[1];
		s=hls[2];

		int max,min;
		int hue;

		if(l<=128) max=(l*(255+(int)s))/255;
		else max=(l*(255-s)+s)/255;
		min=2*l-max;

		if(s==0)r=g=b=l;
		else{
			hue=(int)h+60;
			if(hue>=180) hue=hue-180;
			else if(hue<0) hue=hue+180;
			r=getHLSvalue(min,max,hue);

			hue=(int)h;
			if(hue>=180) hue=hue-180;
			else if(hue<0) hue=hue+180;
			g=getHLSvalue(min,max,hue);

			hue=(int)h-60;
			if(hue>=180) hue=hue-180;
			else if(hue<0) hue=hue+180;
			b=getHLSvalue(min,max,hue);
		}

		cv::Vec3b bgr;
		bgr[0]=b;
		bgr[1]=g;
		bgr[2]=r;
		return bgr;
	}

	// 7*2*2 valiations of color
	cv::Vec3b assignColor(size_t ID){
		int hue_id = ID % 7;
		int lum_id = ID / 7;
		int sat_id = lum_id / 3;
		lum_id %= 3;
		sat_id %= 2;
		if(hue_id == 6){
			cv::Vec3b bgr;
			int gray_level = 255 - (sat_id + lum_id * 2) * 32;
			for(int c=0;c<3;c++){
				bgr[c] = gray_level;
			}
			return bgr;
		}
		cv::Vec3b hls;
		hls[0] = hue_id * 30; // hue
		hls[1] = 255 - lum_id * 64;
		hls[2] = 255 - sat_id * 128;
		//	std::cerr << (int)hls[0] << ", " << (int)hls[1] << ", " << (int)hls[2] << std::endl;
		return convHLS2BGR(hls);
	}


	cv::Mat visualizeRegionLabel(const cv::Mat& label,size_t region_num){
		cv::Mat vis = cv::Mat::zeros(label.size(),CV_8UC3);
		if(region_num == 0){
			return vis;
		}
		std::vector<cv::Vec3b> colors(region_num);
		for(size_t i=0;i<colors.size();i++){
			colors[i] = assignColor(i);
			//		std::cerr << (int)colors[i][0] << ", " << (int)colors[i][1] << ", " << (int)colors[i][2] << std::endl;
		}

		for(int y=0;y<label.rows;y++){
			for(int x=0;x<label.cols;x++){
				vis.at<cv::Vec3b>(y,x) = colors[label.at<short>(y,x)-1];
			}
		}
		return vis;
	}

	template<class T> void setWeight(const T& mask,double* w1, double* w2){
		*w1 = mask;
		*w2 = 1.0 - mask;
	}
	template<> void setWeight<unsigned char>(const unsigned char& mask, double* w1, double* w2){
		*w1 = mask;
		*w2 = 255 - mask;
		*w1 /= 255.0;
		*w2 /= 255.0;
	}

	template<class T> T blend(const T& pix1, const T& pix2, double w1,double w2){
		return static_cast<T>(w1 * pix1 + w2 * pix2);
	}
	template<> cv::Vec3b blend(const cv::Vec3b& pix1,const cv::Vec3b& pix2, double w1, double w2){
		cv::Vec3b val;
		for(size_t i=0;i<3;i++){
			val[i] = static_cast<unsigned char>(w1 * pix1[i] + w2 * pix2[i] + 0.5);
		}
		return val;
	}

	template <class ElemType,class WeightType> class ParallelBlending{
		public:
			ParallelBlending(
					const cv::Mat& src1,
					const cv::Mat& src2,
					const cv::Mat& mask,
					cv::Mat& dest):
				src1(src1),src2(src2),mask(mask),dest(dest){}
			~ParallelBlending(){}
			void operator()(const cv::BlockedRange& range)const{
				for(int i=range.begin();i!=range.end();i++){
					int y = i / mask.cols;
					int x = i % mask.cols;
					double weight1, weight2;
					setWeight<WeightType>(mask.at<WeightType>(y,x), &weight2,&weight1);
					dest.at<ElemType>(y,x) = blend<ElemType>(
							src1.at<ElemType>(y,x),
							src2.at<ElemType>(y,x),
							weight1,weight2);
				}
			}
		protected:
			const cv::Mat& src1;
			const cv::Mat& src2;
			const cv::Mat& mask;
			cv::Mat& dest;

	};



template<class ElemType,class WeightType> void blending(const cv::Mat& src1,const cv::Mat& src2, const cv::Mat& weight_mask, cv::Mat& dest){
		assert(weight_mask.type()==CV_32FC1);
		int channels = src1.size();
		int height = src1.rows;
		int width = src1.cols;
		assert(weight_mask.rows==height);
		assert(weight_mask.cols==width);
		assert(src2.rows==height);
		assert(src2.cols==height);
		assert(src2.channels()== channels);
		cv::parallel_for(
				cv::BlockedRange(0,width*height),
				ParallelBlending<ElemType,WeightType>(src1,src2,weight_mask,dest)
				);
	}


}// namespace skl
