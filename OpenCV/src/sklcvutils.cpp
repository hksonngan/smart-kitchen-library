#include "sklcvutils.h"
#include <highgui.h>

#define CLIP(in, out)\
	in = in < 0 ? 0 : in;\
in = in > 255 ? 255 : in;\
out=in;

#ifdef DEBUG
#define DEBUG_SKLUTILS
#endif

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
		rect.width -= (rect.x-1);
		rect.height -= (rect.y-1);
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
		if(label.type()!=CV_16SC1){
			assert(label.type()==CV_16SC1);
		}
		if(region_num == 0){
			return vis;
		}
		std::vector<cv::Vec3b> colors(region_num);
		for(size_t i=0;i<region_num;i++){
			if(region_num<32){
				colors[i] = assignColor(i);
			}
			else{
				for(size_t c=0;c<3;c++){
					colors[i][c] = rand() % UCHAR_MAX;
				}
			}
			//		std::cerr << (int)colors[i][0] << ", " << (int)colors[i][1] << ", " << (int)colors[i][2] << std::endl;
		}

		for(int y=0;y<label.rows;y++){
			for(int x=0;x<label.cols;x++){
				short l = label.at<short>(y,x);
				if(l==0) continue;
				vis.at<cv::Vec3b>(y,x) = colors[l-1];
			}
		}
		return vis;
	}





	void cvtBayer2BGR_NN(const cv::Mat& bayer, cv::Mat& bgr, int code);
	void cvtBayer2BGR_EDGESENSE(const cv::Mat& bayer, cv::Mat& bgr, int code);

	void cvtBayer2BGR(const cv::Mat& _bayer, cv::Mat& bgr, int code, int algo_type){
		cv::Mat bayer;
		if(!_bayer.isContinuous()){
			bayer = cv::Mat(_bayer.size(),_bayer.type());
			_bayer.copyTo(bayer);
		}
		else{
			bayer = _bayer;
		}
		assert(bayer.isContinuous());
		bgr = cv::Mat::zeros(bayer.rows,bayer.cols,CV_8UC3);
		if(algo_type == BAYER_SIMPLE){
			cv::cvtColor(bayer,bgr,code);
		}
		else if(algo_type == BAYER_NN){
			cvtBayer2BGR_NN(bayer,bgr,code);
		}
		else if(algo_type == BAYER_EDGE_SENSE){
			cvtBayer2BGR_EDGESENSE(bayer,bgr,code);
		}
	}

	void cvtBayer2BGR_NN(const cv::Mat& bayer, cv::Mat& bgr, int type){
		int sx = bayer.cols;
		int sy = bayer.rows;
		const unsigned char* src = bayer.ptr<unsigned char>(0);
		unsigned char* dest = bgr.ptr<unsigned char>(0);

		unsigned char *outR=NULL, *outG=NULL, *outB=NULL;
		register int i,j;

		// sx and sy should be even
		switch (type) {
			case CV_BayerGR2BGR:
			case CV_BayerBG2BGR:
				outR=&dest[0];
				outG=&dest[1];
				outB=&dest[2];
				break;
			case CV_BayerGB2BGR:
			case CV_BayerRG2BGR:
				outR=&dest[2];
				outG=&dest[1];
				outB=&dest[0];
				break;
			default:
				std::cerr << "Bad Bayer pattern ID: " << type << std::endl;
				return;
				break;
		}

		switch (type) {
			case CV_BayerGR2BGR: //-------------------------------------------
			case CV_BayerGB2BGR:
				// copy original RGB data to output images
				for (i=0;i<sy;i+=2) {
					for (j=0;j<sx;j+=2) {
						outG[(i*sx+j)*3]=src[i*sx+j];
						outG[((i+1)*sx+(j+1))*3]=src[(i+1)*sx+(j+1)];
						outR[(i*sx+j+1)*3]=src[i*sx+j+1];
						outB[((i+1)*sx+j)*3]=src[(i+1)*sx+j];
					}
				}
				// R channel
				for (i=0;i<sy;i+=2) {
					for (j=0;j<sx-1;j+=2) {
						outR[(i*sx+j)*3]=outR[(i*sx+j+1)*3];
						outR[((i+1)*sx+j+1)*3]=outR[(i*sx+j+1)*3];
						outR[((i+1)*sx+j)*3]=outR[(i*sx+j+1)*3];
					}
				}
				// B channel
				for (i=0;i<sy-1;i+=2)  { //every two lines
					for (j=0;j<sx-1;j+=2) {
						outB[(i*sx+j)*3]=outB[((i+1)*sx+j)*3];
						outB[(i*sx+j+1)*3]=outB[((i+1)*sx+j)*3];
						outB[((i+1)*sx+j+1)*3]=outB[((i+1)*sx+j)*3];
					}
				}
				// using lower direction for G channel

				// G channel
				for (i=0;i<sy-1;i+=2)//every two lines
					for (j=1;j<sx;j+=2)
						outG[(i*sx+j)*3]=outG[((i+1)*sx+j)*3];

				for (i=1;i<sy-2;i+=2)//every two lines
					for (j=0;j<sx-1;j+=2)
						outG[(i*sx+j)*3]=outG[((i+1)*sx+j)*3];

				// copy it for the next line
				for (j=0;j<sx-1;j+=2)
					outG[((sy-1)*sx+j)*3]=outG[((sy-2)*sx+j)*3];

				break;
			case CV_BayerBG2BGR: //-------------------------------------------
			case CV_BayerRG2BGR:
				// copy original data
				for (i=0;i<sy;i+=2) {
					for (j=0;j<sx;j+=2) {
						outB[(i*sx+j)*3]=src[i*sx+j];
						outR[((i+1)*sx+(j+1))*3]=src[(i+1)*sx+(j+1)];
						outG[(i*sx+j+1)*3]=src[i*sx+j+1];
						outG[((i+1)*sx+j)*3]=src[(i+1)*sx+j];
					}
				}
				// R channel
				for (i=0;i<sy;i+=2){
					for (j=0;j<sx-1;j+=2) {
						outR[(i*sx+j)*3]=outR[((i+1)*sx+j+1)*3];
						outR[(i*sx+j+1)*3]=outR[((i+1)*sx+j+1)*3];
						outR[((i+1)*sx+j)*3]=outR[((i+1)*sx+j+1)*3];
					}
				}
				// B channel
				for (i=0;i<sy-1;i+=2) { //every two lines
					for (j=0;j<sx-1;j+=2) {
						outB[((i+1)*sx+j)*3]=outB[(i*sx+j)*3];
						outB[(i*sx+j+1)*3]=outB[(i*sx+j)*3];
						outB[((i+1)*sx+j+1)*3]=outB[(i*sx+j)*3];
					}
				}
				// using lower direction for G channel

				// G channel
				for (i=0;i<sy-1;i+=2)//every two lines
					for (j=0;j<sx-1;j+=2)
						outG[(i*sx+j)*3]=outG[((i+1)*sx+j)*3];

				for (i=1;i<sy-2;i+=2)//every two lines
					for (j=0;j<sx-1;j+=2)
						outG[(i*sx+j+1)*3]=outG[((i+1)*sx+j+1)*3];

				// copy it for the next line
				for (j=0;j<sx-1;j+=2)
					outG[((sy-1)*sx+j+1)*3]=outG[((sy-2)*sx+j+1)*3];

				break;

			default:  //-------------------------------------------
				std::cerr << "Bad Bayer pattern ID: " << type << std::endl;
				return;
				break;
		}

	}

	template<> void setWeight<unsigned char>(const unsigned char& mask, double* w1, double* w2){
		*w1 = mask;
		*w2 = 255 - mask;
		*w1 /= 255.0;
		*w2 /= 255.0;
	}

	template<> cv::Vec3b blend(const cv::Vec3b& pix1,const cv::Vec3b& pix2, double w1, double w2){
		cv::Vec3b val;
		for(size_t i=0;i<3;i++){
			val[i] = static_cast<unsigned char>(w1 * pix1[i] + w2 * pix2[i] + 0.5);
		}
		return val;
	}


	void ClearBorders(unsigned char* dest, int sx, int sy, int w);
	void cvtBayer2BGR_EDGESENSE(const cv::Mat& bayer, cv::Mat& bgr, int type){
		int sx = bayer.cols;
		int sy = bayer.rows;
		const unsigned char* src = bayer.ptr<unsigned char>(0);
		unsigned char* dest = bgr.ptr<unsigned char>(0);
		unsigned char *outR=NULL, *outG=NULL, *outB=NULL;
		register int i,j,idx;
		int dh, dv;
		int tmp;

		// sx and sy should be even
		switch (type) {
			case CV_BayerGR2BGR:
			case CV_BayerBG2BGR:
				outR=&dest[0];
				outG=&dest[1];
				outB=&dest[2];
				break;
			case CV_BayerGB2BGR:
			case CV_BayerRG2BGR:
				outR=&dest[2];
				outG=&dest[1];
				outB=&dest[0];
				break;
			default:
				std::cerr << "Bad Bayer pattern ID: " << type << std::endl;
				return;
				break;
		}

		switch (type) {
			case CV_BayerGR2BGR://---------------------------------------------------------
			case CV_BayerGB2BGR:
				// copy original RGB data to output images
				for (i=0;i<sy;i+=2) {
					for (j=0;j<sx;j+=2) {
						idx = i*sx+j;
						outG[idx*3]=src[idx];
						outR[(idx+1)*3]=src[idx+1];
						idx += sx;
						outB[idx*3]=src[idx];
						outG[(idx+1)*3]=src[idx+1];
					}
				}
				// process GREEN channel
				for (i=3;i<sy-2;i+=2) {
					for (j=2;j<sx-3;j+=2) {
						idx = i*sx+j;
						dh=abs((outB[(idx-2)*3]+outB[(idx+2)*3])/2-outB[idx*3]);
						dv=abs((outB[(idx-2*sx)*3]+outB[(idx+2*sx)*3])/2-outB[(idx)*3]);
						if (dh<dv)
							tmp=(outG[(idx-1)*3]+outG[(idx+1)*3])/2;
						else {
							if (dh>dv)
								tmp=(outG[(idx-sx)*3]+outG[(idx+sx)*3])/2;
							else
								tmp=(outG[(idx-1)*3]+outG[(idx+1)*3]+outG[(idx-sx)*3]+outG[(idx+sx)*3])/4;
						}
						CLIP(tmp,outG[idx*3]);
					}
				}

				for (i=2;i<sy-3;i+=2) {
					for (j=3;j<sx-2;j+=2) {
						idx = i*sx+j;
						dh=abs((outR[(idx-2)*3]+outR[(idx+2)*3])/2-outR[idx*3]);
						dv=abs((outR[(idx-2*sx)*3]+outR[(idx+2*sx)*3])/2-outR[idx*3]);
						if (dh<dv)
							tmp=(outG[(idx-1)*3]+outG[(idx+1)*3])/2;
						else {
							if (dh>dv)
								tmp=(outG[(idx-sx)*3]+outG[(idx+sx)*3])/2;
							else
								tmp=(outG[(idx-1)*3]+outG[(idx+1)*3]+outG[(idx-sx)*3]+outG[(idx+sx)*3])/4;
						} 
						CLIP(tmp,outG[idx*3]);
					}
				}
				// process RED channel
				for (i=0;i<sy-1;i+=2) {
					for (j=2;j<sx-1;j+=2) {
						idx = i*sx+j;
						tmp=outG[idx*3]+(outR[(idx-1)*3]-outG[(idx-1)*3]+
								outR[(idx+1)*3]-outG[(idx+1)*3])/2;
						CLIP(tmp,outR[idx*3]);
					}
				}
				for (i=1;i<sy-2;i+=2) {
					for (j=1;j<sx;j+=2) {
						idx = i*sx+j;
						tmp=outG[idx*3]+(outR[(idx-sx)*3]-outG[(idx-sx)*3]+
								outR[(idx+sx)*3]-outG[(idx+sx)*3])/2;
						CLIP(tmp,outR[idx*3]);
					}
					for (j=2;j<sx-1;j+=2) {
						idx = i*sx+j;
						tmp=outG[idx*3]+(outR[(idx-sx-1)*3]-outG[(idx-sx-1)*3]+
								outR[(idx-sx+1)*3]-outG[(idx-sx+1)*3]+
								outR[(idx+sx-1)*3]-outG[(idx+sx-1)*3]+
								outR[(idx+sx+1)*3]-outG[(idx+sx+1)*3])/4;
						CLIP(tmp,outR[idx*3]);
					}
				}

				// process BLUE channel
				for (i=1;i<sy;i+=2) {
					for (j=1;j<sx-2;j+=2) {
						idx = i*sx+j;
						tmp=outG[idx*3]+(outB[(idx-1)*3]-outG[(idx-1)*3]+
								outB[(idx+1)*3]-outG[(idx+1)*3])/2;
						CLIP(tmp,outB[idx*3]);
					}
				}
				for (i=2;i<sy-1;i+=2) {
					for (j=0;j<sx-1;j+=2) {
						idx = i*sx+j;
						tmp=outG[idx*3]+(outB[(idx-sx)*3]-outG[(idx-sx)*3]+
								outB[(idx+sx)*3]-outG[(idx+sx)*3])/2;
						CLIP(tmp,outB[idx*3]);
					}
					for (j=1;j<sx-2;j+=2) {
						idx = i*sx+j;
						tmp=outG[idx*3]+(outB[(idx-sx-1)*3]-outG[(idx-sx-1)*3]+
								outB[(idx-sx+1)*3]-outG[(idx-sx+1)*3]+
								outB[(idx+sx-1)*3]-outG[(idx+sx-1)*3]+
								outB[(idx+sx+1)*3]-outG[(idx+sx+1)*3])/4;
						CLIP(tmp,outB[idx*3]);
					}
				}
				break;

			case CV_BayerBG2BGR: //---------------------------------------------------------
			case CV_BayerRG2BGR:
				// copy original RGB data to output images
				for (i=0;i<sy;i+=2) {
					for (j=0;j<sx;j+=2) {
						idx = i*sx+j;
						outB[idx*3]=src[idx];
						outG[(idx+1)*3]=src[idx+1];

						idx += sx;
						outG[idx*3]=src[idx];
						outR[(idx+1)*3]=src[idx+1];
					}
				}
				// process GREEN channel
				for (i=2;i<sy-2;i+=2) {
					for (j=2;j<sx-3;j+=2) {
						idx = i*sx+j;
						dh=abs((outB[(idx-2)*3]+outB[(idx+2)*3])/2-outB[idx*3]);
						dv=abs((outB[(idx-2*sx)*3]+outB[(idx+2*sx)*3])/2-outB[idx*3]);
						if (dh<dv)
							tmp=(outG[(idx-1)*3]+outG[(idx+1)*3])/2;
						else {
							if (dh>dv)
								tmp=(outG[(idx-sx)*3]+outG[(idx+sx)*3])/2;
							else
								tmp=(outG[(idx-1)*3]+outG[(idx+1)*3]+outG[(idx-sx)*3]+outG[(idx+sx)*3])/4;
						}
						CLIP(tmp,outG[(idx)*3]);
					}
				}
				for (i=3;i<sy-3;i+=2) {
					for (j=3;j<sx-2;j+=2) {
						idx = i*sx+j;
						dh=abs((outR[(idx-2)*3]+outR[(idx+2)*3])/2-outR[(idx)*3]);
						dv=abs((outR[(idx-2*sx)*3]+outR[(idx+2*sx)*3])/2-outR[(idx)*3]);
						if (dh<dv)
							tmp=(outG[(idx-1)*3]+outG[(idx+1)*3])/2;
						else {
							if (dh>dv)
								tmp=(outG[(idx-sx)*3]+outG[(idx+sx)*3])/2;
							else
								tmp=(outG[(idx-1)*3]+outG[(idx+1)*3]+outG[(idx-sx)*3]+outG[(idx+sx)*3])/4;
						}
						CLIP(tmp,outG[(idx)*3]);
					}
				}
				// process RED channel
				for (i=1;i<sy-1;i+=2) { // G-points (1/2)
					for (j=2;j<sx-1;j+=2) {
						idx = i * sx + j;
						tmp=outG[(i*sx+j)*3]+(outR[(i*sx+j-1)*3]-outG[(i*sx+j-1)*3]+
								outR[(i*sx+j+1)*3]-outG[(i*sx+j+1)*3])/2;
						CLIP(tmp,outR[(i*sx+j)*3]);
					}
				}
				for (i=2;i<sy-2;i+=2)  {
					for (j=1;j<sx;j+=2) { // G-points (2/2)
						tmp=outG[(i*sx+j)*3]+(outR[((i-1)*sx+j)*3]-outG[((i-1)*sx+j)*3]+
								outR[((i+1)*sx+j)*3]-outG[((i+1)*sx+j)*3])/2;
						CLIP(tmp,outR[(i*sx+j)*3]);
					}
					for (j=2;j<sx-1;j+=2) { // B-points
						tmp=outG[(i*sx+j)*3]+(outR[((i-1)*sx+j-1)*3]-outG[((i-1)*sx+j-1)*3]+
								outR[((i-1)*sx+j+1)*3]-outG[((i-1)*sx+j+1)*3]+
								outR[((i+1)*sx+j-1)*3]-outG[((i+1)*sx+j-1)*3]+
								outR[((i+1)*sx+j+1)*3]-outG[((i+1)*sx+j+1)*3])/4;
						CLIP(tmp,outR[(i*sx+j)*3]);
					}
				}

				// process BLUE channel
				for (i=0;i<sy;i+=2) {
					for (j=1;j<sx-2;j+=2) {
						tmp=outG[(i*sx+j)*3]+(outB[(i*sx+j-1)*3]-outG[(i*sx+j-1)*3]+
								outB[(i*sx+j+1)*3]-outG[(i*sx+j+1)*3])/2;
						CLIP(tmp,outB[(i*sx+j)*3]);
					}
				}
				for (i=1;i<sy-1;i+=2) {
					for (j=0;j<sx-1;j+=2) {
						idx = i*sx+j;
						tmp=outG[idx*3]+(outB[(idx-sx)*3]-outG[(idx-sx)*3]+
								outB[(idx+sx)*3]-outG[(idx+sx)*3])/2;
						CLIP(tmp,outB[idx*3]);
					}
					for (j=1;j<sx-2;j+=2) {
						idx = i*sx+j;
						tmp=outG[idx*3]+(outB[(idx-sx-1)*3]-outG[(idx-sx-1)*3]+
								outB[(idx-sx+1)*3]-outG[(idx-sx+1)*3]+
								outB[(idx+sx-1)*3]-outG[(idx+sx-1)*3]+
								outB[(idx+sx+1)*3]-outG[(idx+sx+1)*3])/4;
						CLIP(tmp,outB[idx*3]);
					}
				}
				for (i=1;i<sy-1;i+=2) {
					for (j=0;j<sx-1;j+=2) {
						tmp=outG[(i*sx+j)*3]+(outB[((i-1)*sx+j)*3]-outG[((i-1)*sx+j)*3]+
								outB[((i+1)*sx+j)*3]-outG[((i+1)*sx+j)*3])/2;
						CLIP(tmp,outB[(i*sx+j)*3]);
					}
					for (j=1;j<sx-2;j+=2) {
						tmp=outG[(i*sx+j)*3]+(outB[((i-1)*sx+j-1)*3]-outG[((i-1)*sx+j-1)*3]+
								outB[((i-1)*sx+j+1)*3]-outG[((i-1)*sx+j+1)*3]+
								outB[((i+1)*sx+j-1)*3]-outG[((i+1)*sx+j-1)*3]+
								outB[((i+1)*sx+j+1)*3]-outG[((i+1)*sx+j+1)*3])/4;
						CLIP(tmp,outB[(i*sx+j)*3]);
					}
				}
				break;
			default: //---------------------------------------------------------
				std::cerr << "Bad Bayer pattern ID: " << type << std::endl;
				return;
				break;
		}

		ClearBorders(dest, sx, sy, 3);
	}

	void ClearBorders(unsigned char* dest, int sx, int sy, int w)
	{
		int i,j;

		// repeat the edge at cv::Rect(2,2,sx-4,sy-4);
		i=3*sx*w-1;
		j=3*sx*(sy-1)-1;
		int idx;
		int offset = 3*sx*w;
		int inv = sx*sy*w-1;
		for(j=0,idx=0;j<3;j++){
			for(i=0;i<sx*w;i++,idx++){
				dest[idx] = dest[offset + i];
				dest[inv - idx] = dest[inv - ( offset + i )];
			}
		}

		inv = sx * w - 1;
		for(j=0;j<sy;j++){
			offset = j * sx * w;
			for(i=0;i<3 * w;i++){
				dest[offset + i] = dest[offset + 3 * w + i%w];
				dest[offset + inv - i] = dest[offset + inv - (3 * w + i%w)];
			}
		}
	}


	void edge_difference(const cv::Mat& src1, const cv::Mat& src2, cv::Mat& edge1, cv::Mat& edge2, double canny_thresh1, double canny_thresh2, int aperture_size, int dilate_size){
		assert(src1.size()==src2.size());
		cv::Mat gray1,gray2;
		if(src1.channels()==1){
			gray1 = src1.clone();
		}
		else{
			gray1 = cv::Mat(src1.size(),CV_8UC1);
			cv::cvtColor(src1,gray1,CV_BGR2GRAY);
		}
		if(src2.channels()==1){
			gray2 = src2.clone();
		}
		else{
			gray2 = cv::Mat(src2.size(),CV_8UC1);
			cv::cvtColor(src2,gray2,CV_BGR2GRAY);
		}

		cv::Mat _edge1 = cv::Mat(src1.size(),CV_8UC1);
		cv::Mat _edge2 = cv::Mat(src2.size(),CV_8UC1);
		cv::Canny(gray1,_edge1, canny_thresh1, canny_thresh2, aperture_size);
		cv::Canny(gray2,_edge2, canny_thresh1, canny_thresh2, aperture_size);
#ifdef DEBUG_SKLUTILS
		cv::namedWindow("edge1",0);
		cv::namedWindow("edge2",0);
		cv::imshow("edge1",_edge1);
		cv::imshow("edge2",_edge2);
#endif

		cv::Size kernel_size(dilate_size,dilate_size);
		cv::Mat dick_edge1 = cv::Mat(src1.size(),CV_8UC1);
		cv::Mat dick_edge2 = cv::Mat(src2.size(),CV_8UC1);
		cv::blur(_edge1,dick_edge1,kernel_size);
		cv::threshold(dick_edge1,dick_edge1,0,255,CV_THRESH_BINARY);
		cv::blur(_edge2,dick_edge2,kernel_size);
		cv::threshold(dick_edge2,dick_edge2,0,255,CV_THRESH_BINARY);

		edge1 = _edge1 - dick_edge2;
		edge2 = _edge2 - dick_edge1;
///
/*		cv::imwrite("src.png",src1);
		cv::imwrite("fg_edge.png",edge1);
		cv::imwrite("src_fg_edge.png",_edge1);
		cv::imwrite("bg_edge_dick.png",dick_edge2);

		cv::imwrite("bg.png",src2);
		cv::imwrite("bg_edge.png",edge2);
		cv::imwrite("src_bg_edge.png",_edge2);
		cv::imwrite("fg_edge_dick.png",dick_edge1);
*/
///
	}

}// namespace skl

template<typename ValType> bool _checkMat(
		const ValType val,
		const ValType condition,
		const ValType skipConditionVal, 
		const std::string& debug_comment){
	if(condition == skipConditionVal) return true;// skip
	if(val == condition) return true;
#ifdef DEBUG
	std::cerr << "ERROR: " << debug_comment << std::endl;
	std::cerr << "       value '" << val << "' does not equal to '"<<condition<<"'." << std::endl;
#endif
	return false;
}
bool checkMat(const cv::Mat& mat, int depth,int channels,cv::Size size){
	bool result = true;
	assert(!mat.empty());
	result &= _checkMat(mat.depth(),depth,-1,"depth does not fit.");
	result &= _checkMat(mat.channels(),channels,0,"channels does not fit.");
	result &= _checkMat(mat.cols,size.width,0,"image width does not fit.");
	result &= _checkMat(mat.rows,size.height,0,"image height does not fit.");
	return result;
}

