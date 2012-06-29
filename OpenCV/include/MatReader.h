/*!
 * @file MatReader.h
 * @author a_hasimoto
 * @date Date Created: 2012/Jun/29
 * @date Last Change:2012/Jun/29.
 */
#ifndef __SKL_MAT_READER_H__
#define __SKL_MAT_READER_H__
#include <cv.h>
#include <fstream>
#include "skl.h"
namespace skl{

/*!
 * @class MatReader
 * @brief read cv::Mat data from file, which was output by operator<<(). (only for 1 channel matrix)
 */
template<class ElemType=float,int CV_TYPE=CV_32F> class MatReader{
	public:
		MatReader();
		virtual ~MatReader();
		static bool read(const std::string& filename,cv::Mat& mat);
		static bool read(std::istream& in,cv::Mat& mat);
	protected:
		
	private:
		
};


/*!
 * @brief デフォルトコンストラクタ
 */
template<class ElemType,int CV_TYPE> MatReader<ElemType,CV_TYPE>::MatReader(){

}

/*!
 * @brief デストラクタ
 */
template<class ElemType,int CV_TYPE> MatReader<ElemType,CV_TYPE>::~MatReader(){

}

template<class ElemType,int CV_TYPE> bool MatReader<ElemType,CV_TYPE>::read(const std::string& filename,cv::Mat& mat){
	std::ifstream fin;
	fin.open(filename.c_str());
	if(!fin){
		return false;
	}
	read(fin,mat);
	fin.close();
	return true;
}

template<class ElemType,int CV_TYPE> bool MatReader<ElemType,CV_TYPE>::read(std::istream& in,cv::Mat& mat){
	std::string str;
	std::vector<std::string> buf;
	int row_num = 1;
	bool isAllocMat(false);
	while(!in.eof()){
		std::getline(in,str);
		if(str[0]=='['){
			str = str.substr(1,str.npos);
		}
		buf = skl::split_strip(str,",");
//		std::cerr << "buf.size() = " << buf.size() << std::endl;
//		std::cerr << row_num << std::endl;
		if(!isAllocMat){
			if(buf.size()!=mat.cols){
				mat = cv::Mat(cv::Size(buf.size(),row_num),CV_TYPE);
			}
			isAllocMat = true;
		}
		if(mat.rows < row_num){
			mat.resize(row_num);
		}

//		std::cerr << "mat.size() = " << mat.cols << "x" << mat.rows << std::endl;
		for(int x=0;x<mat.cols;x++){
			mat.at<ElemType>(row_num-1,x) = static_cast<ElemType>(atof(buf[x].c_str()));
		}
		row_num++;
	}

	return true;
}

} // skl

#endif // __SKL_MAT_READER_H__

