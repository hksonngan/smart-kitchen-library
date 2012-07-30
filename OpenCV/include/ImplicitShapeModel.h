/*!
 * @file ImplicitShapeModel.h
 * @author a_hasimoto
 * @date Date Created: 2012/Jul/11
 * @date Last Change:2012/Jul/26.
 */
#ifndef __SKL_IMPLICIT_SHAPE_MODEL_H__
#define __SKL_IMPLICIT_SHAPE_MODEL_H__

#include <cv.h>
#include <list>

namespace skl{

class ISMEntry{
	public:
		ISMEntry(
				const cv::Point2f& pt = cv::Point2f(0.f,0.f),
				const std::vector<float>& scale = std::vector<float>(2,1.f),
				int class_response = 0,
				const cv::Mat& patch = cv::Mat()
				);
		ISMEntry(const ISMEntry& others);
		~ISMEntry();
		bool isSameEvidence(const ISMEntry& other)const;
		bool merge(const ISMEntry& other);
		inline const cv::Point2f& pt()const{return _pt;}
		inline int class_response()const{return _class_response;}
		inline const cv::Mat& patch()const{return _patch;}
		void read(std::istream& in);
		void write(std::ostream& out)const;
	protected:
		cv::Point2f _pt;
		int _class_response;
		cv::Mat _patch;
};

class ISMEntries{
	public:
		ISMEntries(const cv::Size& max_dist_size=cv::Size(128,128),int roughness = 16);
		~ISMEntries();
		void insert(const ISMEntry& entry);
		std::list<size_t> getIndex(const cv::Point2f& pt,float roughness)const;
		inline std::vector<ISMEntry>::const_iterator begin()const { return entries.begin();}
		inline std::vector<ISMEntry>::const_iterator end()const{ return entries.end();}
		inline size_t size()const{return entries.size();}
		inline bool empty()const{return entries.empty();}
		inline const ISMEntry& operator[](size_t idx)const{return entries[idx];}
		inline ISMEntry& operator[](size_t idx){return entries[idx];}
		void clear();
		int _roughness;//>set,get
	protected:
		std::vector<ISMEntry> entries;
		std::vector<std::vector<std::list<size_t> > > index;
		std::list<size_t>* _getIndex(const cv::Point2f& pt);
		void getIndex(const cv::Point2f& pt, int& x, int&y)const;
		cv::Size _max_dist_size;
};

/*!
 * @class ImplicitShapeModel(Under Construction!)
 * @brief Learn/Detect Object as implicit shape model
 */
class ImplicitShapeModel{
	public:
		enum FeatureType{
			NORMALIZED = 0,
			POSITIVE = 1,
			OTHER = 2
		};
	public:
		ImplicitShapeModel(
				const cv::Mat& vocaburary=cv::Mat(),
				float entry_threshold=0.5,
				float hypothesis_threshold=0.5,
				float kapper1 = 0.8,
				float kapper2 = 0.5,
				float object_kernel_ratio = 0.05,
				float std_size=1,
				FeatureType feature_type = NORMALIZED);
		virtual ~ImplicitShapeModel();
		bool add(
				const cv::Mat& features,
				const std::vector<cv::KeyPoint>& feature_locations,
				const std::vector<cv::Mat>* patches, //! additional. if object region is not available in training phase, patches == NULL.
				cv::KeyPoint& shape_location,
				int class_response,
				std::vector<std::vector<ISMEntry> >* current_occurrences=NULL);

		bool predict(
				const cv::Mat& features,
				const std::vector<cv::KeyPoint>& feature_points,
				std::vector<cv::KeyPoint>& locations,
				std::vector<float>& scales,
				std::vector<std::map<int,float> >& response_likelihoods,
				std::map<int, cv::Mat>* voting_images=NULL,
				const cv::Size& size = cv::Size(0,0))const;// not implemented

		bool predict(
				const cv::Mat& features,
				const std::vector<cv::KeyPoint>& feature_points,
				const cv::KeyPoint& location,
				float kernel_size,
				std::map<int,float> & response_likelihoods,
				std::map<int, cv::Mat>* backprojections=NULL,
				const cv::Size& size = cv::Size(0,0))const;

		void MDLVerification();// not implemented

		void release();

		bool read(const std::string& filename);
		bool read_header(const std::string& filename);
		bool read_entries(const std::string& filename);
		bool write(const std::string& filename)const;
		bool write_header(const std::string& filename)const;
		bool write_entries(const std::string& filename)const;

		cv::Mat visualize(
				const cv::Mat& base_image,
				const cv::Mat& features,
				const std::vector<cv::KeyPoint>& keypoints,
				const std::vector<std::vector<ISMEntry> >& occurrences,
				const std::map<int,cv::Scalar>& word_color_map)const;

		inline float entry_threshold()const{return _entry_threshold;}
		inline void entry_threshold(float __entry_threshold){_entry_threshold = __entry_threshold;}
		inline float standard_size()const{return _std_size;}
		inline void standard_size(float std_size){_std_size = std_size;}
		void vocaburary(const cv::Mat& vocaburary);
		inline cv::Mat vocaburary()const{return _vocaburary.t();}
		inline size_t wordNum()const{return occurrences.size();}
		inline float hypothesis_threshold()const{return _hypothesis_threshold;}
		inline void hypothesis_threshold(float __hypothesis_threshold){_hypothesis_threshold = __hypothesis_threshold;}
		inline float kapper1()const{return _kapper1;}
		inline void kapper1(float __kapper1){_kapper1 = __kapper1;}
		inline float kapper2()const{return _kapper2;}
		inline void kapper2(float __kapper2){_kapper2 = __kapper2;}
		inline float object_kernel_ratio()const{return _object_kernel_ratio;}
		inline void object_kernel_ratio(float __object_kernel_ratio){_object_kernel_ratio = __object_kernel_ratio;}
		inline FeatureType feature_type()const{return _feature_type;}
		inline void feature_type(FeatureType feature_type){_feature_type = feature_type;getNorm(_vocaburary.t(),v_norm);}
	protected:
		cv::Mat _vocaburary;
		float _entry_threshold;
		float _hypothesis_threshold;
		float _kapper1;
		float _kapper2;
		float _object_kernel_ratio;
		float _std_size;
		FeatureType _feature_type;

		bool hasVocaburary;
		std::vector<ISMEntries> occurrences;
		std::vector<float> v_norm;
		std::map<int,std::vector<size_t> > word_hit_num;
//		std::map<int,size_t> sample_num;

		bool getSimilarity(const cv::Mat& features, cv::Mat& similarity)const;
		void getNorm(const cv::Mat& features, std::vector<float>& norm)const;
		static cv::Point2f getRelativeLocation(
				const cv::KeyPoint& v1,
				const cv::KeyPoint& v2, 
				float scale1,
				float scale2);
		inline static cv::Point2f getRelativeLocation(
				const cv::KeyPoint& v1,
				const cv::KeyPoint& v2,
				float scale){
			return getRelativeLocation(v1,v2,scale,scale);
		}

	private:
		bool read_header(std::istream& in);
		bool read_entries(std::istream& in);
		bool write_header(std::ostream& out)const;
		bool write_entries(std::ostream& out)const;
};

} // skl

#endif // __SKL_IMPLICIT_SHAPE_MODEL_H__
