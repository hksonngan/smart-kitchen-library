#ifndef __SKL__STRING__H__
#define __SKL__STRING__H__

#include <vector>
#include <string>
#include <sstream>
#include <map>
#include <algorithm>

namespace skl{
	const std::string true_strings[] = {"1","true","TRUE","True","on","ON","On"};
	static const int true_string_num = 7;

	const std::string false_strings[] = {"0","false","FALSE","False","off","OFF","Off",""};
	static const int false_string_num = 8;

	std::string strip(const std::string& str);
	std::vector<std::string> split(const std::string& str, const std::string& deliminator, int length=-1);
	std::string join(const std::vector<std::string>& buf, const std::string& separator);
	std::vector<std::string> split_strip(const std::string& str, const std::string& deliminator, int length=-1);

	bool parse_conffile(std::istream& in, std::map<std::string,std::string>& param_map, const std::string& deliminator=":");
	bool parse_conffile(const std::string& filename, std::map<std::string,std::string>& param_map, const std::string& deliminator=":");

	template<class T> bool convert(const std::string& src, T* dest){
		std::stringstream ss;
		ss.str(src);
		ss >> *dest;
		return true;
	};
	template<> bool convert(const std::string& src, bool* dest);
	template<> bool convert(const std::string& src, std::string* dest);

	template<class T,class Container> bool convert2container(const std::string& src, Container* dest, const std::string& deliminator=":", int length=-1){
		std::vector<std::string> buf = split(src,deliminator,length);
		std::vector<T> buf2(buf.size());
		for(size_t i=0;i<buf.size();i++){
			if(!convert<T>(buf[i],&buf2[i])){
				return false;
			}
		}
		dest->clear();
		dest->insert(dest->begin(),buf2.begin(),buf2.end());
		return true;
	}
	std::string DirName(const std::string& filepath);
	std::string ExtName(const std::string& filepath);
	std::string BaseName(const std::string& filepath,const std::string& extname="");
} // namespace skl
#endif // __SKL__STRING_H__
