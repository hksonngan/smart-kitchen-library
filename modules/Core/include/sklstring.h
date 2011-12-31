#ifndef __SKL__STRING__H__
#define __SKL__STRING__H__

#include <vector>
#include <string>
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

	bool parse_conffile(const std::string& filename, std::map<std::string,std::string>& param_map, const std::string& deliminator=":");

	template<class T> bool convert(const std::string& src, T* dest){
		std::stringstream ss;
		ss << src;
		ss >> *dest;
		return true;
	};
	template<> bool convert(const std::string& src, bool* dest);
	template<> bool convert(const std::string& src, std::string* dest);

	template<class T> bool convert_vector(const std::string& src, std::vector<T>* dest, const std::string& deliminator=":", int length=-1);

}
#endif // __SKL__STRING_H__
