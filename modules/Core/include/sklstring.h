#ifndef __SKL__STRING__H__
#define __SKL__STRING__H__

#include <vector>
#include <string>
#include <map>

namespace skl{

	std::string strip(const std::string& str);
	std::vector<std::string> split(const std::string& str, const std::string& deliminator, int length=-1);
	std::string join(const std::vector<std::string>& buf, const std::string& separator);
	std::vector<std::string> split_strip(const std::string& str, const std::string& deliminator, int length=-1);

	bool parse_conffile(const std::string& filename, std::map<std::string,std::string>& param_map, const std::string& deliminator=":");

}
#endif // __SKL__STRING_H__
