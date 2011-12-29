/*!
 * @file StringSplitter.h
 * @author 橋本敦史
 * @date Last Change:2011/Jul/18.
 */
#ifndef __STRINGSPRITTER_H__
#define __STRINGSPRITTER_H__

#include <vector>
#include <string>

namespace mmpl{

/*!
 * @class 指定された情報に従って、文字列を分割しvector<string>にするクラス
 */
class StringSplitter{

	public:
		StringSplitter(char deliminator=' ',bool validateDoubleQuot=true);
		virtual ~StringSplitter();
		void apply(const std::string& str,std::vector<std::string>* tar)const;
	protected:
		char deliminator;
		bool validateDoubleQuot;
		static void skipFrontSpace(std::string* str);
		static void skipRearSpace(std::string* str);
		bool jointDoubleQuotElement(const std::string& input_str,std::string* dquot_str)const;
	private:
		
};

} // namespace skl
#endif // __STRINGSPRITTER_H__
