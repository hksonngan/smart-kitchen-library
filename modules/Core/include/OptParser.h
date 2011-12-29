/*!
 * @file OptParser.h
 * @author a_hasimoto
 * @date Date Created: 2011/Dec/27
 * @date Last Change:2011/Dec/29.
 */
#ifndef __SKL_OPT_PARSER_H__
#define __SKL_OPT_PARSER_H__

#include <map>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <cassert>
#include "sklstring.h"
namespace skl{

/*!
 * @class Parse Option from CommandLine
 */
class OptParser{

	public:
		OptParser(const std::string& conf_deliminator=":");
		virtual ~OptParser();
		void on(const std::string& long_form, const std::string& expression, const std::string& explanation);
		void on(const std::string& short_form, const std::string& long_form,const std::string& expression, const std::string& explanation);
		template<class T> bool get(const std::string& var_name, T* var)const;
		template<class T> bool get_vector(const std::string& var_name, std::vector<T>* var, const std::string& deliminator=":")const;
		std::vector<std::string> parse(int argc,char* argv[]);
		void usage()const;
		bool help()const;
	protected:
		std::map<std::string,std::string> explanations;
		std::map<std::string, std::string> option_values;
		std::map<std::string, std::string> short_long_map;

		std::string conf_deliminator;

		int checkType(const std::string& str)const;
		static void parse_short_form(const std::string& form,std::string* var_name);
		static void parse_long_form(const std::string& form,std::string* var_name);

		static std::string get_short_form(const std::string& str);
		static std::string get_long_form(const std::string& str);
		static std::string make_usage(
				const std::string& short_name,
				const std::string& var_name,
				const std::string& expression,
				const std::string& explanation);
		bool _help;
	private:
		int parse_error(const std::string& str)const;
		static const int true_string_num;
		static const std::string true_strings[];
		static const int false_string_num;
		static const std::string false_strings[];
};

/*
 * @class parser for indivisual valiables (interface)
 * */
class OptParserAtom{
	public:
		OptParserAtom(){
			next = atom_top;
			atom_top = this;
		}
		virtual ~OptParserAtom(){};
		virtual void on(OptParser* parser)=0;
		virtual void get(OptParser* parser)=0;
		OptParserAtom* next;
		static OptParserAtom* atom_top;
};



template<class T> bool OptParser::get(const std::string& var_name,T* var)const{
	std::stringstream ss;
	std::map<std::string,std::string>::const_iterator pp = option_values.find(var_name);
	if(option_values.end()==pp) return false;
	if("0"==pp->second) return true;
	ss << pp->second;
	ss >> *var;
	return true;
}

template<class T> bool OptParser::get_vector(const std::string& var_name, std::vector<T>* var,const std::string& deliminator)const{
	std::map<std::string,std::string>::const_iterator pp = option_values.find(var_name);
	if(option_values.end()==pp) return false;
	if("0"==pp->second) return true;

	std::vector<std::string> buf;
	buf = split(pp->second,deliminator);
	var->resize(buf.size());
	for(size_t i=0;i<buf.size();i++){
		std::stringstream ss;
		ss << buf[i];
		ss >> var->at(i);
	}
	return true;
}


#define opt_on_bool(VAR,SHORT_FORM,EXPLANATION)\
	opt_on(bool, VAR, false, SHORT_FORM, "", EXPLANATION)

#define opt_on(TYPE, VAR, DEFAULT_VAL, SHORT_FORM, EXPRESSION, EXPLANATION)\
	TYPE VAR(DEFAULT_VAL);\
generate_atomic_parser(TYPE,VAR,SHORT_FORM, EXPRESSION, EXPLANATION);\

#define opt_on_vector(TYPE, VAR, SHORT_FORM, EXPRESSION, EXPLANATION, DELIM)\
	generate_atomic_parser_vector(TYPE,VAR,SHORT_FORM, EXPRESSION, EXPLANATION, DELIM);\


#define opt_parse(PARSER,ARGC, ARGV, ARGS)\
	skl::OptParserAtom* __opt_parser_func_list__ = skl::OptParserAtom::atom_top;\
while(__opt_parser_func_list__!=NULL){\
	__opt_parser_func_list__->on(& PARSER);\
	__opt_parser_func_list__ = __opt_parser_func_list__->next;\
}\
ARGS = PARSER.parse(ARGC,ARGV);\
__opt_parser_func_list__ = skl::OptParserAtom::atom_top;\
while(__opt_parser_func_list__!=NULL){\
	__opt_parser_func_list__->get(& PARSER);\
	__opt_parser_func_list__ = __opt_parser_func_list__->next;\
}


#define generate_atomic_parser(TYPE,VAR, SHORT_FORM, EXPRESSION, EXPLANATION)\
	namespace skl{\
		class OptParserAtom_##VAR : public OptParserAtom{\
			public:\
				   OptParserAtom_##VAR(TYPE* VAR):VAR(VAR){}\
			~OptParserAtom_##VAR(){}\
			void on(OptParser* parser){\
				parser->on(SHORT_FORM, std::string("--")+#VAR,EXPRESSION,EXPLANATION);\
			}\
			void get(OptParser* parser){\
				parser->get<TYPE>(#VAR , VAR);\
			}\
			protected:\
					  TYPE* VAR;\
		};\
	}\
skl::OptParserAtom_##VAR __opt_parser_atom_##VAR(&VAR)


#define generate_atomic_parser_vector(TYPE,VAR, SHORT_FORM, EXPRESSION, EXPLANATION, DELIM)\
	namespace skl{\
		class OptParserAtom_##VAR : public OptParserAtom{\
			public:\
				   OptParserAtom_##VAR(std::vector<TYPE>* VAR):VAR(VAR){}\
			~OptParserAtom_##VAR(){}\
			void on(OptParser* parser){\
				parser->on(SHORT_FORM, std::string("--")+#VAR,EXPRESSION,EXPLANATION);\
			}\
			void get(OptParser* parser){\
				parser->get_vector<TYPE>(#VAR , VAR, DELIM);\
			}\
			protected:\
					  std::vector<TYPE>* VAR;\
		};\
	}\
skl::OptParserAtom_##VAR __opt_parser_atom_##VAR(&VAR)

} // skl

#endif //__SKL_OPT_PARSER_H__

