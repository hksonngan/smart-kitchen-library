/*!
 * @file   Time.h
 * @author Takahiro Suzuki
 * @date Last Change:2012/Jan/13.
 **/

#ifndef __SKL_TIME_H__
#define __SKL_TIME_H__

/**** C++ ****/
#include <iostream>
#include <string>

/**** C ****/
#include <ctime>
#include <cmath>
#include <cassert>
#include <cstring>

/**** 環境依存 ****/
/**** WIN32 ****/
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX 1
#endif
#include <windows.h>
/**** Linux ****/
#elif __linux__
#include <sys/time.h>
#endif	/**** ifdef linux ****/

/**** Mmpl ****/
#include "Serializable.h"
#include "TimeFormatException.h"
/**** typedef ****/
#ifdef _WIN32
typedef long suseconds_t;
#endif	/**** ifdef WIN32 ****/

namespace skl{
	class TimeFormatter;
	/*!
	 * @brief Time
	 * @brief 非負で1msec単位のタイムスタンプを扱えるクラス
	 * */
	class Time : public Serializable{
		public:
			/*!
			 * @brief 現在時刻を表すオブジェクトを返す静的メソッド
			 * @return 現在時刻を表す Time オブジェクト
			 */
			static Time now(){
				// 環境依存
#ifdef WIN32
				SYSTEMTIME systime;
				GetLocalTime(&systime);

				struct tm tm;
				tm.tm_year = systime.wYear-1900;
				tm.tm_mon = systime.wMonth-1;
				tm.tm_mday =  systime.wDay;
				tm.tm_hour = systime.wHour;
				tm.tm_min = systime.wMinute;
				tm.tm_sec = systime.wSecond;
				struct timeval tv;
				tv.tv_sec=mktime(&tm);
				tv.tv_usec=systime.wMilliseconds*1000;
#elif __linux__
				struct timeval tv;
				gettimeofday(&tv,NULL);
#endif	/**** ifdef linux ****/
				return Time(static_cast<int>(tv.tv_sec),tv.tv_usec/1000,tv.tv_usec%1000);
			};

			Time();
			Time(long nSec,int nMSec,int nUSec=0);
			Time(const Time& other);	//コピーコンストラクタ
			Time(int YYYY,int MM,int DD,int hh, int mm, int ss=0, int mmm=0);	// Set
			~Time();

			Time& operator=(const Time& other);

///////////////////////////////////////////////////////////
// Accessor
			int getYear()const;
			int getMonth()const;
			int getDay()const;
			int getHour()const;
			int getMinute()const;
			int getSecond()const;
			int getMilliSecond()const;

///////////////////////////////////////////////////////////
// TimeFormatter
			void parseString(const std::string& strxx) throw (TimeFormatException);
			std::string toString()const;
			void parseString(const std::string& str,TimeFormatter* timeformatter) throw (TimeFormatException);	// 指定したTimeFormatter型で自分のオブジェクトに変換する
			std::string toString(TimeFormatter* timeformatter)const;	// 指定したTimeFormatter型で自分のオブジェクトをStringに変換する

			void setDefaultTimeFormatter(const TimeFormatter* tf);
////////////////////////////////////////////////////////////
// 演算子オーバーロード
			// 比較演算子
			bool operator==(const Time& other) const;
			bool operator!=(const Time& other) const;
			bool operator<(const Time& other) const;
			bool operator>(const Time& other) const;
			bool operator<=(const Time& other) const;
			bool operator>=(const Time& other) const;

			// 加減演算
			long long operator-(const Time& other) const;
/*			Time operator+(const Time& other) const;
			Time operator-(const Time& other) const;
			Time& operator+=(const Time& other);
			Time& operator-=(const Time& other);
*/			Time operator+(long long MSec) const;
			Time operator-(long long MSec) const;
			Time& operator+=(long long MSec);
			Time& operator-=(long long MSec);
			long long operator%(long long MSec);
			//
			void print(std::ostream& out)const;

////////////////////////////////////////////////////////////
// Serialzable 純粋仮想関数
			long _buf_size()const;
			void _serialize();
			void _deserialize(const char* buf,long buf_size);
		private:
			// TimeFormatter
			TimeFormatter* defaulttf;
#ifdef _WIN32
			struct timeval{
				time_t tv_sec;
				suseconds_t tv_usec; 
			}tv_st;
#endif	// ifdef WIN32
#ifdef __linux__
			struct timeval tv_st;	//timeval構造体
#endif	// ifdef linux

	};
	// 出力演算子
	std::ostream& operator<<(std::ostream& lhs, const Time& rhs);
} // namespace skl
#endif
