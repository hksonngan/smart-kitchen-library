/*!
 * @file StopWatch.h
 * @author a_hasimoto
 * @date Date Created: 2012/Jan/06
 * @date Last Change:2012/Jan/06.
 */
#ifndef __SKL_STOP_WATCH_H__
#define __SKL_STOP_WATCH_H__

#include <vector>
#include <string>
#include "sklTime.h"
#include "TimeInterval.h"

namespace skl{

	/*!
	 * @class 実行時間を計測するクラス(DEBUG or STOP_WATCHが定義されている時のみ動作)
	 */
	class StopWatch{

		public:
			StopWatch();
			virtual ~StopWatch();

			// very basic functions as a stop watch
			void start();
			TimeInterval stop();
			void reset();

			// additional functions to record laps
			const std::vector<TimeInterval>& lap_time()const{return lap_times;}
			TimeInterval lap_time(size_t i)const;
			TimeInterval lap();
			size_t record_num()const;

			// alias for reset
			void clear(){reset();}
		protected:
			Time base_time;
			std::vector<TimeInterval> lap_times;
		private:
	};




} // skl

#endif // __SKL_STOP_WATCH_H__

