/****************************************************************
* File: args.h
* Created Time: 五  5/ 1 13:16:09 2020
* Saying: Keep on going never give up! Believe in yourself.
****************************************************************/
#pragma once

#include <string>

namespace fasttext {

class Args{
	public:
		Args();
		int maxn;
		int minn;
		int bucket;
		int verbose;
		int minCount;
		int minCountLabel;
		int wordNgrams;
		std::string label;
		double t;

};
Args::Args(){
	maxn = 6;
	minn = 3;
	bucket = 2000000;
	verbose = 0;
	//minCount = 5;
	minCount = 0;
	minCountLabel = 0;
	label = "__label__";
	wordNgrams = 1;
	t = 1e-4;
}


}
