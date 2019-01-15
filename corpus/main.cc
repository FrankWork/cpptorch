#include <iostream>
#include <string>
#include <fstream>
#include <cstdio>

// fastText	dictinoary.cc
// NOTE: 以空格作为分隔符，对中文支持不好
bool readWord(std::istream& in, std::string& word) 
{
	int c;
	std::streambuf& sb = *in.rdbuf();
	word.clear();
	while ((c = sb.sbumpc()) != EOF) {
		if (c == ' ' || c == '\n' || c == '\r' || c == '\t' || c == '\v' ||
				c == '\f' || c == '\0') {
			if (word.empty()) {
				if (c == '\n') {
					// word += EOS; // EOS = "</s>";
					return true;
				}	 
				continue;
			} else {
				if (c == '\n')
					sb.sungetc();
				return true;
			}	 
		}	 
		word.push_back(c);
	}
	// trigger eofbit
	in.get();
	return !word.empty();
}

int countWords(std::string& finName){
	std::ifstream fin(finName);
	std::string buf;
	int nWords = 0;

	while(readWord(fin, buf)){
		nWords++;
	}
	return nWords;
}

int countLines(std::string& finName){
	std::ifstream fin(finName);
	std::string buf;
	int nLines=0;

	while(getline(fin, buf)){
		nLines++;
	}
	return nLines;
}

int main(int argc, char* argv[]){
	if(argc==1){
		std::cout << "no filename"<<std::endl;
		return 0;
	}
	std::string finName(argv[1]);
	std::cout << finName <<std::endl;
	//int nWords = countWords(finName);
	//printf("nWords %d\n", nWords);
	int nLines = countLines(finName);
	printf("nLines %d\n", nLines);
	
	return 0;
}
