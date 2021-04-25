#include <iostream>
#include <fstream>

#include "string_util.h"

int main(int argc, char* argv[]) {
    //std::string filepath="1.txt";
    //std::ifstream infile(filepath.c_str());
    if(argc < 2) {
        std::cerr << "argc=" << argc << "\n";
        return -1;
    }
    std::ifstream infile(argv[1]);
    std::string line;

    while (std::getline(infile, line)){ 
        //for(char c: line) {
        for (const auto c : string_util::UTF8ToUnicodeText(line)){
            string_util::UnicodeText uw;
            uw.push_back(c);
            std::cout << string_util::UnicodeTextToUTF8(uw) << " ";
        }
    }
    std::cout << "\n";
    return 0;
}
