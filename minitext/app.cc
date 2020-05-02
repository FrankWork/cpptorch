#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include "dictionary.h"
#include "vector.h"
#include "densematrix.h"

int main(int argc, char* argv[]){
	//std::string input = argv[1];

	//auto args_ = std::make_shared<fasttext::Args>();
  	//auto dict_ = std::make_shared<fasttext::Dictionary>(args_);

	//std::ifstream ifs(input);

	//dict_->readFromFile(ifs);
	//ifs.close();
	fasttext::Vector a(3);
	for (size_t i=0; i < a.size(); ++i){
		a[i] = 1. * (i+2);
	}
	std::cout << a << "\n";

	fasttext::DenseMatrix m(2, 3);
	fasttext::real* m_ptr = m.data();
	for (size_t i=0; i < 6; ++i){
		m_ptr[i] = 1. * i;
		std::cout << m_ptr[i] << " ";
	}
	std::cout << "\n";

	m.multiplyRow(a, 0, -1);
	for (size_t i=0; i < 6; ++i){
		std::cout << m_ptr[i] << " ";
	}
	std::cout << "\n";

	return 0;
}
