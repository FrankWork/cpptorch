#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include "tree.h"
#include "gbdt.h"

int main(){
    
	
	Matrix Xtrain, Xtest;
	std::vector<float> ytrain, ytest;
	
	load_data("train.txt", Xtrain, ytrain);
	load_data("test.txt", Xtest, ytest);

	std::cout << "train: " << Xtrain.NumRows() << ", " << Xtrain.NumCols() << "\n";
	std::cout << "test: " << Xtest.NumRows() << ", " << Xtest.NumCols() << "\n";
	std::cout << "train: " << ytrain.size()  << "\n";
	std::cout << "test: " << ytest.size() <<  "\n";

	auto it = std::max_element(ytrain.cbegin(), ytrain.cend());
	int n_labels = *it + 1;
	std::cerr << "n_labels: " << n_labels << "\n";

	GradientBoostingDT tree(10, -1, n_labels, false);
	tree.Fit(Xtrain, ytrain);
	return 0;
}








