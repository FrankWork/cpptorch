#include <iostream>
#include <vector>
#include <numeric>
//#include "tree.h"

template <typename T>
class BaseEstimator {
public:
    void Fit(const std::vector<float>&y);
    T predict();
    virtual ~BaseEstimator(){}
private:
    T prob;
};

template <typename T>
class BaseClassifier : public BaseEstimator<T>{
public:
	BaseClassifier(int nclass): prob(nclass, 0){}
	// count label frequency
	void Fit(const std::vector<float>& y);
	T predict() {return prob;}
    virtual ~BaseClassifier(){}
private:
	T prob;
};

template <typename T>
void BaseClassifier<T>::Fit(const std::vector<float>& y) {
		// count label frequency
		for(int i = 0;i < y.size(); ++i){
			int label = y[i];
			prob[label] += 1;
		}
		for(int i = 0; i < prob.size(); ++i){
			prob[i] /= y.size();
		}
}

int main(){
    //Matrix a(3, 4);
    //std::cout << a.NumRows() << " " << a.NumCols() << "\n";
    //BaseClassifier<std::vector<float>>* c = new BaseClassifier<std::vector<float>>(4);
    BaseEstimator<std::vector<float>>* c = new BaseClassifier<std::vector<float>>(4);
    std::vector<float> y{1,2,3,2,0,1,2};
    c->Fit(y);
    std::vector<float> prob = c->predict();
    for(auto x: prob)
    std::cout << x <<" ";
    std::cout <<"\n";
}


/*
class BaseClassifier : BaseEstimator<std::vector<float>>{
public:
	BaseClassifier(int nclass): prob(nclass, 0){}
	// count label frequency
	void Fit(const std::vector<float>& y);
	std::vector<float> predict() {return prob;}
private:
	std::vector<float> prob;
};

class BaseRegressor : BaseEstimator<float>{
public:
	BaseRegressor(){};
	void Fit(const std::vector<float>& y);
	float predict() {return mean;}
private:
	float mean;
};

void BaseClassifier::Fit(const std::vector<float>& y) {
		// count label frequency
		for(int i = 0;i < y.size(); ++i){
			int label = y[i];
			prob[label] += 1;
		}
		for(int i = 0; i < prob.size(); ++i){
			prob[i] /= y.size();
		}
}
*/




/*void BaseRegressor::Fit(const std::vector<float>& y){
		mean = std::accumulate(y.begin(), y.end(), 0);
		mean /= y.size();
}*/




