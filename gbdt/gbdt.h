#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include "tree.h"

#pragma once

class BaseEstimator {
public:
	BaseEstimator(){}
    virtual void Fit(const std::vector<float>&y)=0;
    std::vector<float> predict() {return prob;}
    virtual ~BaseEstimator(){}
protected:
    std::vector<float> prob;
};

class BaseClassifier : public BaseEstimator{
public:
	BaseClassifier(int nclass): _nclass(nclass){}
	// count label frequency
	void Fit(const std::vector<float>& y);
	//std::vector<float> predict() {return prob;}
    virtual ~BaseClassifier(){}
private:
	//std::vector<float> prob;
	int _nclass;
};

class BaseRegressor : public BaseEstimator{
public:
	BaseRegressor(){};
	void Fit(const std::vector<float>& y);
	//std::vector<float> predict() {return prob;}

};


class Loss {
public:
	//virtual float base_estimator(const std::vector<float>& y);
	Loss(){};
	virtual std::vector<float> grad(const std::vector<float>&y_true, const std::vector<float>& y_pred)=0;
	virtual ~Loss(){};
};

class MSELoss : public Loss {
public:
	//float base_estimator(const std::vector<float>& y);
	MSELoss(){};
	std::vector<float> grad(const std::vector<float>&y_true, const std::vector<float>& y_pred);
	~MSELoss(){};
};

class CrossEntropyLoss : public Loss {
public:
	//float base_estimator(const std::vector<float>& y);
	CrossEntropyLoss(){};
	std::vector<float> grad(const std::vector<float>&y_true, const std::vector<float>& y_pred);
	~CrossEntropyLoss(){};
};

class GradientBoostingDT {
public:
	GradientBoostingDT(int n_trees, int max_depth, int nclass, bool regression);
	void Fit(const Matrix& X, const std::vector<float>& y);
	Matrix predict_proba(const Matrix& X);
	std::vector<float> predict(const Matrix& X);

private:
	int n_trees;
	int max_depth;
	int nclass;
	bool regression;
	//std::unique_ptr<Loss> loss;
	Matrix weights;
	std::vector<std::vector<BaseEstimator*>> learners;

};

