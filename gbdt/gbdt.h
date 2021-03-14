#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include "tree.h"

#pragma once


class BaseClassifier{
public:
	BaseClassifier(int nclass): prob(nclass, 0){}
	// count label frequency
	void Fit(const std::vector<float>& y);
	std::vector<float> predict() {return prob;}
private:
	std::vector<float> prob;
};

class BaseRegressor{
public:
	BaseRegressor(){};
	void Fit(const std::vector<float>& y);
	float predict() {return mean;}
private:
	float mean;
};

class Loss {
public:
	virtual float base_estimator(const std::vector<float>& y);
	virtual std::vector<float> grad(const std::vector<float>&y_true, const std::vector<float>& y_pred);
};

class MSELoss : public Loss {
public:
	float base_estimator(const std::vector<float>& y);
	std::vector<float> grad(const std::vector<float>&y_true, const std::vector<float>& y_pred);
};

class CrossEntropyLoss : public Loss {
public:
	float base_estimator(const std::vector<float>& y);
	std::vector<float> grad(const std::vector<float>&y_true, const std::vector<float>& y_pred);
};

class GradientBoostingDT {
public:
	GradientBoostingDT(int n_trees, int max_depth, bool regression);
	void Fit(const Matrix& X, const std::vector<float>& y);
	Matrix predict_proba(const Matrix& X);
	std::vector<float> predict(const Matrix& X);

private:
	int n_trees;
	int max_depth;
	bool regression;
	std::unique_ptr<Loss> loss;
};

