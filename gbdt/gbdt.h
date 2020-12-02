#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include "tree.h"

#pragma once

class Loss {
public:
	virtual float base_estimator(const std::vector<float>& y);
};

class MSELoss : public Loss {
public:
	float base_estimator(const std::vector<float>& y);
};

class CrossEntropyLoss : public Loss {
public:
	float base_estimator(const std::vector<float>& y);
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

