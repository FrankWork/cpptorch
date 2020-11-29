#include <iostream>
#include <string>
#include <vector>

#pragma once

class Matrix {
public:
    int NumRows() {return _n_rows;}
    int NumCols() {return _n_cols;}
private:
    size_t _n_rows;
    size_t _n_cols;
};

class Node {
public:
    Node() {}
	Node(const Matrix& x, const std::vector<int>& y):x(x), y(y){}
    void SetData(const Matrix& X, const std::vector<int>& y);
    int NumLabels();
    int NumFeatures();
    void ComputeGain(int i, float& gain, float& threshold);
        
    ~Node(){}
private:
    Matrix x;
    std::vector<int> y;
	std::vector<float> label_prob;
    std::shared_ptr<Node> left;
    std::shared_ptr<Node> right;
	int best_idx;
	float threshold;
};

class DecisionTree {

public:
    void Fit(const Matrix& X, const std::vector<float>& y);
    void Grow(const std::shared_ptr<Node>& root, const Matrix& X, const std::vector<float>& y);
    std::vector<float> predict(const Matrix& X);
    ~DecisionTree(){}

private:
	std::shared_ptr<Node> root;
	int max_depth;
	float epsilon;
};

