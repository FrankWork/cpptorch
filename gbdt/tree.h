#include <iostream>
#include <string>
#include <vector>
#include <memory>

#pragma once

class Matrix {
public:
    Matrix():_n_rows(0), _n_cols(0) {};
    int NumRows() const {return _n_rows;}
    int NumCols() const {return _n_cols;}
    std::vector<float> operator[](int i) const {return data[i];};
    std::vector<float>& operator[](int i)  {return data[i];};
    void push_back(const std::vector<float> x) {
        if (_n_cols == 0) _n_cols = x.size();
        data.push_back(x);
        ++_n_rows;
    };
private:
    size_t _n_rows;
    size_t _n_cols;
    std::vector<std::vector<float>> data;
};

class Node {
public:
    Node() {}
	//Node(const Matrix& x, const std::vector<int>& y):x(x), y(y){}
    int NumLabels();
    int NumFeatures(const Matrix&x);
    int NumSamples(const Matrix&x);
    std::vector<float> FeatureValues(const Matrix&x, int feat_idx);
    float FeatureSelection(const Matrix&x, const std::vector<int>&y);
    void ComputeLabelProb(const std::vector<int>&y, int num_label);
    float ComputeGain(const Matrix& x, const std::vector<int>&y, 
        int i, float threshold);
    float Gini(std::vector<int>& labels);
    void Split(const Matrix& x, const std::vector<int>& y,
		Matrix& x1, std::vector<int>& y1,
		Matrix& x2, std::vector<int>& y2);

	std::vector<float> label_prob;
    std::shared_ptr<Node> left;
    std::shared_ptr<Node> right;
	int best_idx;
	float threshold;

    ~Node(){}
private:
    //Matrix x;
    //std::vector<int> y;
};

class DecisionTree {

public:
    DecisionTree(){DecisionTree(0);}
    DecisionTree(int max_depth);
    void Fit(const Matrix& X, const std::vector<int>& y);
    void Grow(const std::shared_ptr<Node>& root, const Matrix& X, const
    std::vector<int>& y, int depth);
    Matrix predict_proba(const Matrix& X);
    std::vector<int> predict(const Matrix& X);
    std::vector<float> predict(std::shared_ptr<Node> node, const
        std::vector<float>& X);
    float accuracy(const std::vector<int>& y_pred, 
        const std::vector<int>& y_true);
    ~DecisionTree(){}

private:
	std::shared_ptr<Node> root;
	int max_depth;
    int n_labels;
	float epsilon;
};

