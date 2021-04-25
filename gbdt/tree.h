#include <iostream>
#include <string>
#include <vector>
#include <memory>

#pragma once

class Matrix {
public:
    Matrix():_n_rows(0), _n_cols(0) {};
    Matrix(int n_row, int n_col):_n_rows(n_row), _n_cols(n_col), data(n_row, std::vector<float>(n_col)) {}
    int NumRows() const {return _n_rows;}
    int NumCols() const {return _n_cols;}
    std::vector<float> operator[](int i) const {return data[i];};
    std::vector<float>& operator[](int i)  {return data[i];};
    void push_back(const std::vector<float> x) {
        if (_n_cols == 0) _n_cols = x.size();
        data.push_back(x);
        ++_n_rows;
    };
    float mean_by_column(int column_idx);
    std::vector<float> slice_by_column(int column_idx);
    void set_column(const std::vector<float>& vec, int column_idx);

private:
    size_t _n_rows;
    size_t _n_cols;
    std::vector<std::vector<float>> data;
};

class Node {
public:
    Node() : left(nullptr), right(nullptr), best_idx(-1), threshold(-1){}
	//Node(const Matrix& x, const std::vector<int>& y):x(x), y(y){}
    int NumLabels();
    int NumFeatures(const Matrix&x);
    int NumSamples(const Matrix&x);
    std::vector<float> FeatureValues(const Matrix&x, int feat_idx);
    float FeatureSelection(const Matrix&x, const std::vector<float>&y, const std::string& criterion);
    void ComputeLabelProb(const std::vector<float>&y, int num_label, const std::string& criterion);
    float ComputeGain(const Matrix& x, const std::vector<float>&y, 
        int i, float threshold, const std::string& criterion);
    float Gini(const std::vector<float>& labels);
    float Entropy(const std::vector<float>& labels);
    float MeanSquaredError(const std::vector<float>& labels);
    void Split(const Matrix& x, const std::vector<float>& y,
		Matrix& x1, std::vector<float>& y1,
		Matrix& x2, std::vector<float>& y2);

	std::vector<float> label_prob;
    float pred_label;
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
    DecisionTree(const std::string& criterion):epsilon(1e-7), criterion(criterion), max_depth(0){}
    DecisionTree(const std::string& criterion, int max_depth):epsilon(1e-7), criterion(criterion), max_depth(max_depth){}
    DecisionTree(int max_depth):epsilon(1e-7), max_depth(max_depth),criterion("entropy") {}

    void Fit(const Matrix& X, const std::vector<float>& y);
    void Grow(const std::shared_ptr<Node>& root, const Matrix& X, const
    std::vector<float>& y, int depth);
    Matrix predict_proba(const Matrix& X);
    std::vector<float> predict(const Matrix& X);
    
    float accuracy(const std::vector<float>& y_pred, 
        const std::vector<float>& y_true);
    float MeanSquaredError(const std::vector<float>& y_pred, const std::vector<float>& y_true);
    ~DecisionTree(){}

private:
    std::shared_ptr<Node> _predict(std::shared_ptr<Node> node, const
        std::vector<float>& X);

	std::shared_ptr<Node> root;
    std::string criterion;
	int max_depth;
    int n_labels;
	float epsilon;
};

void load_data(const std::string& filename, Matrix& features, std::vector<float>& labels);
