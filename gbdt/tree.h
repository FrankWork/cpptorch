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
    void SetData(const Matrix& X, const std::vector<int>& y);
    int NumLabels();
    int NumFeatures();
    void ComputeGain(int i, float& gain, float& threshold);
        
    ~Node(){}
private:
    Matrix x;
    std::vector<int> y;
    std::unordered_map<int, float> label_prob;
    std::shared_ptr<Node> left;
    std::shared_ptr<Node> right;
};

class DecisionTree {

public:
    void Fit(const Matrix& X, const std::vector<float>& y);
    void Grow(const unique_ptr<Node>& root);
    std::vector<float> predict(const Matrix& X);
    ~DecisionTree(){}

private:
    unique_ptr<Node> root;
};

