#include <iostream>
#include <string>
#include <vector>

#pragma once

class Matrix {
public:
private:
    size_t _n_rows;
    size_t _n_cols;
};

class DecisionTree {

public:
    DecisionTree(){}
    void fit(const Matrix& X, const std::vector<float>& y);
    std::vector<float> predict(const Matrix& X);
    ~DecisionTree(){}

private:
};

