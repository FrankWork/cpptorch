#include "gbdt.h"
#include <algorithm>
#include <numeric>

Matrix onehot(const std::vector<float>& yvec, int n_out){
    Matrix Y(yvec.size(), n_out);
    if (n_out == 1){
        for(int i=0;i < yvec.size(); ++i) {
            Y[i][0] = yvec[i];
        }
        return Y;
    }
    for(int i=0;i < yvec.size(); ++i) {
        for(int j=0;j < n_out; ++j){
            Y[i][j] = 0;    
        }
        Y[i][yvec[i]] = 1;
    }
    return Y;
}

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

void BaseRegressor::Fit(const std::vector<float>& y){
		mean = std::accumulate(y.begin(), y.end(), 0);
		mean /= y.size();
}

float MSELoss::base_estimator(const std::vector<float>& y){
    float ans = std::accumulate(y.begin(), y.end(), 0);
    return ans / y.size();
}

float CrossEntropyLoss::base_estimator(const std::vector<float>& y){
    float ans = std::accumulate(y.begin(), y.end(), 0);
    return ans / y.size();
}

std::vector<float> MSELoss::grad(const std::vector<float>&y_true, const std::vector<float>& y_pred) {
    // grad = \frac{1}{n}\sum_i(y_i-\hat{y_i}) 
    int n = y_true.size();
    std::vector<float> res(n, 0);
    for(int i=0;i < n; ++i){
        res[i] = (y_true[i] - y_pred[i]) / n;
    }
    return res;
}

std::vector<float> CrossEntropyLoss::grad(const std::vector<float>&y_true, const std::vector<float>& y_pred) {
    int n = y_true.size();

}

GradientBoostingDT::GradientBoostingDT(int n_trees, int max_depth, bool regression): 
		n_trees(n_trees), max_depth(max_depth), regression(regression){
    if(regression) {
        loss = std::make_unique<MSELoss>();
    } else {
        loss = std::make_unique<CrossEntropyLoss>();
    }
}

void GradientBoostingDT::Fit(const Matrix& X, const std::vector<float>& yvec) {
    int n_out = 1;
    if(!regression) {
        n_out = *max_element(yvec.begin(), yvec.end()) + 1;
    }
    Matrix Y = onehot(yvec, n_out);

    Matrix preds(n_trees, n_out);

    // initial learner
    for(int i=0;i < n_out; ++i) {
        preds[0][i] = Y.mean_by_column(i);
    }

    for(int i=1; i < n_trees; ++i){
        for(int j=0; j < n_out; ++j){
            std::vector<float> y_j = Y.slice_by_column(j);
            std::vector<float> grad = loss->grad();

            DecisionTree tree("mse",  max_depth);
            tree.Fit(X, grad);
            std::vector<float> pred_j = tree.predict(X);
            preds.set_column(pred_j, j);
        }
    }
}