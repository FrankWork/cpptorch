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

float MSELoss::base_estimator(const std::vector<float>& y){
    float ans = std::accumulate(y.begin(), y.end(), 0);
    return ans / y.size();
}

float CrossEntropyLoss::base_estimator(const std::vector<float>& y){
    float ans = std::accumulate(y.begin(), y.end(), 0);
    return ans / y.size();
}

float MSELoss::grad(const std::vector<float>&y_true, const std::vector<float>& y_pred) {

}

float CrossEntropyLoss::grad(const std::vector<float>&y_true, const std::vector<float>& y_pred) {
    
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

            DecisionTree tree("mse", max_depth);
            tree.Fit(X, grad);
            std::vector<float> pred_j = tree.predict(X);
            preds.set_column(pred_j, j);
        }
    }
}