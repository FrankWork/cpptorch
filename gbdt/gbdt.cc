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
		prob.resize(_nclass);
		for(int i = 0;i < y.size(); ++i){
			int label = y[i];
			prob[label] += 1;
			std::cout << label <<" ";
		}
		for(int i = 0; i < prob.size(); ++i){
			prob[i] /= y.size();
		}
}

void BaseRegressor::Fit(const std::vector<float>& y){
	prob.resize(1);
	prob[0] = std::accumulate(y.begin(), y.end(), 0);
	prob[0] /= y.size();
}

/*float MSELoss::base_estimator(const std::vector<float>& y){
    float ans = std::accumulate(y.begin(), y.end(), 0);
    return ans / y.size();
}

float CrossEntropyLoss::base_estimator(const std::vector<float>& y){
    float ans = std::accumulate(y.begin(), y.end(), 0);
    return ans / y.size();
}*/

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
    std::vector<float> res;
    float eps = 2.220446049250313e-16;
    
    return res;
}

GradientBoostingDT::GradientBoostingDT(int n_trees, int max_depth, int nclass, bool regression): 
		n_trees(n_trees), max_depth(max_depth), nclass(nclass), regression(regression){
    
}

void GradientBoostingDT::Fit(const Matrix& X, const std::vector<float>& yvec) {
    int n_out = 1;
    if(!regression) {
        n_out = *max_element(yvec.begin(), yvec.end()) + 1;
    }
    // assert n_out == nclass
    Matrix Y = onehot(yvec, n_out);

    // base estimator
    std::unique_ptr<BaseEstimator> base;
    if(regression){
        //base = std::make_unique<BaseRegressor>();// c++14
        base = std::unique_ptr<BaseRegressor>(new BaseRegressor());
        nclass = 1;
    }else{
        //base = std::make_unique<BaseClassifier>(nclass);    
        base = std::unique_ptr<BaseClassifier>(new BaseClassifier(nclass));
    }
    base->Fit(yvec);
    learners[0].push_back(base);
    
    // initial learner
    int n_sample = X.NumRows();
    Matrix preds(n_sample, nclass);

    for(int i=0;i < n_sample; ++i) {
        preds[i] = base->predict();
        for(int j=0;j < nclass;++j){
            std::cout << preds[i][j] << " ";
        }
        std::cout <<"\n";
    }

    // tree estimator
    std::unique_ptr<Loss> loss;
    if(regression) {
        loss = std::unique_ptr<MSELoss>(new MSELoss()); // c++14
    } else {
        loss = std::unique_ptr<CrossEntropyLoss>(new CrossEntropyLoss());
    }

    for(int i=1; i < n_trees; ++i){
        for(int j=0; j < n_out; ++j){
            std::vector<float> y_j = Y.slice_by_column(j);
            std::vector<float> grad = loss->grad();

            DecisionTree tree("mse",  max_depth);
            tree.Fit(X, -grad);
            std::vector<float> pred_j = tree.predict(X);
            preds.set_column(pred_j, j);
    }

    /*
    

    }*/
}