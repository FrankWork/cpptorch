#include "tree.h"
#include <unordered_map>
#include <unordered_set>
#include <algorithm>

void Node::SetData(const Matrix& X, const std::vector<int>& y) {
    x=X;
    y=y;
    for(int label: y) {
        label_prob[label] += 1;
    }
    for(int label: label_prob) {
        label_prob[label] /= y.size();
    }
}

int Node::NumLabels() {
    return label_prob.size();
}

int Node::NumFeatures() {
    return x.NumCols();
}

int Node::NumSamples() {
    return x.NumRows();
}

std::vector<float> Node::FeatureValues(int feat_idx) {
    std::unordered_set<float> value_set;
    for(int i=0;i<node.NumSamples();++i) {
        value_set.insert(x[i][feat_idx]);
    }
    std::vector<float> values(value_set.begin(), value_set.end());
    std::sort(values.begin(), values.end());

    std::vector<float> candidates;
    for(int i=0;i < values.size()-1; ++i){
        float t = (values[i] + values[i+1]) / 2;
        candidates.push_back(t);
    }
    return candidates;
}

void Node::FeatureSelection() {
    float max_gain = 0;
    int best_idx = 0;
    float threshold = 0;
    
    for(int i=0; i< NumFeatures(); ++i) {
        std::vector<float> points = FeatureValues(i);
        for(int j=0; j< points.size();++j) {
            
            float gain = ComputeGain(i, points[j]);
            if (max_gain < gain) {
                max_gain = gain;
                best_idx = i;
            }
        }
    }
}

float Node::ComputeGain(int feat_idx, float threshold) {
    float entropy_d = 0;
       
    for(int i=0;i<x.NumRows();++i) {
        if(x[i][feat_idx] > threshold) {
        
        } else {
        
        }
    }

}

float Node::Gini(std::vector<int>& labels) {
    std::unordered_map<int, float> freq;
    for(int v : labels) {
        freq[v] += 1;
    }
    float sum=0;
    for(auto it& : freq) {
        it->second /= labels.size();
        sum += (it->second*it->second);
    }
    
    return 1-sum;
}

DecisionTree::DecisionTree(){
    root = std::make_unique<Node>();
}

void DecisionTree::Fit(const Matrix& X, const std::vector<int>& y) {
    //unordered_map<int> labels;
    //int n_labels = max_element(y.begin(), y.end()) + 1;

    root->SetData(X, y);
    Grow(root);    
}

void DecisionTree::Grow(const unique_ptr<Node>& node) {
    if(node->NumLabels() == 1) {
       return;
    }
    if(node->NumFeatures() == 1) {
        return
    }
    if(max_gain < epsilon) {
        return 
    }
    node->FeatureSelection();
    node->Split(best_idx, threshold);
    Grow(node->left);
    Grow(node->right);

}

std::vector<float> DecisionTree::predict(const Matrix& X) {
}
