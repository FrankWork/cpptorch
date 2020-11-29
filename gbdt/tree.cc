#include "tree.h"
#include <unordered_map>
#include <unordered_set>
#include <algorithm>

void Node::SetData(const Matrix& X, const std::vector<int>& y) {
    x=X;
    y=y;
}

void Node::ComputeLabelProb(const std::vector<int>& y, int num_label) {
	label_prob.resize(num_label);
	for(int i=0;i < num_label; ++i) {
		label_prob[i] = 0;
	}
    for(int label: y) {
        label_prob[label] += 1;
    }
    for(int i=0;i < num_label; ++i) {
        label_prob[i] /= y.size();
    }
}

int Node::NumLabels() {
    return label_prob.size();
}

int Node::NumFeatures(const Matrix& x) {
    return x.NumCols();
}

int Node::NumSamples(const Matrix& x) {
    return x.NumRows();
}

std::vector<float> Node::FeatureValues(const Matrix& x, int feat_idx) {
    std::unordered_set<float> value_set;
    for(int i=0;i<NumSamples(x); ++i) {
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

float Node::FeatureSelection(const Matrix& x, const vector<int>& y) {
    float max_gain = -1;
    
    for(int i=0; i< NumFeatures(x); ++i) {
        std::vector<float> points = FeatureValues(x, i);
        for(int j=0; j< points.size();++j) {
            
            float gain = ComputeGain(i, points[j]);
            if (max_gain < gain) {
                max_gain = gain;
                best_idx = i;
				threshold = points[j];
            }
        }
    }
	return max_gain;
}

float Node::ComputeGain(const Matrix& x, const std::vector<int>& y, 
		int feat_idx, float threshold) {
    float entropy_d = 0;
	std::vector<int> part1, part2;       
    for(int i=0;i<x.NumRows();++i) {
        if(x[i][feat_idx] > threshold) {
        	part1.push_back(y[i]);
        } else {
        	part2.push_back(y[i]);
        }
    }
	return part1.size()/y.size()*Gini(part1) + \
		part2.size()/y.size()*Gini(part2)

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

void Node::Split(const Matrix& x, const std::vector<int>& y,
		Matrix& x1, std::vector<int>& y1,
		Matrix& x2, std::vector<int>& y2,
		) {
	for(int i=0;i<NumSamples(x); ++i){
		if(x[i][best_idx] > threshold) {
			x1.push_back(x[i]);
			y1.push_back(y[i]);
		} else {
			x2.push_back(x[i]);
			y2.push_back(y[i]);
		}
	}
	left = std::make_shared<Node>();
	right = std::make_shared<Node>();
}

DecisionTree::DecisionTree(int max_depth):
	epsilon(1e-5), max_depth(max_depth){
    root = std::make_shared<Node>();
}

void DecisionTree::Fit(const Matrix& X, const std::vector<int>& y) {
    //unordered_map<int> labels;
    //int n_labels = max_element(y.begin(), y.end()) + 1;

    Grow(root, X, y, max_depth);    
}

void DecisionTree::Grow(const std::shared_ptr<Node>& node,
		const Matrix& X, const std::vector<int>& y, int depth) {
	if (depth<0) return;
	node->ComputeLabelProb(y);

    if(node->NumLabels() <= 1) {
       return;
    }
    if(node->NumFeatures(X) == 1) {
        return;
    }
    float max_gain = node->FeatureSelection(X, y);
    if(max_gain < epsilon) {
        return;
    }

	Matrix x1, x2;
	std::vector<int> y1,y2;
    node->Split(X, y, x1, y1, x2, y2);
    Grow(node->left, x1, y1, depth-1);
    Grow(node->right, X2, y2, depth-1);
}

std::vector<float> DecisionTree::predict(const Matrix& X) {
	std::vector<float> preds;
	for(int i=0;i < X.NumRows(); ++i) {
		
	}
}

std::vector<float> DecisionTree::predict(std::shared_ptr<Node> node, 
		const std::vector<float> x) {
	std::vector<float> pred;
	if (node->left == nullptr and node->right == nullptr) {
		return node->label_prob;
	}


	if(x[node->best_idx] > node->threshold) {
		return predict(node->left, x);
	} else {
		return predict(node->right, x);
	}
}


