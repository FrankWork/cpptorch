#include "tree.h"
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <fstream>
#include <cmath>


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

float Node::FeatureSelection(const Matrix& x, const std::vector<int>& y) {
    float max_gain = -1;
    
    for(int i=0; i< NumFeatures(x); ++i) {
        std::vector<float> points = FeatureValues(x, i);
        for(int j=0; j< points.size();++j) {
            
            float gain = ComputeGain(x, y, i, points[j]);
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
		part2.size()/y.size()*Gini(part2);

}

float Node::Gini(std::vector<int>& labels) {
    std::unordered_map<int, float> freq;
    for(int v : labels) {
        freq[v] += 1;
    }
    float sum=0;

    for(auto &it : freq) {
        it.second /= labels.size();
        sum += (it.second*it.second);
    }
    
    return 1-sum;
}

void Node::Split(const Matrix& x, const std::vector<int>& y,
		Matrix& x1, std::vector<int>& y1,
		Matrix& x2, std::vector<int>& y2
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
    //root = std::make_shared<Node>();
}

void DecisionTree::Fit(const Matrix& X, const std::vector<int>& y) {
    //unordered_map<int> labels;
	if (y.empty()) {
		std::cerr << "label vector is empty\n";
		return;
	}
	std::cerr  << "ytrain: " << y.size() << "\n";
	auto it = max_element(y.cbegin(), y.cend());
	//std::cerr << *it << "\n";
    n_labels = *it + 1;

	std::cerr << "n_labels: " << n_labels << "\n";
	if (root == nullptr){
		root = std::make_shared<Node>();
	}
    Grow(root, X, y, 0);
}

void DecisionTree::Grow(const std::shared_ptr<Node>& node,
		const Matrix& X, const std::vector<int>& y, int depth) {
	if (node == nullptr) return;
	node->ComputeLabelProb(y, n_labels);
	//std::cout << "tree depth: " << depth << "\n";
	
	if(max_depth>0 && depth > max_depth){
		return;
	}
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
    Grow(node->left, x1, y1, depth+1);
    Grow(node->right, x2, y2, depth+1);
}

Matrix DecisionTree::predict_proba(const Matrix& X) {
	if (root == nullptr) {
		std::cerr << "tree root is nullptr" << "\n";
	}
	std::vector<float> preds;
    Matrix all_preds;
	for(int i=0;i < X.NumRows(); ++i) {
		preds = predict(root, X[i]);
        all_preds.push_back(preds);
	}
    return all_preds;
}

std::vector<int> DecisionTree::predict(const Matrix&X) {
    Matrix y_prob = predict_proba(X);
    std::vector<int> y_pred;
    for(int i=0;i < y_prob.NumRows(); ++i) {
		//std::cerr << "i: " << i << " size: " << y_prob[i].size() << "\n";
        //auto max_p = std::max_element(y_prob[i].cbegin(), y_prob[i].cend());
        auto max_p = std::max_element(y_prob[i].begin(), y_prob[i].end());
		//std::cerr << "max_p: " << *max_p << "\n";
        //int label = std::distance(y_prob[i].cbegin(), max_p);
        int label = std::distance(y_prob[i].begin(), max_p);
		//std::cerr << "label: " << label << "\n";
        y_pred.push_back(label);
    };
    return y_pred;
}

std::vector<float> DecisionTree::predict(std::shared_ptr<Node> node, 
		const std::vector<float>& x) {
	std::vector<float> pred;
	if (node->left == nullptr and node->right == nullptr) {
		return node->label_prob;
	}

	if(x[node->best_idx] > node->threshold) {
		if (node->left != nullptr) {
			return predict(node->left, x);
		}
		
	} else {
		if (node->right != nullptr){
			return predict(node->right, x);
		}
		
	}
	return node->label_prob;
}
float DecisionTree::accuracy(const std::vector<int>&y_pred, const std::vector<int>& y_true) {
    float tp = 0;
    for(int i=0; i< y_pred.size();++i) {
        if(y_pred[i] == y_true[i]) {
            ++tp;
        }
    }
    return tp / y_pred.size();
}

void load_data(const std::string& filename, Matrix& features, std::vector<int>& labels) {
	std::ifstream ifs(filename);
	std::string line;
	std::string word;
	std::vector<std::string> parts;
	while(getline(ifs, line)) {
        //std::cout << line << "\n";
		int size = line.length();
		for(int i=0;i < size;++i) {
			if(line[i] != '\t') {
				word.push_back(line[i]);
			} else {
				parts.push_back(word);
				word.clear();
			}
		}
		if (!word.empty()) {
			parts.push_back(word);
			word.clear();
		}
        //std::cout << parts.size() <<"\n";
		labels.push_back(std::stoi(parts[0]));
		std::vector<float> feat(parts.size()-1);
		for(int i=1;i < parts.size(); ++i) {
			feat[i-1] = std::stof(parts[i]);
		}
		features.push_back(feat);
		parts.clear();
	};
}


int main(int argc, char** argv) {
    //test_matmul();
	Matrix Xtrain, Xtest;
	std::vector<int> ytrain, ytest;
	
	load_data("train.txt", Xtrain, ytrain);
	load_data("test.txt", Xtest, ytest);

	std::cout << "train: " << Xtrain.NumRows() << ", " << Xtrain.NumCols() << "\n";
	std::cout << "test: " << Xtest.NumRows() << ", " << Xtest.NumCols() << "\n";
	std::cout << "train: " << ytrain.size()  << "\n";
	std::cout << "test: " << ytest.size() <<  "\n";
	float lr = 0.1;
	if (argc >= 2) {
		lr = std::stof(std::string(argv[1]));
		std::cout << "lr :" << lr << "\n";
	}
	DecisionTree clf;
	///clf.fit(Xtrain, ytrain, 2000, Xtest, ytest);
	clf.Fit(Xtrain, ytrain);
	std::vector<int> y_pred = clf.predict(Xtest);
	float acc = clf.accuracy(y_pred, ytest);
	std::cout << "acc: " << acc << "\n";
}

