#include "tree.h"
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <numeric>

int argmax(const std::vector<float>& vec) {
	auto max_p = std::max_element(vec.begin(), vec.end());
    int label = std::distance(vec.begin(), max_p);
	return label;
}

float Matrix::mean_by_column(int column_idx) {
	float ans = 0;
	for(int i=0;i < _n_rows; ++i) {
		ans += data[i][column_idx];
	}
	return ans / _n_rows;
}

std::vector<float> Matrix::slice_by_column(int column_idx) {
	std::vector<float> vec;
	for(int i=0; i< _n_rows; ++i){
		vec.push_back(data[i][column_idx]);
	}
	return vec;
}

void Matrix::set_column(const std::vector<float>& vec, int column_idx) {
	for(int i=0; i< _n_rows; ++i){
		data[i][column_idx] = vec[i];
	}
}

void Node::ComputeLabelProb(const std::vector<float>& y, int num_label,
	const std::string& criterion) {
	// bincount
	if (criterion == "mse") {
		pred_label = std::accumulate(y.begin(), y.end(), 0);
		pred_label /= y.size();
	} else {
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
		pred_label = argmax(label_prob);
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

	// candidate split point
    std::vector<float> candidates;
    for(int i=0;i < values.size()-1; ++i){		
        float t = (values[i] + values[i+1]) / 2;
        candidates.push_back(t);
    }
    return candidates;
}

// TODO: move it to DecisionTree
float Node::FeatureSelection(const Matrix& x, const std::vector<float>& y, const std::string& criterion) {
    float max_gain = -1;
    
    for(int i=0; i< NumFeatures(x); ++i) {
        std::vector<float> points = FeatureValues(x, i);
        for(int j=0; j< points.size();++j) {
            
            float gain = ComputeGain(x, y, i, points[j], criterion);
            if (max_gain < gain) {
                max_gain = gain;
                best_idx = i;
				threshold = points[j];
            }
        }
    }
	//std::cerr << "max_gain: " << max_gain 
	//		<< " best_idx: " << best_idx
	//		<< " threshold: " << threshold
	//		<< "\n";

	return max_gain;
}

float Node::ComputeGain(const Matrix& x, const std::vector<float>& y, 
		int feat_idx, float threshold, const std::string& criterion) {
	// CART: minimize Gini(D, A), is equal to maxmize Gini(D) - Gini(D, A)
	// ID3: maxmize IG(D, A) = H(D) - H(D|A)
	// MSE: minimize MSE(D, A) is equal to maxmize MSE(D) - MSE(D, A)
	std::vector<float> part1, part2;
    for(int i=0;i<x.NumRows();++i) {
        if(x[i][feat_idx] <= threshold) {
        	part1.push_back(y[i]);
        } else {
        	part2.push_back(y[i]);
        }
    }
	float p1 = part1.size() / float(y.size());
	float p2 = part2.size() / float(y.size());
	if (criterion == "gini") {
		float parent_loss = Gini(y);
		float child_loss = p1*Gini(part1) + p2*Gini(part2);
		return parent_loss - child_loss;
	} else if (criterion == "entropy") {
		float parent_loss = Entropy(y);
		float child_loss = p1*Entropy(part1) + p2*Entropy(part2);
		return parent_loss - child_loss;
	}
	float parent_loss = MeanSquaredError(y);
	float child_loss = p1*MeanSquaredError(part1) + p2*MeanSquaredError(part2);
	return parent_loss - child_loss;
}

float Node::Entropy(const std::vector<float>& labels) {
    std::unordered_map<int, float> freq;
    for(int v : labels) {
        freq[v] += 1;
    }
    float sum=0;

    for(auto &it : freq) {
        it.second /= labels.size();
        sum += (it.second*std::log2(it.second+1e-5));
    }
    
    return 1-sum;
}

float Node::Gini(const std::vector<float>& labels) {
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

float Node::MeanSquaredError(const std::vector<float>& labels) {
	if (labels.empty()) return 0;
	float mean_y = std::accumulate(labels.begin(), labels.end(), 0) / labels.size();
	float error = 0;
	for(auto y: labels) {
		float t = y - mean_y;
		error += t*t;
	}
	return error/labels.size();
}

void Node::Split(const Matrix& x, const std::vector<float>& y,
		Matrix& x1, std::vector<float>& y1,
		Matrix& x2, std::vector<float>& y2
		) {
	for(int i=0;i<NumSamples(x); ++i){
		if(x[i][best_idx] <= threshold) {
			x1.push_back(x[i]);
			y1.push_back(y[i]);
		} else {
			x2.push_back(x[i]);
			y2.push_back(y[i]);
		}
	}
	// TODO: move it to DecisionTree::Grow
	left = std::make_shared<Node>();
	right = std::make_shared<Node>();
}

void DecisionTree::Fit(const Matrix& X, const std::vector<float>& y) {
    //unordered_map<int> labels;
	if (y.empty()) {
		std::cerr << "label vector is empty\n";
		return;
	}
	std::cerr  << "ytrain: " << y.size() << "\n";

	if (criterion != "mse") {
		auto it = std::max_element(y.cbegin(), y.cend());
		n_labels = *it + 1;
		std::cerr << "n_labels: " << n_labels << "\n";
	}
	
	if (root == nullptr){
		root = std::make_shared<Node>();
	}
    Grow(root, X, y, 0);
}

void DecisionTree::Grow(const std::shared_ptr<Node>& node,
		const Matrix& X, const std::vector<float>& y, int depth) {
	if (node == nullptr) return;
	
	node->ComputeLabelProb(y, n_labels, criterion);

	//std::cout << "tree depth: " << depth << "\n";
	
	if(max_depth>0 && depth > max_depth){
		return;
	}
    if(criterion!="mse" && node->NumLabels() <= 1) {
       return;
    }
    if(node->NumFeatures(X) == 1) {
        return;
    }
	
    float max_gain = node->FeatureSelection(X, y, criterion);
	
    if(max_gain < epsilon) {
        return;
    }

	Matrix x1, x2;
	std::vector<float> y1,y2;
    node->Split(X, y, x1, y1, x2, y2);
    Grow(node->left, x1, y1, depth+1);
    Grow(node->right, x2, y2, depth+1);
}

Matrix DecisionTree::predict_proba(const Matrix& X) {
	if (root == nullptr) {
		std::cerr << "tree root is nullptr" << "\n";
	}
    Matrix all_preds;
	for(int i=0;i < X.NumRows(); ++i) {
		std::shared_ptr<Node> leaf = _predict(root, X[i]);
        all_preds.push_back(leaf->label_prob);
	}
    return all_preds;
}

std::vector<float> DecisionTree::predict(const Matrix&X) {
    std::vector<float> y_pred;
    for(int i=0;i < X.NumRows(); ++i) {
		std::shared_ptr<Node> leaf = _predict(root, X[i]);
        y_pred.push_back(leaf->pred_label);
    };
    return y_pred;
}

std::shared_ptr<Node> DecisionTree::_predict(std::shared_ptr<Node> node, 
		const std::vector<float>& x) {
	if (node->left == nullptr and node->right == nullptr) {
		return node;
	}

	if(node->best_idx != -1) {
		if(x[node->best_idx] <= node->threshold) {
			if (node->left != nullptr) {
				return _predict(node->left, x);
			}
		} else {
			if (node->right != nullptr){
				return _predict(node->right, x);
			}
		}
	}
	return node;
}

float DecisionTree::accuracy(const std::vector<float>&y_pred, const std::vector<float>& y_true) {
    float tp = 0;
    for(int i=0; i< y_pred.size();++i) {
        if(y_pred[i] == y_true[i]) {
            ++tp;
        }
    }
    return tp / y_pred.size();
}
float DecisionTree::MeanSquaredError(const std::vector<float>& y_pred, const std::vector<float>& y_true) {
	float error = 0;
	for(int i=0;i < y_pred.size(); ++i) {
		float t = y_pred[i] - y_true[i];
		error += t*t;
	}
	return error / y_pred.size();
}

void load_data(const std::string& filename, Matrix& features, std::vector<float>& labels) {
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
		labels.push_back(std::stof(parts[0]));
		std::vector<float> feat(parts.size()-1);
		for(int i=1;i < parts.size(); ++i) {
			feat[i-1] = std::stof(parts[i]);
		}
		features.push_back(feat);
		parts.clear();
	};
}

void test_classifier() {
    //test_matmul();
	Matrix Xtrain, Xtest;
	std::vector<float> ytrain, ytest;
	
	load_data("train.txt", Xtrain, ytrain);
	load_data("test.txt", Xtest, ytest);

	std::cout << "train: " << Xtrain.NumRows() << ", " << Xtrain.NumCols() << "\n";
	std::cout << "test: " << Xtest.NumRows() << ", " << Xtest.NumCols() << "\n";
	std::cout << "train: " << ytrain.size()  << "\n";
	std::cout << "test: " << ytest.size() <<  "\n";
	
	//DecisionTree clf("entropy"); // max_depth is no limit, 0.9778
	DecisionTree clf("gini"); // max_depth is no limit, 0.9778
	///clf.fit(Xtrain, ytrain, 2000, Xtest, ytest);
	clf.Fit(Xtrain, ytrain);
	std::vector<float> y_pred = clf.predict(Xtest);
	float acc = clf.accuracy(y_pred, ytest);
	std::cout << "acc: " << acc << "\n"; // 0.9778
}

void test_regressor() {
    //test_matmul();
	Matrix Xtrain, Xtest;
	std::vector<float> ytrain, ytest;
	
	load_data("boston.train", Xtrain, ytrain);
	load_data("boston.test", Xtest, ytest);

	std::cout << "train: " << Xtrain.NumRows() << ", " << Xtrain.NumCols() << "\n";
	std::cout << "test: " << Xtest.NumRows() << ", " << Xtest.NumCols() << "\n";
	std::cout << "train: " << ytrain.size()  << "\n";
	std::cout << "test: " << ytest.size() <<  "\n";
	
	DecisionTree clf("mse"); // max_depth is no limit
	clf.Fit(Xtrain, ytrain);
	std::vector<float> y_pred = clf.predict(Xtest);
	float error = clf.MeanSquaredError(y_pred, ytest);
	std::cout << "error: " << error << "\n"; // 33.7251 vs 32.41637254901961
}
/*
int main(int argc, char** argv) {
	//test_classifier();
	test_regressor();
}*/

