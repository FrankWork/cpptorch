#include <iostream>
#include <vector>
#include "logistic_regression.h"
#include <cmath>
#include <string>
#include <fstream>
#include <unordered_set>

void test_matmul() {
    Matrix<int> a( {
        {1,2,3},
        {4,5,6}
    });
    a.Show();
    Matrix<int> b({
        {7,8},
        {9,10},
        {11,12}
    });
    b.Show();
    Matrix<int> c = a.MatMul(b);
    c.Show();

}

constexpr int64_t SIGMOID_TABLE_SIZE = 512;
constexpr int64_t MAX_SIGMOID = 8;
constexpr int64_t LOG_TABLE_SIZE = 512;

LogisticRegression::LogisticRegression(float learning_rate): 
	learning_rate(learning_rate), bias(0), lambda(0.1) {
	_t_sigmoid.reserve(SIGMOID_TABLE_SIZE + 1);
  	for (int i = 0; i < SIGMOID_TABLE_SIZE + 1; i++) {
    float x = float(i * 2 * MAX_SIGMOID) / SIGMOID_TABLE_SIZE - MAX_SIGMOID;
    _t_sigmoid.push_back(1.0 / (1.0 + std::exp(-x)));
  }
  
	_t_log.reserve(LOG_TABLE_SIZE + 1);
  for (int i = 0; i < LOG_TABLE_SIZE + 1; i++) {
    float x = (float(i) + 1e-5) / LOG_TABLE_SIZE;
    _t_log.push_back(std::log(x));
  }
}

template<class T>
void LogisticRegression::fit(const Matrix<T>& features, 
    const std::vector<int>&labels, int num_epochs,
    const Matrix<T>& test_features, 
    const std::vector<int>&test_labels){
	std::unordered_set<int> labels_set(labels.begin(), labels.end());
    n_labels = labels_set.size();

    size_t n_sample = features.Rows();
    size_t hidden_size = features.Cols();
	std::cout << n_sample << " " << hidden_size << "\n";

	//weight.Initialize(hidden_size, 1);
	std::minstd_rand rng(0);
  	std::uniform_real_distribution<> uniform(-1./hidden_size, 1./hidden_size);
	weight.resize(hidden_size);
	for(int j=0;j<hidden_size;++j){
		weight[j] = uniform(rng);
	}

    
	T prev_loss = 100000;
	for(int i=0;i < num_epochs; ++i) {
	/*	std::cout << "weight: ";
		for(int j=0;j<hidden_size;++j) {
			std::cout << weight.get(j, 0) << " ";
		}
		std::cout << "\n";
*/

		std::vector<T> hidden = features.MatMul(this->weight);// + this->bias;
		for(int i=0;i < hidden.size();++i){
			hidden[i] += this->bias;
		}
		std::vector<T> logits = sigmoid(hidden);
		//std::cout << "hidden shape" << hidden.Shape() << "\n";
        /*
		std::cout << "hidden: ";
		for(int j=0; j< 10;++j) {
			std::cout << hidden.get(j, 0) << " ";
		}
		std::cout << "\n";
		//std::cout << hidden.Shape() << "\n";

		std::cout << "logits: ";
		for(int j=0; j< 10;++j) {
			std::cout << logits.get(j, 0) << "/" << labels[j] << " ";
		}
		std::cout << "\n";
		//std::cout << logits.Shape() << "\n";
*/


		T loss = nll_loss(logits, labels);
		backward(features, logits, labels);
	    float acc = accuracy(logits, labels);
	    //std::cout << "acc: " << acc << "\n";
		std::cout << "epoch: " << i <<  " loss: " << loss <<  " acc: " << acc
        << "\n";
		std::vector<T> test_logits = predict_proba(test_features);
        float test_acc = accuracy(test_logits, test_labels);
	    std::cout << "test_acc: " << test_acc << "\n";
		if (prev_loss - loss < 1e-7) {
			//break;
		}
		prev_loss = loss;

	}
}

template<class T>
std::vector<T> LogisticRegression::predict(const Matrix<T>& features){
		std::vector<T> hidden = features.MatMul(this->weight);// + this->bias;
		for(int i=0;i < hidden.size();++i){
			hidden[i] += this->bias;
		}
	
	return hidden;
}

template<class T>
std::vector<T> LogisticRegression::predict_proba(const Matrix<T>& features) {
		std::vector<T> hidden = features.MatMul(this->weight);// + this->bias;
		for(int i=0;i < hidden.size();++i){
			hidden[i] += this->bias;
		}
		std::vector<T> logits = sigmoid(hidden);
	return logits;
}

template<class T>
void LogisticRegression::backward(const Matrix<T>&features, const std::vector<T>& logits, 
		const std::vector<int>& labels) {
	std::vector<T> grad_w(weight.size());
	//grad_w.Zero();
	Zero(grad_w);
	T grad_b = 0;

	for(int i=0;i < labels.size(); ++i) {
		for(int j=0; j < features.Cols(); ++j) {
			T grad = (logits[i] - labels[i])*features.get(i,j);
			//std::cout << "grad: " << grad << "\n";
			if (grad > 10) grad = 10;
			if (grad < -10) grad = -10;
			grad_w[j] += grad;
		}
		grad_b += (logits[i] - labels[i]);
	}

	for(int i=0;i < weight.size(); ++i) {
			weight[i] -= learning_rate*(grad_w[i]+2*lambda*weight[i]);
	}
	bias = bias-learning_rate*(grad_b+2*lambda*bias);
	
}

template<class T>
T LogisticRegression::sigmoid(const T& x) {
  if (x < -MAX_SIGMOID) {
    return 0.0;
  } else if (x > MAX_SIGMOID) {
    return 1.0;
  } else {
    int64_t i =
        int64_t((x + MAX_SIGMOID) * SIGMOID_TABLE_SIZE / MAX_SIGMOID / 2);
    return _t_sigmoid[i];
  }
}

template<class T>
T LogisticRegression::log(const T& x) {
  if (x > 1.0) {
    return 0.0;
  }
  int64_t i = int64_t(x * LOG_TABLE_SIZE);
  return _t_log[i];
}


template<class T>
std::vector<T> LogisticRegression::sigmoid(const std::vector<T>& hidden){
	std::vector<T> res(hidden.size());
	
	for(int i=0;i<hidden.size();++i) {
		res[i] = sigmoid(hidden[i]);
	}
	return res;
}

template<class T>
float LogisticRegression::nll_loss(const std::vector<T>& logits, const
	std::vector<int>& labels) {
	T loss_sum = 0;
	//std::cout << "labels size: " << labels.size() << "\n";
	for(int i=0;i < labels.size(); ++i) {
		if(labels[i]) {
			loss_sum -= log(logits[i]);
			//std::cout << "loss: " <<  log(logits.get(i, 0)) << "\n";
		} else {
			loss_sum -= log(1-logits[i]);
			//std::cout << "loss: " << log(1-logits.get(i, 0)) << "\n";
		}
	}
	loss_sum += lambda * l2_norm(weight);
	loss_sum += lambda * bias * bias;
	return loss_sum / labels.size();
}


template<class T>
T LogisticRegression::l2_norm(const std::vector<T>& vec) {
	T res = 0;
	for(int i=0;i < vec.size();++i) {
		res += vec[i]*vec[i];
	}
	return res;
}

template<class T>
void LogisticRegression::Zero( std::vector<T>& vec) {
	
	for(int i=0;i < vec.size();++i) {
		vec[i] = 0;
	}
	//return res;
}


template<class T>
float LogisticRegression::accuracy(const std::vector<T>& logits, const
	std::vector<int>& labels, float threshold) {
	float n_correct = 0;
	//std::cout << logits.Shape() << "\n";
	for(int i=0;i < labels.size(); ++i) {
		if (logits[i] > threshold) {
			n_correct += 1;
		}
	}
	
	return n_correct / labels.size();
}

void load_data(const std::string& filename, Matrix<float>& features, std::vector<int>& labels) {
	std::ifstream ifs(filename);
	std::string line;
	std::string word;
	std::vector<std::string> parts;
	while(getline(ifs, line)) {
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
	Matrix<float> Xtrain, Xtest;
	std::vector<int> ytrain, ytest;
	
	load_data("train.txt", Xtrain, ytrain);
	load_data("test.txt", Xtest, ytest);

	std::cout << "train: " << Xtrain.Rows() << ", " << Xtrain.Cols() << "\n";
	std::cout << "test: " << Xtest.Rows() << ", " << Xtest.Cols() << "\n";
	float lr = 0.001;
	if (argc >= 2) {
		lr = std::stof(std::string(argv[1]));
		std::cout << "lr :" << lr << "\n";
	}
	LogisticRegression clf(lr);
	clf.fit(Xtrain, ytrain, 1000, Xtest, ytest);
	std::vector<float> logits = clf.predict_proba(Xtest);
	float acc = clf.accuracy(logits, ytest);
	std::cout << "acc: " << acc << "\n";
}
