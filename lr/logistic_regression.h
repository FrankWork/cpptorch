#include <iostream>
#include <vector>
#include <random>

#pragma monce

template<class T>
class Matrix {
public:
	Matrix():_n_row(0), _n_col(0){}
    Matrix(size_t n_row, size_t n_col):_n_row(n_row), _n_col(n_col),
        data(n_row, std::vector<T>(n_col)){
    };
    Matrix(const std::vector<std::vector<T>>& vec2d) {
        _n_row=0;
        for(auto it=vec2d.begin(); it!=vec2d.end(); ++it){
            this->data.push_back(*it);
            ++_n_row;
        }
        _n_col = data[0].size();
    }
    Matrix<T> operator+(const std::vector<T>& bias) const {
        Matrix<T> res(_n_row, _n_col);
        for(int i=0;i <_n_row;++i) {
            for(int j=0;j<_n_col;++j) {
                res.data[i][j] = data[i][j] + bias[i];
            }
        }
        return res;
    }
	
	Matrix<T> operator+(const T& bias) const {
        Matrix<T> res(_n_row, _n_col);
        for(int i=0;i <_n_row;++i) {
            for(int j=0;j<_n_col;++j) {
                res.data[i][j] = data[i][j] + bias;
            }
        }
        return res;
    }

	void push_back(const std::vector<T>& vec) {
		data.push_back(vec);
		_n_row ++;
		if (_n_col == 0){
			_n_col = vec.size();
		} else if (_n_col != vec.size()) {
			std::cout << "Error column size, expect " << _n_col << ", get " << vec.size() << "\n";
		}
	}

	T get(size_t i, size_t j) const {
		return data[i][j];
	}

	void set(size_t i, size_t j, T val) {
		data[i][j] = val;
	}

    void Initialize(size_t n_row, size_t n_col) {
        _n_row = n_row;
        _n_col = n_col;
        data.resize(_n_row);

		std::minstd_rand rng(0);
  		std::uniform_real_distribution<> uniform(-1./_n_row, 1./_n_row);
		std::cout << "uniform: " << -1./_n_row << "\n";
        for(int i=0;i < _n_row; ++i){
            data[i].resize(_n_col);
			for(int j=0;j<_n_col;++j){
				data[i][j] = uniform(rng);
			}
        }
    }

    Matrix<T> MatMul(const Matrix<T>& b) const {
        Matrix<T> result(_n_row, b.Cols());
        for(int i=0;i < _n_row; ++i){
            for(int j=0;j < result.Cols();++j){
                result.data[i][j] = 0;
                for(int k=0;k < _n_col; ++k) {
                    result.data[i][j] += (data[i][k] * b.data[k][j]);
                }
            }
        }
        return result;
    }
    size_t Rows() const{return _n_row;}
    size_t Cols() const {return _n_col;}
	std::string Shape() const {
		std::string tmp;
		tmp += ( "(" +std::to_string(_n_row) +  ", " + std::to_string(_n_col) + ")");
		return tmp;
	}

    void Show() {
        for(int i=0;i<_n_row;++i){
            for(int j=0;j<_n_col;++j){
            std::cout << data[i][j] << " ";
            }
            std::cout <<"\n";
        }
    }

    ~Matrix(){}
private:
    size_t _n_row;
    size_t _n_col;
    std::vector<std::vector<T>>  data;
};


class LogisticRegression {

public:

    LogisticRegression(float lr);

    template<class T>
    void fit(const Matrix<T>& features, const std::vector<int>& labels, int
    num_epochs, const Matrix<T>& test_features, const std::vector<int>&
    test_labels);
    
    template<class T>
    Matrix<T> predict(const Matrix<T>& features);
    
    template<class T>
    Matrix<T> predict_proba(const Matrix<T>& features);
    
	template<class T>
	void backward(const Matrix<T>&features, const Matrix<T>& logits, const std::vector<int>& labels);

    template<class T>
    float accuracy(const Matrix<T>& logits, const std::vector<int>& labels, float threshold=0.5);

    template<class T>
    float nll_loss(const Matrix<T>& logits, const std::vector<int>& labels);

	template<class T>
	Matrix<T> sigmoid(const Matrix<T>& hidden);

	template<class T>
	T sigmoid(const T& x);

	template<class T>
	T log(const T& x);
    
    ~LogisticRegression(){}

private:
    float learning_rate;
    Matrix<float> weight;
    float bias;
    std::vector<float> _t_sigmoid;
    std::vector<float> _t_log;
    int n_labels;
};


