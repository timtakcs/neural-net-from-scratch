#include <Eigen/Dense>
#include <iostream>

using namespace std;
using namespace Eigen;

class Net {
public:
	Net() {
	}

	void init_weights() {
		//do this tomorrow i cba
		//use kaiming initialization
	}

	Matrix<float, 1, 10> forward_pass(MatrixXd &data) {
		data = data * W1;
		data = data.rowwise() + b1.transpose();

		data.unaryExpr(&relu);

		data = data * W2;
		data = data.rowwise() + b2.transpose();

		data.unaryExpr(&relu);

		data = data * Out;
		data = softmax(data);

		return data;
	}

	Matrix<float, 1, 10> softmax(Matrix<float, 1, 10> out) {
		out = out.array();
		double sum = 0.0;

		for (auto& x : out) {
			sum = sum + exp(x);
		}

		for (int i = 0; i < out.size(); i++) {
			out[i] = exp(out[i]) / sum;
		}

		Matrix<float, 1, 10> result;
		
		for (auto& z : out) {
			result << z;
		}

		return result;
	}

	double relu(double x) {
		return max(0.0, x);
	}
private:
	//input layer
	Matrix<float, 1, 100> Input;

	//first hidden layer
	Matrix<float, 100, 64> W1;
	Vector<float, 64> b1;

	//second hidden layer
	Matrix<float, 64, 64> W2;
	Vector<float, 64> b2;

	//output layer
	Matrix<float, 64, 10> Out;
};