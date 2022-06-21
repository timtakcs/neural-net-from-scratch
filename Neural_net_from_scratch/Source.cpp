#include <Eigen/Dense>
#include <iostream>

using namespace std;
using namespace Eigen;

float cross_entropy(Matrix<float, 1, 10> out, Vector<float, 10> label) {
	out = out.array();
	label = label.array();

	float sum = 0;

	for (int i = 0; i < out.size(); i++) {
		label[i] * log(out[i]);
	}

	return -1 * sum;
}

float relu(float x) {
	return max(float(0.0), x);
}

Matrix<float, 1, 10> softmax(Matrix<float, 1, 10> out) {
	out = out.array();
	float sum = 0.0;

	for (auto& x : out) {
		sum = sum + exp(x);
	}

	for (int i = 0; i < out.size(); i++) {
		out[i] = exp(out[i]) / sum;
	}

	out = out.matrix();

	return out;
}

class Net {
public:
	void init_weights() {
		//do this tomorrow i cba
		//use kaiming initialization
	}

	Matrix<float, 1, 10> forward_pass(MatrixXf &data) {
		data = data * W1;
		data = data.rowwise() + b1.transpose();

		data = data.unaryExpr(&relu);

		data = data * W2;
		data = data.rowwise() + b2.transpose();
	
		data = data.unaryExpr(&relu);

		data = data * Out;
		data = softmax(data);

		return data;
	}

private:
	//input layer
	Matrix<float, 1, 100> Input;

	//first hidden layer
	Matrix<float, 100, 64> W1 = Matrix<float, 100, 64>::Random();
	Vector<float, 64> b1 = Vector<float, 64>::Random();

	//second hidden layer
	Matrix<float, 64, 64> W2 = Matrix<float, 64, 64>::Random();
	Vector<float, 64> b2 = Vector<float, 64>::Random();

	//output layer
	Matrix<float, 64, 10> Out = Matrix<float, 64, 10>::Random();
};

int main() {
	Net net;

	MatrixXf inp = MatrixXf::Random(1, 100);

	Matrix<float, 1, 10> out = net.forward_pass(inp);

}