#include <Eigen/Dense>
#include <iostream>

using namespace std;
using namespace Eigen;

float cross_entropy(Matrix<float, 1, 10> out, Vector<float, 10> label) {
	out = out.array();
	label = label.array();

	float sum = 0;

	for (int i = 0; i < out.size(); i++) {
		sum = sum + (label[i] * log(out[i]));
	}
	return -1 * sum, 84;
}

float relu(float x) {
	return max(float(0.0), x);
}

float relu_derivative(float x) {
	if (x > 0) return 1.0;
	else return 0.0;
}

MatrixXf softmax(MatrixXf out) {
	out = out.array();
	float sum = 1.0;

	for (int j = 0; j < out.size(); j++) {
		sum = sum + exp(out(j));
	}

	for (int i = 0; i < out.size(); i++) {
		out(i) = exp(out(i)) / sum;
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

	void forward_pass(MatrixXf &data) {
		pass1 = data * W1;
		pass1 = pass1.rowwise() + b1.transpose();

		activation1 = pass1.unaryExpr(&relu);

		pass2 = activation1 * W2;
		pass2 = pass2.rowwise() + b2.transpose();
	
		activation2 = pass2.unaryExpr(&relu);

		pass_out = activation2 * Out;
		activation_out = softmax(pass_out);
	}

	void get_adjustments(Vector<float, 10> &labels_one_hot) {
		//using simple subtraction for loss, should really use cross entropy

		//out layer
		MatrixXf dz_out = activation_out.rowwise() - labels_one_hot.transpose();
		dw_out = (dz_out.transpose() * activation2) / labels_one_hot.size();
		//this might be activation2
		MatrixXf p2_inv = pass2.unaryExpr(&relu_derivative);

		//third layer
		MatrixXf dz_2 = (Out * dz_out.transpose()) * p2_inv;
		dw_2 = (dz_2 * activation1.transpose()) / labels_one_hot.size();
		db_2 = dz_2 / labels_one_hot.size();
		//this might be activation1
		MatrixXf p1_inv = pass1.unaryExpr(&relu_derivative);

		//second layer
		MatrixXf dz_1 = (W2 * dz_2) * p1_inv.transpose();
		dw_1 = (dz_1 * Input) / labels_one_hot.size();
		db_1 = dz_1 / labels_one_hot.size();
	}

	void backpropagation() {
		Out = Out - learning_rate * dw_out;

		W2 = W2 - learning_rate * dw_2;
		b2 = b2 - learning_rate * db_2;

		W1 = W1 - learning_rate * dw_1;
		b2 = b2 - learning_rate * db_1;
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

	//variables for propagating through the net
	MatrixXf pass1;
	MatrixXf activation1;
	MatrixXf pass2;
	MatrixXf activation2;
	MatrixXf pass_out;
	MatrixXf activation_out;

	//variables for getting the difference for weights and biases
	MatrixXf dw_out;
	MatrixXf dw_2;
	MatrixXf db_2;
	MatrixXf dw_1;
	MatrixXf db_1;

	//hyperparams
	int epochs = 100;
	int batch_size = 10;
	float learning_rate = 0.001;
};

int main() {
	Net net;
	MatrixXf inp = MatrixXf::Random(1, 100);

	net.forward_pass(inp);
	Vector<float, 10> labels;

	labels << 0, 0, 1, 0, 0, 0, 0, 0, 0, 0;
	net.get_adjustments(labels);
}