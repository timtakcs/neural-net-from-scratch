#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include "functions.h"
#include "net.h"

using namespace std;
using namespace Eigen;

Net::Net(int epoch, int batch_s, float l_rate) {
	epochs = epoch;
	batch_size = batch_s;
	learning_rate = l_rate;
	init_weights();
}

void Net::init_weights() {
	W1 = Matrix<float, 100, 64>::Random() * sqrt(2 / 784);
	b1 = Vector<float, 64>::Random();
	
	W2 = Matrix<float, 64, 64>::Random() * sqrt(2 / 64);
	b2 = Vector<float, 64>::Random();
	
	Out = Matrix<float, 64, 10>::Random() * sqrt(1 / 64);
}

void Net::forward_pass(MatrixXf& data) {
	pass1 = data * W1;
	pass1 = pass1.rowwise() + b1.transpose();

	activation1 = pass1.unaryExpr(&relu);

	pass2 = activation1 * W2;
	pass2 = pass2.rowwise() + b2.transpose();

	activation2 = pass2.unaryExpr(&relu);

	pass_out = activation2 * Out;
	activation_out = softmax(pass_out);
}

void Net::backpropagate(VectorXf &labels_one_hot) {
	//TODO: using simple subtraction for loss, should really use cross entropy

	//out layer
	MatrixXf dz_out = activation_out.rowwise() - labels_one_hot.transpose();
	dw_out = (dz_out.transpose() * activation2) / labels_one_hot.size();

	//third layer
	MatrixXf dz_2 = Out * dz_out.transpose();
	dz_2 = dz_2.unaryExpr(&relu_derivative);
	dw_2 = (dz_2 * activation1) / labels_one_hot.size();
	db_2 = dz_2 / labels_one_hot.size();

	//second layer
	MatrixXf dz_1 = (W2 * dz_2);
	dz_1 = dz_1.unaryExpr(&relu_derivative);
	dw_1 = (dz_1 * Input) / labels_one_hot.size();
	db_1 = dz_1 / labels_one_hot.size();
}

void Net::adjust() {
	Out = Out - learning_rate * dw_out.transpose();

	W2 = W2 - learning_rate * dw_2.transpose();
	b2 = b2 - learning_rate * db_2;

	W1 = W1 - learning_rate * dw_1.transpose();
	b2 = b2 - learning_rate * db_1;
}

void Net::train(vector<Data> &train_data) {
	for (int i = 0; i < epochs; i++) {
		for (int j = 0; j < train_data.size(); j++) {
			Data data = train_data[j];
			MatrixXf input = data.get_image();
			VectorXf label = one_hot_encode(data.get_label());
			forward_pass(input);
			backpropagate(label);
			adjust();
		}
	}
}
