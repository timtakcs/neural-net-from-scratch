#pragma once
#include <Eigen/Dense>
#include <vector>

using namespace Eigen;

class Net {
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

	int epochs;
	int batch_size;
	float learning_rate;

public:
	Net(int epochs, int batch_size, float learning_rate);
	void init_weights();
	void forward_pass(MatrixXf& data);
	void get_adjustments(Vector<float, 10>& labels_one_hot);
	void backpropagate();
	/*void train(std::vector<MatrixXf> batch);*/
};