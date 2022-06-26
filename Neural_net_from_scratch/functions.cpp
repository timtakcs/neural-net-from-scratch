#include <Eigen/Dense>
#include "functions.h"

using namespace Eigen;

float cross_entropy(Matrix<float, 1, 10> out, Vector<float, 10> label) {
	out = out.array();
	label = label.array();

	float sum = 0;

	for (int i = 0; i < out.size(); i++) {
		sum = sum + (label[i] * log(out[i]));
	}
	return -1 * sum;
}

float relu(float x) {
	return std::max(float(0.0), x);
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