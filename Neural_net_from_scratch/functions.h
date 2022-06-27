#pragma once
#include <Eigen/Dense>

using namespace Eigen;

float cross_entropy(Matrix<float, 1, 10> out, Vector<float, 10> label);
float relu(float x);
float relu_derivative(float x);
MatrixXf softmax(MatrixXf out);
VectorXf one_hot_encode(int lbl);