#include <iostream> 
#include <Eigen/Dense>
#include "net.h"

int main() {
	Net net(100, 10, 0.001);
	MatrixXf inp = MatrixXf::Random(1, 100);

	net.forward_pass(inp);
	Vector<float, 10> labels;

	labels << 0, 0, 1, 0, 0, 0, 0, 0, 0, 0;
	net.get_adjustments(labels);
	net.backpropagate();
}