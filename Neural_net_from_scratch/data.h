#pragma once
#include <fstream>
#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;

class Data {
private:
	MatrixXf img;
	int lbl;

public:
	Data(MatrixXf& image, int label);
	void flatten();
	int get_label();
	MatrixXf get_image();
};