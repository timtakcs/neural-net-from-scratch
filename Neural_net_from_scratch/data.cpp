#include <fstream>
#include <iostream>
#include <Eigen/Dense>
#include "data.h"

using namespace Eigen;

Data::Data(MatrixXf& image, int label) {
	flatten();
	img = image;
	lbl = label;
}

void Data::flatten() {
	img.resize(1, 784);
}

int Data::get_label() {
	return lbl;
}

MatrixXf Data::get_image() {
	return img;
}
