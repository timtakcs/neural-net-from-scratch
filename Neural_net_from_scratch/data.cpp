#include <fstream>
#include <iostream>
#include <Eigen/Dense>
#include "data.h"

using namespace Eigen;

Data::Data(MatrixXf& image, int label) {
	img = image;
	lbl = label;
}

void Data::flatten() {
	//
}

int Data::get_label() {
	return lbl;
}
