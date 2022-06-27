#pragma once
#include <fstream>
#include <Eigen/Dense>
#include <vector>
#include <string>
#include "data.h"

using namespace Eigen;

class Loader {
public:
	std::vector<Data> load_train();
	std::vector<Data> load_test();
private:
	std::string train_path = "mnist_train.csv";
	std::string test_path = "mnist_test.csv";
};