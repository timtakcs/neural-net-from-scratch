#include <fstream>
#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <string>
#include "data.h"
#include "loader.h"

using namespace Eigen;


std::vector<Data> Loader::load_train() {
	std::ifstream train;
	train.open(train_path);

	std::vector<std::vector<float>> temp_image;
	MatrixXf img_mx(28, 28);
	std::vector<Data> data;

	std::string label;
	std::getline(train, label, ',');

	while (train.good()) {
		for (int i = 0; i < 28; i++) {
			for (int j = 0; j < 28; j++) {
				std::string line;
				std::getline(train, line, ',');
				temp_image.at(i).push_back(std::stof(line));
			}
		}

		for (int i = 0; i < 28; i++) {
			img_mx.row(i) = VectorXf::Map(&temp_image[i][0], temp_image[i].size());
		}

		data.push_back(Data(img_mx, std::stoi(label)));
	}
	return data;
}

std::vector<Data> Loader::load_test() {
	std::ifstream test;
	test.open(test_path);

	std::vector<std::vector<float>> temp_image;
	MatrixXf img_mx(28, 28);
	std::vector<Data> data;

	std::string label;
	std::getline(test, label, ',');

	while (test.good()) {
		for (int i = 0; i < 28; i++) {
			for (int j = 0; j < 28; j++) {
				std::string line;
				std::getline(test, line, ',');
				temp_image.at(i).push_back(std::stof(line));
			}
		}

		for (int i = 0; i < 28; i++) {
			img_mx.row(i) = VectorXf::Map(&temp_image[i][0], temp_image[i].size());
		}

		data.push_back(Data(img_mx, std::stoi(label)));
	}
	return data;
}
