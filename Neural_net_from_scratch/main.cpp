#include <iostream> 
#include <Eigen/Dense>
#include "net.h"
#include "loader.h"
#include "data.h"

using namespace std;

int main() {
	Net net(100, 10, 0.001);
	Loader loader;
	vector<Data> x = loader.load_train();
	vector<Data> test = loader.load_test();

	net.train(x);
}