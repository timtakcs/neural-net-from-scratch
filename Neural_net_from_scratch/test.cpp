//#include <Eigen/Dense>
//#include <iostream>
//
//using namespace std;
//using namespace Eigen;
//
//double relu(double z) {
//	return max(0.0, z);
//}
//
//double softmax(double z, double total) {
//	return exp(z) / total;
//}
//
//int main() {
//	//input layer
//	MatrixXd Input = MatrixXd::Random(1, 100);
//
//	//first hidden layer
//	Matrix<double, 100, 64> W1 = Matrix<double, 100, 64>::Random();
//	Vector<double, 64> b1 = Vector<double, 64>::Random();
//
//	Matrix<double, 1, 64> n = Input * W1;
//	n = n.rowwise() + b1.transpose();
//	double mean = 1.0;
//	cout << n.unaryExpr(&relu);
//
//	/*n = relu(n.array());*/
//	
//	//second hidden layer
//	Matrix<double, 64, 64> W2;
//	Vector<double, 64> b2;
//
//	//output layer
//	Matrix<double, 64, 10> Out;
//}
//
