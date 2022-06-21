//#include<iostream>
//#include<vector>
//#include<Eigen/Dense>
//
//using namespace Eigen;
//using namespace std;
//
//int main() {
//	//dynamic and resizable matrix
//	MatrixXd matrix;
//
//	//static and fixed matrix
//	Matrix2d new_mat;
//
//	Matrix3Xd new_new_mat;
//
//	new_new_mat.resize(3, 15);
//
//	new_mat = Matrix2d::Random();
//
//	//fixed size matrix is faster - so if you, can use it
//	/*cout << "size " << new_new_mat.size();
//	cout << new_mat << endl;*/ 
//	
//	Matrix<double, 3, 3> m;
//	m << 1, 2, 7,
//		3, 5, 9,
//		0, 7, 1;
//
//	Matrix<double, 1, 3> n;
//	n << 1, 2, 3;
//
//	MatrixXd x = n * m;
//
//	/*cout << m << endl;
//	cout << n << endl;*/
//	cout << x << endl;
//}
