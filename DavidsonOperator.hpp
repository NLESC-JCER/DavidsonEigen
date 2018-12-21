#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Core>

typedef Eigen::Matrix<double,Eigen::Dynamic,1> Vect;
typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> Mat;
typedef Eigen::Matrix<double,1,Eigen::Dynamic> RowVect;

#ifndef _DAVIDSON_OP_
#define _DAVIDSON_OP_

class DavidsonOperator
{
	public: 

		DavidsonOperator(int size);

		Vect apply_to_vect(Vect b);
		Mat apply_to_mat(Mat m);
		Mat get_full_mat();

		Vect diagonal();
		RowVect row(int index);
		int OpSize();


	private:

		int OpSizeVal;
		double OpEpsVal;
		Vect diag_el;
		
};
#endif