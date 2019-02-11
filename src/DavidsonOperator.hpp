#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Core>
#include "MatrixFreeOperator.hpp"

#ifndef _DAVIDSON_OP_
#define _DAVIDSON_OP_

class DavidsonOperator : public MatrixFreeOperator
{
	public: 

		DavidsonOperator(int n);
		Eigen::RowVectorXd row(int index) const;
		Eigen::VectorXd col(int index) const;

	private:
		double _sparsity = 0.1;
};

#endif

