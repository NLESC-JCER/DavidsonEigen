#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Core>
#include "MatrixFreeOperator.hpp"

#ifndef _DAVIDSON_OP_
#define _DAVIDSON_OP_

class DavidsonOperator : public MatrixFreeOperator
{
	public: 

		DavidsonOperator(int n, double eps, bool d);
		Eigen::VectorXd col(int index) const;

	private:
		double _sparsity = 0.1;
		bool _odiag = false;
};

#endif

