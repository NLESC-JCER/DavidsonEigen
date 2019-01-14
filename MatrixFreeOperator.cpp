#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Core>
#include "MatrixFreeOperator.hpp"

MatrixFreeOperator::MatrixFreeOperator(){}


// virtual here : get a row of the operator
Eigen::RowVectorXd MatrixFreeOperator::row(int index) const
{
	
    std::cout << "MatrixFreeOperator.row() not defined in class" << std::endl;
    exit(0);
}

// virtual here : get a col of the operator
Eigen::VectorXd MatrixFreeOperator::col(int index) const
{

    std::cout << "MatrixFreeOperator.col() not defined in class" << std::endl;
    exit(0);
}

Eigen::VectorXd MatrixFreeOperator::diagonal()
{
    Eigen::VectorXd D = Eigen::VectorXd::Zero(OpSizeVal,1);
    Eigen::RowVectorXd row_data;
    for(int i=0; i<OpSizeVal;i++)
    {
        row_data = this->row(i);
        D(i,0) = row_data(0,i);
    }
    return D;
}

// get the full matrix if we have to
Eigen::MatrixXd MatrixFreeOperator::get_full_mat()
{
	Eigen::MatrixXd matrix = Eigen::MatrixXd::Zero(OpSizeVal,OpSizeVal);
    for(int i=0; i<OpSizeVal; i++)
        matrix.row(i) = this->row(i);
    return matrix; 
}


// get the size
int MatrixFreeOperator::OpSize()
{
	return OpSizeVal;
}



