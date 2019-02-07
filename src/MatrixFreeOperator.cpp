#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Core>
#include "MatrixFreeOperator.hpp"

MatrixFreeOperator::MatrixFreeOperator(){}

// virtual here : get a row of the operator
Eigen::RowVectorXd MatrixFreeOperator::row(int index) const
{
   throw std::runtime_error("MatrixFreeOperator.row() not defined in class");
}

// virtual here : get a col of the operator
Eigen::VectorXd MatrixFreeOperator::col(int index) const
{
    throw std::runtime_error("MatrixFreeOperator.col() not defined in class");
}

Eigen::VectorXd MatrixFreeOperator::diagonal() const
{
    Eigen::VectorXd D = Eigen::VectorXd::Zero(_size,1);
    Eigen::RowVectorXd row_data;
    for(int i=0; i<_size;i++) {
        row_data = this->row(i);
        D(i) = row_data(i);
    }
    return D;
}

// get the full matrix if we have to
Eigen::MatrixXd MatrixFreeOperator::get_full_mat() const
{
	Eigen::MatrixXd matrix = Eigen::MatrixXd::Zero(_size,_size);
    for(int i=0; i<_size; i++){
        matrix.row(i) = this->row(i);
    }
    return matrix; 
}



