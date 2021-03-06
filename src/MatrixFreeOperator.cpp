#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Core>
#include "MatrixFreeOperator.hpp"

MatrixFreeOperator::MatrixFreeOperator(){}


// virtual here : get a col of the operator
Eigen::VectorXd MatrixFreeOperator::col(int index) const
{
    throw std::runtime_error("MatrixFreeOperator.col() not defined in class");
}

Eigen::VectorXd MatrixFreeOperator::diagonal() const
{
    Eigen::VectorXd D = Eigen::VectorXd::Zero(_size,1);
    Eigen::VectorXd col_data;
    for(int i=0; i<_size;i++) {
        col_data = this->col(i);
        D(i) = col_data(i);
    }
    return D;
}

// get the full matrix if we have to
Eigen::MatrixXd MatrixFreeOperator::get_full_mat() const
{
	Eigen::MatrixXd matrix = Eigen::MatrixXd::Zero(_size,_size);

    #pragma openmp parallel for
    for(int i=0; i<_size; i++){
        matrix.col(i) = this->col(i);
    }
    return matrix; 
}



