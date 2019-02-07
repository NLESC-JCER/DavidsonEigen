#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Core>
#include "DavidsonOperator.hpp"
#include "MatrixFreeOperator.hpp"


// constructors
DavidsonOperator::DavidsonOperator(int size)
{
    _size = size;
    diag_el = Eigen::VectorXd::Random(size,1) + Eigen::VectorXd::Ones(size,1);
} 

// get a row of the operator
Eigen::RowVectorXd DavidsonOperator::row(int index) const
{
	Eigen::RowVectorXd row_out = Eigen::RowVectorXd::Zero(1,_size);    
    for (int j=0; j< _size; j++)
    {
        if (j==index) {
            row_out(j) = diag_el(j); 
        }
        else{
            row_out(j) = _sparsity / std::pow( static_cast<double>(j-index),2) ;
        }
    }
    return row_out;
}

//  get a col of the operator
Eigen::VectorXd DavidsonOperator::col(int index) const
{
    Eigen::VectorXd col_out = Eigen::VectorXd::Zero(_size,1);    
    for (int j=0; j < _size; j++)
    {
        if (j==index) {
            col_out(j) = diag_el(j); 
        }
        else{
            col_out(j) = _sparsity / std::pow( static_cast<double>(j-index),2) ;
        }
    }
    return col_out;
}



