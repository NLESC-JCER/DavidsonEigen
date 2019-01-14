#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Core>
#include "DavidsonOperator.hpp"
#include "MatrixFreeOperator.hpp"


// constructors
DavidsonOperator::DavidsonOperator(int size)
{
    OpSizeVal = size;
    OpEpsVal = 0.01;
    diag_el = Eigen::VectorXd::Random(size,1) + Eigen::VectorXd::Ones(size,1);
    diag_el *= 1E2;
} 

// get a row of the operator
Eigen::RowVectorXd DavidsonOperator::row(int index) const
{
	Eigen::RowVectorXd row_out = Eigen::RowVectorXd::Zero(1,OpSizeVal);    
    for (int j=0; j< OpSizeVal; j++)
    {
        if (j==index)
        {
            row_out(0,j) = diag_el(j); 
        }
        else
            row_out(0,j) = 0.01 / std::pow( static_cast<double>(j-index),2) ;
    }
    return row_out;
}

//  get a col of the operator
Eigen::VectorXd DavidsonOperator::col(int index) const
{
    Eigen::VectorXd col_out = Eigen::VectorXd::Zero(OpSizeVal,1);    
    for (int j=0; j < OpSizeVal; j++)
    {
        if (j==index)
        {
            col_out(j,0) = diag_el(j); 
        }
        else
            col_out(j,0) = 0.01 / std::pow( static_cast<double>(j-index),2) ;
    }
    return col_out;
}



