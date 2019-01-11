#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Core>
#include "DavidsonOperator.hpp"

// constructors
DavidsonOperator::DavidsonOperator(int size)
{
    OpSizeVal = size;
    OpEpsVal = 0.01;
    diag_el = Eigen::VectorXd::Random(size,1) + Eigen::VectorXd::Ones(size,1);
    diag_el *= 1E2;
} 


// private : get a row of the operator
Eigen::RowVectorXd DavidsonOperator::row(int index)
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

// private : get a col of the operator
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

Eigen::VectorXd DavidsonOperator::diagonal()
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
Eigen::MatrixXd DavidsonOperator::get_full_mat()
{
	Eigen::MatrixXd matrix = Eigen::MatrixXd::Zero(OpSizeVal,OpSizeVal);
    for(int i=0; i<OpSizeVal; i++)
        matrix.row(i) = this->row(i);
    return matrix; 
}


// get the size
int DavidsonOperator::OpSize()
{
	return OpSizeVal;
}



