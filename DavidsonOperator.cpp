#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Core>
#include "DavidsonOperator.hpp"

typedef Eigen::Matrix<double,Eigen::Dynamic,1> Vect;
typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> Mat;
typedef Eigen::Matrix<double,1,Eigen::Dynamic> RowVect;

// constructors
DavidsonOperator::DavidsonOperator(int size)
{
    OpSizeVal = size;
    OpEpsVal = 0.01;
    diag_el = Vect::Random(size,1) + Vect::Ones(size,1);
    diag_el *= 1E2;
} 


// private : get a row of the operator
RowVect DavidsonOperator::row(int index)
{
	RowVect row_out = RowVect::Zero(1,OpSizeVal);    
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
Vect DavidsonOperator::col(int index) const
{
    Vect col_out = Vect::Zero(OpSizeVal,1);    
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

Vect DavidsonOperator::diagonal()
{
    Vect D = Vect::Zero(OpSizeVal,1);
    RowVect row_data;
    for(int i=0; i<OpSizeVal;i++)
    {
        row_data = this->row(i);
        D(i,0) = row_data(0,i);
    }
    return D;
}

// get the full matrix if we have to
Mat DavidsonOperator::get_full_mat()
{
	Mat matrix = Mat::Zero(OpSizeVal,OpSizeVal);
    for(int i=0; i<OpSizeVal; i++)
        matrix.row(i) = this->row(i);
    return matrix; 
}


// get the size
int DavidsonOperator::OpSize()
{
	return OpSizeVal;
}



