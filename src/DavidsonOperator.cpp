#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Core>
#include "DavidsonOperator.hpp"
#include "MatrixFreeOperator.hpp"


// constructors
DavidsonOperator::DavidsonOperator(int size, bool odiag)
{
    _size = size;
    _odiag = odiag;

    diag_el = Eigen::VectorXd(_size);
    for (int i=0; i<_size;i++)
        diag_el(i) = static_cast<double> (1. + (std::rand() %1000 ) / 10.);

} 

// get a row of the operator
Eigen::RowVectorXd DavidsonOperator::row(int index) const
{
	Eigen::RowVectorXd row_out = Eigen::RowVectorXd::Zero(1,_size);    
    for (int j=0; j< _size; j++)
    {
        if (j==index) {
            if(_odiag) row_out(j) = static_cast<double> (j+1);
            else row_out(j) = diag_el(j); 
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
            if (_odiag) col_out(j) = static_cast<double> (j+1);
            else col_out(j) =  diag_el(j); 
        }
        else{
            col_out(j) = _sparsity / std::pow( static_cast<double>(j-index),2) ;
        }
    }
    return col_out;
}



