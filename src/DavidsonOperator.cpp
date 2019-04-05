#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Core>
#include "DavidsonOperator.hpp"
#include "MatrixFreeOperator.hpp"


// constructors
DavidsonOperator::DavidsonOperator(int size, double eps, bool odiag)
{
    _size = size;
    _odiag = odiag;
    _sparsity = eps;

    diag_el = Eigen::VectorXd(_size);
    for (int i=0; i<_size;i++){
        if (_odiag) diag_el(i) = static_cast<double> (i+1);
        else diag_el(i) = static_cast<double> (1. + (std::rand() %1000 ) / 10.);
    }
    
} 

Eigen::ArrayXd DavidsonOperator::_sort_index(Eigen::VectorXd& V) const
{
    Eigen::ArrayXd idx = Eigen::ArrayXd::LinSpaced(V.rows(),0,V.rows()-1);
    std::sort(idx.data(),idx.data()+idx.size(),
              [&](int i1, int i2){return V[i1]<V[i2];});
    return idx; 
}

//  get a col of the operator
Eigen::VectorXd DavidsonOperator::col(int index_orig) const
{
    int index = index_orig;
    
    Eigen::VectorXd col_out = Eigen::VectorXd::Zero(_size,1);    
    for (int j=0; j < _size; j++)
    {
        if (j==index) {
            col_out(j) =  diag_el(j); 
        }
        else{
            col_out(j) = _sparsity / std::pow( static_cast<double>(j-index),2) ;
        }
    }

    return col_out;

}



