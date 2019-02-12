
#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE davidson_test

#include <boost/test/unit_test.hpp>

#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/QR>
#include <Eigen/Eigenvalues>
#include <chrono>

#include "../src/DavidsonSolver.hpp"
#include "../src/DavidsonOperator.hpp"
#include "../src/MatrixFreeOperator.hpp"

// intiialize a full matrix 
Eigen::MatrixXd init_matrix(int N, double eps, bool diag)
{
    Eigen::MatrixXd matrix;
    matrix =  eps * Eigen::MatrixXd::Random(N,N);
    Eigen::MatrixXd tmat = matrix.transpose();
    matrix = matrix + tmat; 

    for (int i = 0; i<N; i++) {
        if(diag)    matrix(i,i) = static_cast<double> (i+1);   
        else        matrix(i,i) =  static_cast<double> (1. + (std::rand() %1000 ) / 10.);
    }
    return matrix;
}

// test to derived the matrix free operator class
class TestOperator : public MatrixFreeOperator
{
    public : 
    TestOperator(int n) {_size = n;}
    Eigen::VectorXd col(int index) const;
};

//  get a col of the operator
Eigen::VectorXd TestOperator::col(int index) const
{
    Eigen::VectorXd col_out = Eigen::VectorXd::Zero(_size,1);    
    for (int j=0; j < _size; j++)
    {
        if (j==index) {
            col_out(j) = static_cast<double> (j+1); 
        }
        else{
            col_out(j) = 0.01 / std::pow( static_cast<double>(j-index),2) ;
        }
    }
    return col_out;
}

//BOOST_AUTO_TEST_SUITE(davidson_test)

BOOST_AUTO_TEST_CASE(davidson_full_matrix) {

    int size = 1000;
    int neigen = 10;
    double eps = 0.01;
    Eigen::MatrixXd A = init_matrix(size,eps,false);

    DavidsonSolver DS;
    DS.solve(A,neigen);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(A);

    auto lambda = DS.eigenvalues();
    auto lambda_ref = es.eigenvalues().head(neigen);
    bool check_eigenvalues = lambda.isApprox(lambda_ref,1E-6);
    
    BOOST_CHECK_EQUAL(check_eigenvalues,1);

}


BOOST_AUTO_TEST_CASE(olsen_full_matrix) {

    int size = 1000;
    int neigen = 10;
    double eps = 0.01;
    Eigen::MatrixXd A = init_matrix(size,eps,false);

    DavidsonSolver DS;
    DS.set_correction("OLSEN");
    DS.solve(A,neigen);

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(A);

    auto lambda = DS.eigenvalues();
    auto lambda_ref = es.eigenvalues().head(neigen);
    bool check_eigenvalues = lambda.isApprox(lambda_ref,1E-6);
    
    BOOST_CHECK_EQUAL(check_eigenvalues,1);

}

BOOST_AUTO_TEST_CASE(jacobi_full_matrix) {

    int size = 50;
    int neigen = 2;
    double eps = 0.01;
    Eigen::MatrixXd A = init_matrix(size,eps,false);

    DavidsonSolver DS;
    DS.set_correction("JACOBI");
    DS.set_jacobi_linsolve("CG");
    DS.solve(A,neigen);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(A);

    auto lambda = DS.eigenvalues();
    auto lambda_ref = es.eigenvalues().head(neigen);
    bool check_eigenvalues = lambda.isApprox(lambda_ref,1E-6);
    
    BOOST_CHECK_EQUAL(check_eigenvalues,1);

}


BOOST_AUTO_TEST_CASE(davidson_matrix_free) {

    int size = 1000;
    int neigen = 10;

    TestOperator Aop(size);
    DavidsonSolver DS;
    DS.solve(Aop,neigen);

    Eigen::MatrixXd A = Aop.get_full_mat();
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(A);

    auto lambda = DS.eigenvalues();
    auto lambda_ref = es.eigenvalues().head(neigen);
    bool check_eigenvalues = lambda.isApprox(lambda_ref,1E-6);
    
    BOOST_CHECK_EQUAL(check_eigenvalues,1);

}


BOOST_AUTO_TEST_CASE(olsen_matrix_free) {

    int size = 1000;
    int neigen = 10;
    

    TestOperator Aop(size);
    DavidsonSolver DS;
    DS.set_correction("OLSEN");
    DS.solve(Aop,neigen);

    Eigen::MatrixXd A = Aop.get_full_mat();
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(A);

    auto lambda = DS.eigenvalues();
    auto lambda_ref = es.eigenvalues().head(neigen);
    bool check_eigenvalues = lambda.isApprox(lambda_ref,1E-6);
    
    BOOST_CHECK_EQUAL(check_eigenvalues,1);

}

BOOST_AUTO_TEST_CASE(jacobi_matrix_free) {

    int size = 50;
    int neigen = 2;
    

    TestOperator Aop(size);
    DavidsonSolver DS;
    DS.set_correction("JACOBI");
    DS.set_jacobi_linsolve("CG");
    DS.solve(Aop,neigen);

    Eigen::MatrixXd A = Aop.get_full_mat();
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(A);

    auto lambda = DS.eigenvalues();
    auto lambda_ref = es.eigenvalues().head(neigen);
    bool check_eigenvalues = lambda.isApprox(lambda_ref,1E-6);
    
    BOOST_CHECK_EQUAL(check_eigenvalues,1);

}

//BOOST_AUTO_TEST_SUITE_END()