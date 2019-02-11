
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


//BOOST_AUTO_TEST_SUITE(davidson_test)

BOOST_AUTO_TEST_CASE(davidson_full_matrix_small) {

    int size = 10;
    int neigen = 2;
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

BOOST_AUTO_TEST_CASE(davidson_full_matrix_large) {

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

BOOST_AUTO_TEST_CASE(olsen_full_matrix_small) {

    int size = 10;
    int neigen = 2;
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

BOOST_AUTO_TEST_CASE(olsen_full_matrix_large) {

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

BOOST_AUTO_TEST_CASE(jacobi_full_matrix_small) {

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


//BOOST_AUTO_TEST_SUITE_END()