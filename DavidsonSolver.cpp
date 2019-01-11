#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/QR>
#include <Eigen/Eigenvalues>
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/IterativeSolvers>
#include <chrono>
#include "DavidsonSolver.hpp"
#include "DavidsonOperator.hpp"


typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> Mat;
typedef Eigen::Matrix<double,Eigen::Dynamic,1> Vect;
typedef Eigen::Matrix<double,1,Eigen::Dynamic> Vect_row;

DavidsonSolver::DavidsonSolver() : iter_max(1000), tol(1E-6), max_search_space(100) { }
DavidsonSolver::DavidsonSolver(int iter_max) : iter_max(iter_max) , tol(1E-6), max_search_space(100) { }
DavidsonSolver::DavidsonSolver(int iter_max, double tol) : iter_max(iter_max), tol(tol), max_search_space(100) { }
DavidsonSolver::DavidsonSolver(int iter_max, double tol, int max_search_space) : iter_max(iter_max), tol(tol), max_search_space(max_search_space) { }

void DavidsonSolver::set_iter_max(int N) { this->iter_max = N; }
void DavidsonSolver::set_tolerance(double eps) { this->tol = eps; }
void DavidsonSolver::set_max_search_space(int N) { this->max_search_space = N;}
void DavidsonSolver::set_jacobi_correction() { this->jacobi_correction = true; }
void DavidsonSolver::set_jacobi_linsolve(int method) {this->jacobi_linsolve = method;}

Vect DavidsonSolver::eigenvalues() {return this->_eigenvalues;}
Mat DavidsonSolver::eigenvectors() {return this->_eigenvectors;}


Eigen::ArrayXd DavidsonSolver::_sort_index(Vect V)
{
    Eigen::ArrayXd idx = Eigen::ArrayXd::LinSpaced(V.rows(),0,V.rows()-1);
    std::sort(idx.data(),idx.data()+idx.size(),
              [&](int i1, int i2){return V[i1]<V[i2];});
    return idx; 
}

Mat DavidsonSolver::_get_initial_eigenvectors(Vect d, int size_initial_guess)
{

    Mat guess = Mat::Zero(d.size(),size_initial_guess);
    Eigen::ArrayXd idx = DavidsonSolver::_sort_index(d);

    for (int j=0; j<size_initial_guess;j++)
        guess(idx(j),j) = 1.0;

    return guess;
}

Mat DavidsonSolver::_solve_linear_system(Mat A, Vect r)
{
    Mat w;

    switch(this->jacobi_linsolve)
    {        
        //use cg approximate solver     
        case 0: 
        {
            Eigen::ConjugateGradient<Mat, Eigen::Lower|Eigen::Upper> cg;
            cg.compute(A);
            w = cg.solve(r);
        }   

        //use GMRES approximate solver
        case 1:
        {
            Eigen::GMRES<Mat, Eigen::IdentityPreconditioner> gmres;
            gmres.compute(A);
            w = gmres.solve(r);
        }

        //use llt exact solver
        case 2: w = A.llt().solve(r);
    }

    return w;
}



template <class OpMat>
Mat DavidsonSolver::_jacobi_orthogonal_correction(OpMat A, Vect u, double lambda)
{
    Mat w;

    // form the projector 
    Mat P = -u*u.transpose();
    P.diagonal().array() += 1.0;

    // compute the residue
    auto r = A*u - lambda*u;

    // project the matrix
    // P * (A - lambda*I) * P^T
    Mat projA = A*P.transpose();
    projA -= lambda*P.transpose();
    projA = P * projA;

    return DavidsonSolver::_solve_linear_system(projA,r);
}

template Mat DavidsonSolver::_jacobi_orthogonal_correction<Mat>(Mat A, Vect u, double lambda);
template Mat DavidsonSolver::_jacobi_orthogonal_correction<DavidsonOperator>(DavidsonOperator A, Vect u, double lambda);

template<class OpMat>
void DavidsonSolver::solve(OpMat A, int neigen, int size_initial_guess)
{

    if (this->_debug_)
    {
        std::cout << std::endl;
        std::cout << "===========================" << std::endl; 
        if(this->jacobi_correction)
        {
            std::cout << "= Jacobi-Davidson      " <<  std::endl; 
            std::cout << "    linsolve : " << this->jacobi_linsolve << std::endl;
        }
        else
            std::cout << "= Davidson (DPR)" <<  std::endl; 
        std::cout << "===========================" << std::endl;
        std::cout << std::endl;
    }

    double norm;
    int size = A.rows();

    // if argument not provided we default to 0
    // and set to twice the number of eigenvalues
    if (size_initial_guess == 0)
        size_initial_guess = 2*neigen;

    int search_space = size_initial_guess;

    // initialize the guess eigenvector
    Vect Adiag = A.diagonal();    
    Mat V = DavidsonSolver::_get_initial_eigenvectors(Adiag,size_initial_guess);

    // sort the eigenvalues
    std::sort(Adiag.data(),Adiag.data()+Adiag.size());

    // thin matrix for QR
    Mat thinQ;

    // eigenvalues hodlers
    Vect lambda;
    Vect lambda_old = Vect::Ones(neigen,1);

    // temp varialbes 
    Mat U, w, q;

    // chrono !
    std::chrono::time_point<std::chrono::system_clock> start, end, instart, instop;
    std::chrono::duration<double> elapsed_time;

    if (_debug_)
        std::cout << "iter\tSearch Space\tNorm" << std::endl;
    
    for (int iiter = 0; iiter < iter_max; iiter ++ )
    {
        
        // orthogonalise the vectors
        // use the HouseholderQR algorithm of Eigen
        if (iiter>0)
        {
            thinQ = Mat::Identity(V.rows(),V.cols());
            Eigen::HouseholderQR<Mat> qr(V);
            V = qr.householderQ() * thinQ;
        }

        // project the matrix on the trial subspace
        Mat T = A * V;
        T = V.transpose()*T;

        // diagonalize in the subspace
        // we could replace that with LAPACK ... 
        Eigen::SelfAdjointEigenSolver<Mat> es(T);
        lambda = es.eigenvalues();
        U = es.eigenvectors();
        
        // Ritz eigenvectors
        q = V.block(0,0,V.rows(),search_space)*U;

        // compute correction vectors
        // and append to V
        for (int j=0; j<size_initial_guess; j++)
        {   

            // jacobi-davidson correction
            if (this->jacobi_correction)
                w = DavidsonSolver::_jacobi_orthogonal_correction<OpMat>(A,q.col(j),lambda(j));
            
            // Davidson DPR
            else  
                w = ( A*q.col(j) - lambda(j)*q.col(j) ) / ( lambda(j) - Adiag(j) );  

            // append the correction vector to the search space
            V.conservativeResize(Eigen::NoChange,V.cols()+1);
            V.col(V.cols()-1) = w;
        }

        // check for convergence
        norm = (lambda.head(neigen) - lambda_old).norm();

        if(_debug_)
            printf("%4d\t%12d\t%4.2e/%.0e\n", iiter,search_space,norm,tol);
        
        // break if converged, update otherwise
        if (norm < tol)
            break;
        else
        {
            lambda_old = lambda.head(neigen);
            search_space += size_initial_guess;
        }

        // restart
        if (search_space > max_search_space)
        {
            V = q.block(0,0,V.rows(),size_initial_guess);
            search_space = size_initial_guess;
        }
    }

    // store the eigenvalues/eigenvectors
    this->_eigenvalues = lambda.head(neigen);
    this->_eigenvectors = U.block(0,0,U.rows(),neigen);
   
}

template void DavidsonSolver::solve<Mat>(Mat A, int neigen, int size_initial_guess=0);
template void DavidsonSolver::solve<DavidsonOperator>(DavidsonOperator A, int neigen, int size_initial_guess=0);

