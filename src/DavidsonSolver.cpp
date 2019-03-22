#include <iostream>
#include <stdexcept>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/QR>
#include <Eigen/Eigenvalues>
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/IterativeSolvers>
#include <chrono>

#include "DavidsonSolver.hpp"

DavidsonSolver::DavidsonSolver(){}


void DavidsonSolver::set_correction(std::string method) {
    if (method == "DPR") this->correction = CORR::DPR;
    else if (method == "JACOBI") this->correction = CORR::JACOBI;
    else if (method == "OLSEN") this->correction = CORR::OLSEN;
    else throw std::runtime_error("Not a valid correction method");
}

void DavidsonSolver::set_jacobi_linsolve(std::string method) {
    if (method == "CG") this->jacobi_linsolve = LSOLVE::CG;
    else if (method == "GMRES") this->jacobi_linsolve = LSOLVE::GMRES;
    else if (method == "LLT") this->jacobi_linsolve = LSOLVE::LLT;   
    else throw std::runtime_error("Not a valid linsolve method");
}

Eigen::ArrayXd DavidsonSolver::_sort_index(Eigen::VectorXd& V) const
{
    Eigen::ArrayXd idx = Eigen::ArrayXd::LinSpaced(V.rows(),0,V.rows()-1);
    std::sort(idx.data(),idx.data()+idx.size(),
              [&](int i1, int i2){return V[i1]<V[i2];});
    return idx; 
}

Eigen::MatrixXd DavidsonSolver::_get_initial_eigenvectors(Eigen::VectorXd &d, int size_initial_guess) const
{

    Eigen::MatrixXd guess;
    if (this->guess_vectors =="identity")
    {
            guess = Eigen::MatrixXd::Identity(d.size(),size_initial_guess);
    }

    else if (this->guess_vectors == "random")
    {
        guess = Eigen::MatrixXd::Random(d.size(),size_initial_guess);
        guess = DavidsonSolver::_QR(guess);
    }
    else if (this->guess_vectors=="target")
    {
        guess = Eigen::MatrixXd::Zero(d.size(),size_initial_guess);
        Eigen::ArrayXd idx = DavidsonSolver::_sort_index(d);

        for (int j=0; j<size_initial_guess;j++) {
            guess(idx(j),j) = 1.0;
        }
    }
    return guess;
}

Eigen::MatrixXd DavidsonSolver::_solve_linear_system(Eigen::MatrixXd &A, Eigen::VectorXd &r) const
{
    Eigen::MatrixXd w;
    std::chrono::time_point<std::chrono::system_clock> start, end;
    std::chrono::duration<double> elapsed_time;

    start = std::chrono::system_clock::now();
    switch (this->jacobi_linsolve) {

        case LSOLVE::CG :  {
                Eigen::ConjugateGradient<Eigen::MatrixXd, Eigen::Lower|Eigen::Upper> cg;
                cg.setTolerance(this->linsolve_tol);
                cg.compute(A);
                w = cg.solve(r); 
            }
            break;
        case LSOLVE::GMRES : {
                Eigen::GMRES<Eigen::MatrixXd, Eigen::IdentityPreconditioner> gmres;
                gmres.setTolerance(this->linsolve_tol);
                gmres.compute(A);
                w = gmres.solve(r);
            }
            break;
        case LSOLVE::LLT : 
            w = A.llt().solve(r);
            break;
    }
    end = std::chrono::system_clock::now();
    elapsed_time  = end-start;
    std::cout << "_ solve linear system " << this->jacobi_linsolve << " in " << elapsed_time.count() << " secs" <<  std::endl;
    return w;
}

Eigen::VectorXd DavidsonSolver::_olsen_correction(Eigen::VectorXd &r, Eigen::VectorXd &x, Eigen::VectorXd &D, double lambda) const
{
    /* Compute the olsen correction :

    \delta = (D-\lambda)^{-1} (-r + \epsilon x)

    */

    int size = r.rows();
    Eigen::VectorXd delta = Eigen::VectorXd::Zero(size);

    delta = DavidsonSolver::_dpr_correction(r,D,lambda);

    double _num = - x.transpose() * delta;
    double _denom = - x.transpose() * DavidsonSolver::_dpr_correction(x,D,lambda);
    double eps = _num / _denom;
    delta += eps * x;

    return delta;
}

Eigen::VectorXd DavidsonSolver::_dpr_correction(Eigen::VectorXd &w, Eigen::VectorXd &A0, double lambda) const
{
    int size = w.rows();
    Eigen::VectorXd out = Eigen::VectorXd::Zero(size);
    for (int i=0; i < size; i++) {
        out(i) = w(i) / (lambda - A0(i));
    }

    return out;
}

Eigen::MatrixXd DavidsonSolver::_QR(Eigen::MatrixXd &A) const
{
    
    int nrows = A.rows();
    int ncols = A.cols();
    ncols = std::min(nrows,ncols);
    
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(A);
    return qr.householderQ() * Eigen::MatrixXd::Identity(nrows,ncols);
}


Eigen::MatrixXd DavidsonSolver::_gramschmidt( Eigen::MatrixXd &A, int nstart ) const
{
    Eigen::MatrixXd Q = A;

    for(unsigned int j = nstart; j < A.cols(); ++j) {
        // Replace inner loop over each previous vector in Q with fast matrix-vector multiplication
        Q.col(j) -= Q.leftCols(j) * (Q.leftCols(j).transpose() * A.col(j));
        // Normalize vector if possible (othw. means colums of A almsost lin. dep.
        if( Q.col(j).norm() <= 10e-14 * A.col(j).norm() ) {
            std::cerr << "Gram-Schmidt failed because A has lin. dep columns. Bye." << std::endl;
            break;
        } else {
            Q.col(j).normalize();
        }
    }
    return Q;
}


