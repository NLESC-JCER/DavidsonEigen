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
#include "MatrixFreeOperator.hpp"

DavidsonSolver::DavidsonSolver(){}

Eigen::ArrayXd DavidsonSolver::_sort_index(Eigen::VectorXd& V)
{
    Eigen::ArrayXd idx = Eigen::ArrayXd::LinSpaced(V.rows(),0,V.rows()-1);
    std::sort(idx.data(),idx.data()+idx.size(),
              [&](int i1, int i2){return V[i1]<V[i2];});
    return idx; 
}

Eigen::MatrixXd DavidsonSolver::_get_initial_eigenvectors(Eigen::VectorXd &d, int size_initial_guess)
{

    Eigen::MatrixXd guess = Eigen::MatrixXd::Zero(d.size(),size_initial_guess);
    Eigen::ArrayXd idx = DavidsonSolver::_sort_index(d);

    for (int j=0; j<size_initial_guess;j++) {
        guess(idx(j),j) = 1.0;
    }

    return guess;
}

Eigen::MatrixXd DavidsonSolver::_solve_linear_system(Eigen::MatrixXd &A, Eigen::VectorXd &r)
{
    Eigen::MatrixXd w;
    if(this->jacobi_linsolve == 0) {
        Eigen::ConjugateGradient<Eigen::MatrixXd, Eigen::Lower|Eigen::Upper> cg;
        cg.compute(A);
        w = cg.solve(r);
    }   

    else if (this->jacobi_linsolve == 1) {
        Eigen::GMRES<Eigen::MatrixXd, Eigen::IdentityPreconditioner> gmres;
        gmres.compute(A);
        w = gmres.solve(r);
    }

    else if (this->jacobi_linsolve == 2) {
        w = A.llt().solve(r);
    }

    return w;
}

template <class MatrixReplacement>
Eigen::MatrixXd DavidsonSolver::_jacobi_orthogonal_correction(MatrixReplacement &A, Eigen::VectorXd &r, Eigen::VectorXd &u, double lambda)
{
    // form the projector  P = I -u * u.T
    Eigen::MatrixXd P = -u*u.transpose();
    P.diagonal().array() += 1.0;

    // project the matrix P * (A - lambda*I) * P^T
    Eigen::MatrixXd projA = A*P.transpose();
    projA -= lambda*P.transpose();
    projA = P * projA;

    return DavidsonSolver::_solve_linear_system(projA,r);
}

Eigen::VectorXd DavidsonSolver::_dpr_correction(Eigen::VectorXd &w, Eigen::VectorXd &A0, double lambda, int size)
{
    Eigen::VectorXd out = Eigen::VectorXd::Zero(size);
    
    for (int i=0; i < size; i++){
        out(i) = w(i) / (lambda - A0(i));
    }
    return out;
}


template<class MatrixReplacement>
void DavidsonSolver::solve(MatrixReplacement &A, int neigen, int size_initial_guess)
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

    double res_norm;
    double conv;
    int size = A.rows();

    // initial guess size
    if (size_initial_guess == 0)  size_initial_guess = 2*neigen;
    int search_space = size_initial_guess;

    // initialize the guess eigenvector
    Eigen::VectorXd Adiag = A.diagonal();    
    Eigen::MatrixXd V = DavidsonSolver::_get_initial_eigenvectors(Adiag,size_initial_guess);
    std::sort(Adiag.data(),Adiag.data()+Adiag.size());

    
    Eigen::MatrixXd thinQ; // thin matrix for QR
    Eigen::VectorXd lambda; // eigenvalues hodlers
    
    // temp varialbes 
    Eigen::MatrixXd T, U, q;
    Eigen::VectorXd w, tmp;
    Eigen::VectorXd old_val = Eigen::VectorXd::Zero(neigen);

    // chrono !
    std::chrono::time_point<std::chrono::system_clock> start, end, instart, instop;
    std::chrono::duration<double> elapsed_time;

    
    std::cout << "iter\tSearch Space\tNorm" << std::endl;
    for (int iiter = 0; iiter < iter_max; iiter ++ )
    {
        
        // orthogonalise the vectors
        // use the HouseholderQR algorithm of Eigen
        if (iiter>0) {
            thinQ = Eigen::MatrixXd::Identity(V.rows(),V.cols());
            Eigen::HouseholderQR<Eigen::MatrixXd> qr(V);
            V = qr.householderQ() * thinQ;
        }

        // project the matrix on the trial subspace and diagonalize it
        T = A * V;
        T = V.transpose()*T;
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(T);
        lambda = es.eigenvalues();
        U = es.eigenvectors();
        
        // Ritz eigenvectors
        q = V*U.block(0,0,U.rows(),neigen);
        res_norm = 0.0;

        // residue vectors
        for (int j=0; j<neigen; j++) {   

            // residue vector
            w = A*q.col(j) - lambda(j)*q.col(j);
            res_norm += w.norm() / neigen;
            
            // jacobi-davidson correction
            if (this->jacobi_correction)
            {
                tmp = q.col(j);
                w = DavidsonSolver::_jacobi_orthogonal_correction<MatrixReplacement>(A,w,tmp,lambda(j));
            }
            
            // Davidson DPR
            else  {
                w = DavidsonSolver::_dpr_correction(w,Adiag,lambda(j),size);
            }

            // append the correction vector to the search space
            V.conservativeResize(Eigen::NoChange,V.cols()+1);
            V.col(V.cols()-1) = w;

        }

        // normalize the norm
        conv = (lambda.head(neigen)-old_val).norm();
        printf("%4d\t%12d\t%4.2e/%.0e %4.2e\n", iiter,search_space,res_norm,tol,conv);
        
        // break if converged, update otherwise
        if (res_norm < tol) {
            break;
        }
        else
        {
            search_space = V.cols();
            old_val = lambda.head(neigen);
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
    this->_eigenvectors = q.block(0,0,q.rows(),neigen);
   
}

template void DavidsonSolver::solve<Eigen::MatrixXd>(Eigen::MatrixXd &A, int neigen, int size_initial_guess=0);
template void DavidsonSolver::solve<DavidsonOperator>(DavidsonOperator &A, int neigen, int size_initial_guess=0);

