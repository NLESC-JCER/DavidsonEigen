#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/QR>
#include <Eigen/Eigenvalues>
#include <chrono>
#include "DavidsonSolver.hpp"


typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> Mat;
typedef Eigen::Matrix<double,Eigen::Dynamic,1> Vect;
typedef Eigen::Matrix<double,1,Eigen::Dynamic> Vect_row;

DavidsonSolver::DavidsonSolver() : iter_max(1000), tol(1E-6), max_search_space(100) { }
DavidsonSolver::DavidsonSolver(int iter_max) : iter_max(iter_max) , tol(1E-6), max_search_space(100) { }
DavidsonSolver::DavidsonSolver(int iter_max, double tol) : iter_max(iter_max), tol(tol), max_search_space(100) { }
DavidsonSolver::DavidsonSolver(int iter_max, double tol, int max_search_space) : iter_max(iter_max), tol(tol), max_search_space(max_search_space) { }

void DavidsonSolver::set_iter_max(int N) { iter_max = N; }
void DavidsonSolver::set_tolerance(double eps) { tol = eps; }
void DavidsonSolver::set_max_search_space(int N) { max_search_space = N;}

Vect DavidsonSolver::eigenvalues() {return _eigenvalues;}
Mat DavidsonSolver::eigenvectors() {return _eigenvectors;}


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

void DavidsonSolver::solve(Mat A, int neigen, int size_initial_guess)
{

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
    Mat I = Mat::Identity(size,size);

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

    for (int iiter = 0; iiter < iter_max; iiter ++ )
    {
        
    	if (_debug_)
    	{
        	std::cout << "iteration " << iiter << " / " << iter_max << std::endl;
        	std::cout << "Search Space Size " << V.rows() << "x" << search_space << std::endl;
        }

        // orthogonalise the vectors
        // use the HouseholderQR algorithm of Eigen
        if (iiter>0)
        {
            thinQ = Mat::Identity(V.rows(),V.cols());
            Eigen::HouseholderQR<Mat> qr(V);
            V = qr.householderQ() * thinQ;
        }

        // project the matrix on the trail subspace
        Mat T = V.transpose() * A * V;

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
            w = ( (A-lambda(j)*I) * q.col(j) ) / ( lambda(j) - Adiag(j) );
            V.conservativeResize(Eigen::NoChange,V.cols()+1);
            V.col(V.cols()-1) = w;

        }

        // check for convergence
        norm = (lambda.head(neigen) - lambda_old).norm();

        if(_debug_)
        	std::cout << "Norm " << norm << " / " << tol << std::endl << std::endl << std::endl;
        
        // break if converged update otherwise
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
    _eigenvalues = lambda.head(neigen);
    _eigenvectors = U.block(0,0,U.rows(),neigen);
   
}


void DavidsonSolver::solve(DavidsonOperator A, int neigen, int size_initial_guess)
{

    double norm;
    int size = A.OpSize();

    // if argument not provided we default to 0
    // and set to twice the number of eigenvalues
    if (size_initial_guess == 0)
    	size_initial_guess = 2*neigen;

    int search_space = size_initial_guess;

	// initialize the guess eigenvector
	Vect Adiag = A.diagonal();    
    Mat V = DavidsonSolver::_get_initial_eigenvectors(Adiag,size_initial_guess);
    Mat I = Mat::Identity(size,size);

    // sort the eigenvalues 
    std::cout << "sort eigenvalues" << std::endl;
	std::sort(Adiag.data(),Adiag.data()+Adiag.size());
    
    // thin matrix for QR
    Mat thinQ;

    // eigenvalues hodlers
    Vect lambda;
    Vect lambda_old = Vect::Ones(neigen,1);

    // temp varialbes 
    Mat U, w, q;

    for (int iiter = 0; iiter < iter_max; iiter ++ )
    {
        
    	if (_debug_)
    	{
        	std::cout << "iteration " << iiter << " / " << iter_max << std::endl;
        	std::cout << "Search Space Size " << V.rows() << "x" << search_space << std::endl;
        }

        // orthogonalise the vectors
        // use the HouseholderQR algorithm of Eigen
        if (iiter>0)
        {
            thinQ = Mat::Identity(V.rows(),V.cols());
            Eigen::HouseholderQR<Mat> qr(V);
            V = qr.householderQ() * thinQ;
        }

        // project the matrix on the trail subspace
        Mat T = V.transpose() * A.apply_to_mat(V);

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
        	w = ( A.apply_to_vect(q.col(j)) - lambda(j)*q.col(j) ) / ( lambda(j) - Adiag(j) );  
            V.conservativeResize(Eigen::NoChange,V.cols()+1);
            V.col(V.cols()-1) = w;

        }

        // check for convergence
        norm = (lambda.head(neigen) - lambda_old).norm();

        if(_debug_)
        	std::cout << "Norm " << norm << " / " << tol << std::endl << std::endl << std::endl;
        
        // break if converged update otherwise
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
    _eigenvalues = lambda.head(neigen);
    _eigenvectors = U.block(0,0,U.rows(),neigen);
   
}