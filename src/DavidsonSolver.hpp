#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Core>

#ifndef _DAVIDSON_SOLVER_
#define _DAVIDSON_SOLVER_

class DavidsonSolver
{

	public:

		DavidsonSolver();

		void set_iter_max(int N) { this->iter_max = N; }
		void set_tolerance(double eps) { this->tol = eps; }
		void set_max_search_space(int N) { this->max_search_space = N;}
		void set_initial_guess_size(int N) {this->size_initial_guess=N;}
		void set_jacobi_correction() { this->jacobi_correction = true; }
		void set_jacobi_linsolve(int method) {this->jacobi_linsolve = method;}

		Eigen::VectorXd eigenvalues() {return this->_eigenvalues;}
		Eigen::MatrixXd eigenvectors() {return this->_eigenvectors;}

		template <typename MatrixReplacement>
		void solve(MatrixReplacement &A, int neigen, int size_initial_guess = 0);

	private :

		int iter_max = 1000;
		double tol = 1E-6;
		int max_search_space = 100;
		int size_initial_guess = 0;
		bool jacobi_correction=false;
		std::string jacobi_linsolve = "CG";

		Eigen::VectorXd _eigenvalues;
		Eigen::MatrixXd _eigenvectors; 

		Eigen::ArrayXd _sort_index(Eigen::VectorXd &V);
		Eigen::MatrixXd _get_initial_eigenvectors(Eigen::VectorXd &D, int size);
		Eigen::MatrixXd _solve_linear_system(Eigen::MatrixXd &A, Eigen::VectorXd &b); 

		template <typename MatrixReplacement>
		Eigen::MatrixXd _jacobi_orthogonal_correction(MatrixReplacement &A, Eigen::VectorXd &r, Eigen::VectorXd &u, double lambda);

};


#endif