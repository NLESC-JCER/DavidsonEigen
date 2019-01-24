#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Core>

#ifndef _DAVIDSON_SOLVER_
#define _DAVIDSON_SOLVER_

class DavidsonSolver
{

	public:

		DavidsonSolver();
		DavidsonSolver(int itermax);
		DavidsonSolver(int itermax, double tol);
		DavidsonSolver(int itermax, double tol, int max_search_size);

		void set_iter_max(int set_iter_max);
		void set_tolerance(double tol);
		void set_max_search_space(int size);
		void set_jacobi_correction();
		void set_jacobi_linsolve(int method);

		Eigen::VectorXd eigenvalues();
		Eigen::MatrixXd eigenvectors();

		template <typename OpMat>
		void solve(OpMat &A, int neigen, int size_initial_guess = 0);

	private :

		int iter_max;
		double tol;
		int max_search_space;
		bool _debug_ = true;

		bool jacobi_correction=false;
		int jacobi_linsolve = 0;

		Eigen::VectorXd _eigenvalues;
		Eigen::MatrixXd _eigenvectors; 

		Eigen::ArrayXd _sort_index(Eigen::VectorXd &V);
		Eigen::MatrixXd _get_initial_eigenvectors(Eigen::VectorXd &D, int size);
		Eigen::MatrixXd _solve_linear_system(Eigen::MatrixXd &A, Eigen::VectorXd &b); 

		template <typename OpMat>
		Eigen::MatrixXd _jacobi_orthogonal_correction(OpMat A, Eigen::VectorXd r, Eigen::VectorXd u, double lambda);

};


#endif