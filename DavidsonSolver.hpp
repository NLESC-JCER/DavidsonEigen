#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/QR>
#include <Eigen/Eigenvalues>
#include <chrono>
#include "DavidsonOperator.hpp"

typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> Mat;
typedef Eigen::Matrix<double,Eigen::Dynamic,1> Vect;
typedef Eigen::Matrix<double,1,Eigen::Dynamic> Vect_row;

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

		Vect eigenvalues();
		Mat eigenvectors();

		void solve(Mat A, int neigen, int size_initial_guess = 0);
		void solve(DavidsonOperator A, int neigen, int size_initial_guess = 0);

	private :

		int iter_max;
		double tol;
		int max_search_space;
		bool _debug_ = true;

		Vect _eigenvalues;
		Mat _eigenvectors; 

		Eigen::ArrayXd _sort_index(Vect V);
		Mat _get_initial_eigenvectors(Vect D, int size);
};
#endif