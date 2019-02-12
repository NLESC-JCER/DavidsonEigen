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

		void set_correction(std::string method); 
		void set_jacobi_linsolve(std::string method);

		Eigen::VectorXd eigenvalues() const {return this->_eigenvalues;}
		Eigen::MatrixXd eigenvectors() const {return this->_eigenvectors;}

		template <typename MatrixReplacement>
		void solve(MatrixReplacement &A, int neigen, int size_initial_guess = 0)
		{

		    std::cout << std::endl;
		    std::cout << "===========================" << std::endl; 
		    if(this->correction == CORR::JACOBI)  std::cout << "= Jacobi-Davidson  : " << this->jacobi_linsolve <<  std::endl; 
		    
		    else if (this->correction == CORR::OLSEN)  std::cout << "= Olsen-Davidson  : " <<  std::endl;    
		    
		    else  std::cout << "= Davidson (DPR)" <<  std::endl; 

		    std::cout << "===========================" << std::endl;
		    std::cout << std::endl;

		    double res_norm;
		    double conv;
		    int size = A.rows();
		    bool has_converged = false;

		    // initial guess size
		    if (size_initial_guess == 0)  size_initial_guess = 2*neigen;
		    int search_space = size_initial_guess;

		    // initialize the guess eigenvector
		    Eigen::VectorXd Adiag = A.diagonal();    
		    Eigen::MatrixXd V = DavidsonSolver::_get_initial_eigenvectors(Adiag,size_initial_guess);
		    //Eigen::MatrixXd V = Eigen::MatrixXd::Identity(size,size_initial_guess);

		    // sort the diagonal elements -> apparently detrimental ...
		    // std::sort(Adiag.data(),Adiag.data()+Adiag.size());
		    
		    Eigen::VectorXd lambda; // eigenvalues hodlers
		    Eigen::VectorXd old_val = Eigen::VectorXd::Zero(neigen);
		    
		    // temp varialbes 
		    Eigen::MatrixXd T, U, q;
		    Eigen::VectorXd w, tmp;
		    

		    // project the matrix on the trial subspace
		    T = A * V;
		    T = V.transpose()*T;

		    printf("iter\tSearch Space\tNorm/%.0e\n",tol);
		    std::cout << "-----------------------------------" << std::endl;
		    for (int iiter = 0; iiter < iter_max; iiter ++ )
		    {
		        
		        // std::cout << "\nT:\n" << T << std::endl;
		        // diagonalize the small subspace
		        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(T);
		        lambda = es.eigenvalues();
		        U = es.eigenvectors();


		        // Ritz eigenvectors
		        q = V*U.block(0,0,U.rows(),neigen);
		        res_norm = 0.0;

		        // residue and correction vectors
		        for (int j=0; j<neigen; j++) {   

		            // residue vector
		            w = A*q.col(j) - lambda(j)*q.col(j);
		            res_norm += w.norm() / neigen;
		            
		            // jacobi-davidson correction
		            if (this->correction == CORR::JACOBI) {
		                tmp = q.col(j);
		                w = DavidsonSolver::_jacobi_correction<MatrixReplacement>(A,w,tmp,lambda(j));
		            }

		            else if (this->correction == CORR::OLSEN) {
		                w = DavidsonSolver::_olsen_correction(w,Adiag,lambda(j));
		            }
		            
		            // Davidson DPR
		            else  {
		                w = DavidsonSolver::_dpr_correction(w,Adiag,lambda(j));
		            }

		            // append the correction vector to the search space
		            V.conservativeResize(Eigen::NoChange,V.cols()+1);
		            V.col(V.cols()-1) = w.normalized();
		            
		        }

		        // eigenvalue norm
		        conv = (lambda.head(neigen)-old_val).norm();
		        printf("%4d\t%12d\t%4.2e\n", iiter,search_space,res_norm);

		        // update 
		        search_space = V.cols();
		        old_val = lambda.head(neigen);
		        
		        // break if converged, update otherwise
		        if (res_norm < tol) {
		            has_converged = true;
		            break;
		        }

		        // check if we need to restart
		        if (search_space > max_search_space or search_space > size )
		        {
		            V = q.block(0,0,V.rows(),neigen);
		            for (int j=0; j<neigen; j++) {
		                V.col(j) = V.col(j).normalized();
		            }
		            search_space = neigen;

		            // recompute the projected matrix
		            T = A * V;
		            T = V.transpose()*T;
		        }

		        // continue otherwise
		        else
		        {
		            // orthogonalize the V vectors
		            V = DavidsonSolver::_QR(V);
		            
		            // update the T matrix : avoid recomputing V.T A V 
		            // just recompute the element relative to the new eigenvectors
		            DavidsonSolver::_update_projected_matrix<MatrixReplacement>(T,A,V);

		        }
		        
		    }

		    // store the eigenvalues/eigenvectors
		    this->_eigenvalues = lambda.head(neigen);
		    this->_eigenvectors = q.block(0,0,q.rows(),neigen);

		    // normalize the eigenvectors
		    for (int i=0; i<neigen; i++){
		        this->_eigenvectors.col(i).normalize();
		    }

		    std::cout << "-----------------------------------" << std::endl;
		    if (!has_converged) {
		        std::cout << "- Warning : Davidson didn't converge ! " <<  std::endl; 
		        this->_eigenvalues = Eigen::VectorXd::Zero(neigen);
		        this->_eigenvectors = Eigen::MatrixXd::Zero(size,neigen);
		    }
		    else   {
		        std::cout << "- Davidson converged " <<  std::endl; 
		        printf("- final residue norm %4.2e\n",res_norm);
		        printf("- final eigenvalue norm %4.2e\n",conv);
		    }
		    std::cout << "-----------------------------------" << std::endl;
		    
		}

	private :

		int iter_max = 1000;
		double tol = 1E-6;
		int max_search_space = 100;
		int size_initial_guess = 0;

		enum CORR {DPR,JACOBI,OLSEN};
		enum LSOLVE {CG,GMRES,LLT};

		CORR correction = CORR::DPR;
		LSOLVE jacobi_linsolve = LSOLVE::CG;

		Eigen::VectorXd _eigenvalues;
		Eigen::MatrixXd _eigenvectors; 

		Eigen::ArrayXd _sort_index(Eigen::VectorXd &V) const;
		Eigen::MatrixXd _get_initial_eigenvectors(Eigen::VectorXd &D, int size) const;
		Eigen::MatrixXd _solve_linear_system(Eigen::MatrixXd &A, Eigen::VectorXd &b) const; 
		Eigen::MatrixXd _QR(Eigen::MatrixXd &A) const;

		template <typename MatrixReplacement>
		Eigen::MatrixXd _jacobi_correction(MatrixReplacement &A, Eigen::VectorXd &r, Eigen::VectorXd &u, double lambda) const
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

		Eigen::VectorXd _dpr_correction(Eigen::VectorXd &w, Eigen::VectorXd &A0, double lambda) const;
		Eigen::VectorXd _olsen_correction(Eigen::VectorXd &w, Eigen::VectorXd &A0, double lambda) const;

		template<class MatrixReplacement>
		void _update_projected_matrix(Eigen::MatrixXd &T, MatrixReplacement &A, Eigen::MatrixXd &V) const
		{
		    int nvec_old = T.cols();
		    int nvec = V.cols();
		    int nnew_vec = nvec-nvec_old;

		    Eigen::MatrixXd _tmp = A * V.block(0,nvec_old,nvec,nnew_vec);
		    T.conservativeResize(nvec,nvec);
		    T.block(0,nvec_old,nvec,nnew_vec) = V.transpose() * _tmp;
		    T.block(nvec_old,0,nnew_vec,nvec_old) = T.block(0,nvec_old,nvec_old,nnew_vec).transpose();

		    return;
		}
};


#endif