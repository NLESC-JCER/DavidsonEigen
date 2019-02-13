#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/QR>
#include <Eigen/Eigenvalues>
#include <chrono>
#include <cxxopts.hpp>

#include "DavidsonSolver.hpp"
#include "DavidsonOperator.hpp"
#include "MatrixFreeOperator.hpp"

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

int main (int argc, char *argv[]){

    // parse the input
    cxxopts::Options options(argv[0],  "Eigen Davidson Iterative Solver");
    options.positional_help("[optional args]").show_positional_help();
    options.add_options()
        ("size", "dimension of the matrix", cxxopts::value<std::string>()->default_value("100"))
        ("eps", "sparsity of the matrix", cxxopts::value<std::string>()->default_value("0.01"))
        ("neigen", "number of eigenvalues required", cxxopts::value<std::string>()->default_value("5"))
        ("corr", "correction method", cxxopts::value<std::string>()->default_value("DPR"))
        ("mf", "use matrix free", cxxopts::value<bool>())
        ("diag", "diagonal elements are ordered" , cxxopts::value<bool>())
        ("linsolve", "method to solve the linear system of JOCC (CG, GMRES, LLT)", cxxopts::value<std::string>()->default_value("CG"))
        ("help", "Print the help", cxxopts::value<bool>());
    auto result = options.parse(argc,argv);

    if (result.count("help"))
    {
        std::cout << options.help({""}) << std::endl;
        exit(0);
    }


    int size = std::stoi(result["size"].as<std::string>(),nullptr);
    int neigen = std::stoi(result["neigen"].as<std::string>(),nullptr);
    bool mf = result["mf"].as<bool>();
    bool odiag = result["diag"].as<bool>();
    std::string linsolve = result["linsolve"].as<std::string>();
    std::string correction = result["corr"].as<std::string>();
    bool help = result["help"].as<bool>();
    double eps = std::stod(result["eps"].as<std::string>(),nullptr);

    // chrono    
    std::chrono::time_point<std::chrono::system_clock> start, end;
    std::chrono::duration<double> elapsed_time;

    std::cout << "Matrix size : " << size << "x" << size << std::endl;
    std::cout << "Num Threads : " <<  Eigen::nbThreads() << std::endl;
    std::cout << "eps : " <<  eps << std::endl;

    // Create Operator
    DavidsonOperator Aop(size,odiag);
    Eigen::MatrixXd Afull = Aop.get_full_mat();

    // Davidosn Solver
    start = std::chrono::system_clock::now();
    DavidsonSolver DS;

    DS.set_correction(correction);
    if (correction == "JACOBI") DS.set_jacobi_linsolve(linsolve);

    if (mf) DS.solve(Aop,neigen);
    else  DS.solve(Afull,neigen);
    
    end = std::chrono::system_clock::now();
    elapsed_time = end-start;
    std::cout << std::endl << "Davidson               : " << elapsed_time.count() << " secs" <<  std::endl;
    
    // normal eigensolver
    start = std::chrono::system_clock::now();
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es2(Afull);
    end = std::chrono::system_clock::now();
    elapsed_time = end-start;
    std::cout << "Eigen                  : " << elapsed_time.count() << " secs" <<  std::endl;
    
    auto dseigop = DS.eigenvalues();
    auto eig2 = es2.eigenvalues().head(neigen);
    std::cout << std::endl <<  "      Davidson  \tEigen" << std::endl;
    for(int i=0; i< neigen; i++)
        printf("#% 4d %8.7f \t%8.7f\n",i,dseigop(i),eig2(i));

}