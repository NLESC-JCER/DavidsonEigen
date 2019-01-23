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

Eigen::MatrixXd init_matrix(int N)
{
    Eigen::MatrixXd matrix = Eigen::MatrixXd::Zero(N,N);
    double eps = 0.01;
    matrix = matrix + eps*Eigen::MatrixXd::Random(N,N);
    Eigen::MatrixXd tmat = matrix.transpose();
    matrix = matrix + tmat; 

    for (int i = 0; i<N; i++)
    {
        //matrix(i,i) =  static_cast<double> (i+1);   
        matrix(i,i) =  static_cast<double> (1. + (std::rand() %1000 ) / 10.);
    }

    return matrix;
}

Eigen::MatrixXd init_matrix_B(int N)
{
    Eigen::MatrixXd matrix = Eigen::MatrixXd::Zero(N,N);
    double eps = 0.001;
    matrix = matrix + eps*Eigen::MatrixXd::Random(N,N);
    Eigen::MatrixXd tmat = matrix.transpose();
    matrix = matrix + tmat; 

    for (int i = 0; i<N; i++)
        matrix(i,i) =  static_cast<double> (1.);
    

    return matrix;
}

int main (int argc, char *argv[]){

    // parse the input
    cxxopts::Options options(argv[0],  "Eigen Davidson Iterative Solver");
    options.positional_help("[optional args]").show_positional_help();
    options.add_options()
        ("size", "dimension of the matrix", cxxopts::value<std::string>()->default_value("100"))
        ("neigen", "number of eigenvalues required", cxxopts::value<std::string>()->default_value("5"))
        ("jocc", "use Jacobi-Davidson", cxxopts::value<bool>())
        ("linsolve", "method to solve the linear system of JOCC (0:CG, 1:GMRES, 2:LLT)", cxxopts::value<std::string>()->default_value("0"))
        ("help", "Print the help", cxxopts::value<bool>());
    auto result = options.parse(argc,argv);

    if (result.count("help"))
    {
        std::cout << options.help({""}) << std::endl;
        exit(0);
    }


    int size = std::stoi(result["size"].as<std::string>(),nullptr);
    int neigen = std::stoi(result["neigen"].as<std::string>(),nullptr);
    bool jocc = result["jocc"].as<bool>();
    int linsolve = std::stoi(result["linsolve"].as<std::string>(),nullptr);
    bool help = result["help"].as<bool>();



    // chrono    
    std::chrono::time_point<std::chrono::system_clock> start, end;
    std::chrono::duration<double> elapsed_time;

    std::cout << "Matrix size : " << size << "x" << size << std::endl;
    std::cout << "Num Threads : " <<  Eigen::nbThreads() << std::endl;

    //=======================================
    // Full matrix
    //=======================================

    // init the matrix
    Eigen::MatrixXd A = init_matrix(size);
    Eigen::MatrixXd B = init_matrix_B(size);

    // start the solver
    start = std::chrono::system_clock::now();
    DavidsonSolver DS;

    if (jocc)
    {
        DS.set_jacobi_correction();
        DS.set_jacobi_linsolve(linsolve);
    }

    DS.solve(A,B,neigen);
    end = std::chrono::system_clock::now();
    elapsed_time = end-start;
    std::cout << std::endl << "Davidson               : " << elapsed_time.count() << " secs" <<  std::endl;
    
    // Eigen solver 
    start = std::chrono::system_clock::now();
    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> es(A,B);
    end = std::chrono::system_clock::now();
    elapsed_time = end-start;
    std::cout << "Eigen                  : " << elapsed_time.count() << " secs" <<  std::endl;

    auto dseig = DS.eigenvalues();
    auto eig = es.eigenvalues().head(neigen);
    std::cout << std::endl << "      Davidson  \tEigen" << std::endl;
    for(int i=0; i< neigen; i++)
        printf("#% 4d %8.7f \t%8.7f\n",i,dseig(i),eig(i));
        
    //=======================================
    // Matrix Free
    //=======================================

    // // Create Operator
    // DavidsonOperator Aop(size);
    // Eigen::MatrixXd Afull = Aop.get_full_mat();

    // // Davidosn Solver
    // start = std::chrono::system_clock::now();
    // DavidsonSolver DSop;
    // if (jocc)
    // {
    //     DSop.set_jacobi_correction();
    //     DSop.set_jacobi_linsolve(linsolve);
    // }
    // DSop.solve(Aop,neigen);
    // end = std::chrono::system_clock::now();
    // elapsed_time = end-start;
    // std::cout << std::endl << "Davidson               : " << elapsed_time.count() << " secs" <<  std::endl;
    
    // // normal eigensolver
    // start = std::chrono::system_clock::now();
    // Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es2(Afull);
    // end = std::chrono::system_clock::now();
    // elapsed_time = end-start;
    // std::cout << "Eigen                  : " << elapsed_time.count() << " secs" <<  std::endl;

    // auto dseigop = DSop.eigenvalues();
    // auto eig2 = es2.eigenvalues().head(neigen);
    // std::cout << std::endl <<  "      Davidson  \tEigen" << std::endl;
    // for(int i=0; i< neigen; i++)
    //     printf("#% 4d %8.7f \t%8.7f\n",i,dseigop(i),eig2(i));
    

}