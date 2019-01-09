#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/QR>
#include <Eigen/Eigenvalues>
#include <chrono>
#include <cxxopts.hpp>
#include "DavidsonSolver.hpp"
#include "DavidsonOperator.hpp"

typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> Mat;
typedef Eigen::Matrix<double,Eigen::Dynamic,1> Vect;
typedef Eigen::Matrix<double,1,Eigen::Dynamic> Vect_row;

Mat init_matrix(int N)
{
    Mat matrix = Mat::Zero(N,N);
    double eps = 0.01;
    matrix = matrix + eps*Mat::Random(N,N);
    Mat tmat = matrix.transpose();
    matrix = matrix + tmat; 

    for (int i = 0; i<N; i++)
    {
        //matrix(i,i) =  static_cast<double> (i+1);   
        matrix(i,i) =  static_cast<double> (1. + (std::rand() %1000 ) / 10.);
    }

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
    Mat A = init_matrix(size);

    // start the solver
    start = std::chrono::system_clock::now();
    DavidsonSolver DS;

    if (jocc)
    {
        DS.set_jacobi_correction();
        DS.set_jacobi_linsolve(linsolve);
    }

    DS.solve(A,neigen);
    end = std::chrono::system_clock::now();
    elapsed_time = end-start;
    std::cout << "Davidson               : " << elapsed_time.count() << " secs" <<  std::endl;
    
    // Eigen solver 
    start = std::chrono::system_clock::now();
    Eigen::SelfAdjointEigenSolver<Mat> es(A);
    end = std::chrono::system_clock::now();
    elapsed_time = end-start;
    std::cout << "Eigen                  : " << elapsed_time.count() << " secs" <<  std::endl;

    auto dseig = DS.eigenvalues();
    auto eig = es.eigenvalues().head(neigen);
    std::cout << "Davidson \tEigen" << std::endl;
    for(int i=0; i< neigen; i++)
        printf("%8.7f \t%8.7f\n",dseig(i),eig(i));
        
    //=======================================
    // Matrix Free
    //=======================================

    // Create Operator
    DavidsonOperator Aop(size);
    Mat Afull = Aop.get_full_mat();

    // Davidosn Solver
    start = std::chrono::system_clock::now();
    DavidsonSolver DSop;
    DSop.solve(Aop,neigen);
    end = std::chrono::system_clock::now();
    elapsed_time = end-start;
    std::cout << "Davidson               : " << elapsed_time.count() << " secs" <<  std::endl;
    
    // normal eigensolver
    start = std::chrono::system_clock::now();
    Eigen::SelfAdjointEigenSolver<Mat> es2(Afull);
    end = std::chrono::system_clock::now();
    elapsed_time = end-start;
    std::cout << "Eigen                  : " << elapsed_time.count() << " secs" <<  std::endl;

    auto dseigop = DSop.eigenvalues();
    auto eig2 = es2.eigenvalues().head(neigen);
    std::cout << "Davidson \tEigen" << std::endl;
    for(int i=0; i< neigen; i++)
        printf("%8.7f \t%8.7f\n",dseigop(i),eig2(i));

}