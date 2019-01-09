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
    cxxopts::Options options(argv[0],  "Test of the Davidosn routine");
    options.positional_help("[optional args]").show_positional_help();
    options.add_options()("size", "dimension of the matrix", cxxopts::value<int>(), "100");

    auto result = options.parse(argc,argv);
    int size = result["size"].as<int>();

    // number of eignvalues required
    int neigen = 5;

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