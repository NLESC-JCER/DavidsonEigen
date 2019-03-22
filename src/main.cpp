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


#include <iostream>
#include <fstream>
#include <string>

using namespace std;

#define MAXBUFSIZE  ((int) 1e6)

Eigen::MatrixXd readMatrix(const char *filename)
    {
    int cols = 0, rows = 0;
    double buff[MAXBUFSIZE];

    // Read numbers from file into buffer.
    ifstream infile;
    infile.open(filename);
    while (! infile.eof())
        {
        string line;
        getline(infile, line);

        int temp_cols = 0;
        stringstream stream(line);
        while(! stream.eof())
            stream >> buff[cols*rows+temp_cols++];

        if (temp_cols == 0)
            continue;

        if (cols == 0)
            cols = temp_cols;

        rows++;
        }

    infile.close();

    rows--;

    // Populate matrix with numbers.
    Eigen::MatrixXd result(rows,cols);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            result(i,j) = buff[ cols*i+j ];

    return result;
    };



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
        ("reorder", "reorder diagonal elements" , cxxopts::value<bool>())
        ("linsolve", "method to solve the linear system of JOCC (CG, GMRES, LLT)", cxxopts::value<std::string>()->default_value("CG"))
        ("init", "method to itialize the eigenvector (target, indentity, random)", cxxopts::value<std::string>()->default_value("target"))
        ("tol", "tolerance on the residue norm", cxxopts::value<std::string>()->default_value("1E-4"))
        ("lstol", "tolerance of the linear solver", cxxopts::value<std::string>()->default_value("0.01"))
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
    bool reorder = result["reorder"].as<bool>();
    std::string linsolve = result["linsolve"].as<std::string>();
    std::string eigen_init = result["init"].as<std::string>();
    std::string correction = result["corr"].as<std::string>();
    bool help = result["help"].as<bool>();
    double eps = std::stod(result["eps"].as<std::string>(),nullptr);
    double davidson_tol = std::stod(result["tol"].as<std::string>(),nullptr);
    double lsolve_tol = std::stod(result["lstol"].as<std::string>(),nullptr);

    // chrono    
    std::chrono::time_point<std::chrono::system_clock> start, end;
    std::chrono::duration<double> elapsed_time;

    std::cout << "Matrix size : " << size << "x" << size << std::endl;
    std::cout << "Num Threads : " <<  Eigen::nbThreads() << std::endl;
    std::cout << "eps : " <<  eps << std::endl;

    // Create Operator
    DavidsonOperator Aop(size,eps,odiag,reorder);
    Eigen::MatrixXd Afull = Aop.get_full_mat();
    std::cout << "Afull" << std::endl << Afull.block(0,0,5,5) << std::endl;

    // Davidosn Solver
    start = std::chrono::system_clock::now();
    DavidsonSolver DS;

    DS.set_guess_vectors(eigen_init);
    DS.set_correction(correction);
    DS.set_tolerance(davidson_tol);

    if (correction == "JACOBI") {
        DS.set_jacobi_linsolve(linsolve);
        DS.set_linsolve_tol(lsolve_tol);
    }

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
    std::cout << std::endl <<  "      Davidson  \tEigen \t\t Error" << std::endl;
    for(int i=0; i< neigen; i++)
        printf("#% 4d %8.7f \t%8.7f \t %4.2e\n",i,dseigop(i),eig2(i),abs(eig2(i)-dseigop(i)));

}