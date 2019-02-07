#include <benchmark/benchmark.h>

#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <chrono>
#include <thread>
#include <omp.h>


Eigen::MatrixXd init_matrix(int N)
{
    Eigen::MatrixXd matrix = Eigen::MatrixXd::Zero(N,N);
    double eps = 0.01;
    matrix = matrix + eps*Eigen::MatrixXd::Random(N,N);
    Eigen::MatrixXd tmat = matrix.transpose();
    matrix = matrix + tmat; 

    for (int i = 0; i<N; i++)
        matrix(i,i) =  static_cast<double> (1. + (std::rand() %1000 ) / 10.);

    return matrix;
}


void eigen_diag(int size){

    Eigen::MatrixXd A = init_matrix(size);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> ES(A);
}

static void DIAG(benchmark::State &state) {

	for (auto _ : state)
		eigen_diag(state.range(0));
	
}
BENCHMARK(DIAG)->Arg(128)->Arg(512)->Arg(1024)->Arg(2048)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_MAIN();
