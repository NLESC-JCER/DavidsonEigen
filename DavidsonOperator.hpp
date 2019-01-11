#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Core>

// https://eigen.tuxfamily.org/dox/group__MatrixfreeSolverExample.html

// typedef Eigen::Matrix<double,Eigen::Dynamic,1> Vect;
// typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> Mat;
// typedef Eigen::Matrix<double,1,Eigen::Dynamic> RowVect;

#ifndef _DAVIDSON_OP_
#define _DAVIDSON_OP_


class DavidsonOperator;

namespace Eigen { namespace internal {
		template<>
		struct traits<DavidsonOperator> : public Eigen::internal::traits<Eigen::MatrixXd> {};
	}
}


class DavidsonOperator : public Eigen::EigenBase<Eigen::MatrixXd>
{
	public: 

		typedef double Scalar;
		typedef double RealScalar;
		typedef int StorageIndex;

		enum {
			ColsAtCompileTime = Eigen::Dynamic,
			MaxColsAtCompileTime = Eigen::Dynamic,
			IsRowMajor = false
		};

		Index rows() const {return this-> OpSizeVal;}
		Index cols() const {return this-> OpSizeVal;}

		template<typename Vtype>
		Eigen::Product<DavidsonOperator,Vtype,Eigen::AliasFreeProduct> operator*(const Eigen::MatrixBase<Vtype>& x) const {
			return Eigen::Product<DavidsonOperator,Vtype,Eigen::AliasFreeProduct>(*this, x.derived());
		}

		// custom API
		DavidsonOperator(int size);

		//Vect apply_to_vect(Vect b);
		//Mat apply_to_mat(Mat m);
		Eigen::MatrixXd get_full_mat();
		Eigen::VectorXd diagonal();
		Eigen::RowVectorXd row(int index);
		Eigen::VectorXd col(int index) const;

		int OpSize();

	private:

		int OpSizeVal;
		double OpEpsVal;
		Eigen::VectorXd diag_el;		
};

namespace Eigen{
	
	namespace internal{

		// replacement of the mat*vect operation
		template<typename Vtype>
		struct generic_product_impl<DavidsonOperator, Vtype, DenseShape, DenseShape, GemvProduct> 
		: generic_product_impl_base<DavidsonOperator,Vtype,generic_product_impl<DavidsonOperator,Vtype>>
		{

			typedef typename Product<DavidsonOperator,Vtype>::Scalar Scalar;

			template<typename Dest>
			static void scaleAndAddTo(Dest& dst, const DavidsonOperator& op, const Vtype &v, const Scalar& alpha)
			{
				//returns dst = alpha * op * v
				// alpha must be 1 here
				assert(alpha==Scalar(1) && "scaling is not implemented");
				EIGEN_ONLY_USED_FOR_DEBUG(alpha);

				// make the mat vect product
				for (int i=0; i<op.cols(); i++)
					dst += v(i) * op.col(i);
			}
		};

		// replacement of the mat*mat operation
		template<typename Mtype>
		struct generic_product_impl<DavidsonOperator, Mtype, DenseShape, DenseShape, GemmProduct> 
		: generic_product_impl_base<DavidsonOperator,Mtype,generic_product_impl<DavidsonOperator,Mtype>>
		{

			typedef typename Product<DavidsonOperator,Mtype>::Scalar Scalar;

			template<typename Dest>
			static void scaleAndAddTo(Dest& dst, const DavidsonOperator& op, const Mtype &m, const Scalar& alpha)
			{
				//returns dst = alpha * op * v
				// alpha must be 1 here
				assert(alpha==Scalar(1) && "scaling is not implemented");
				EIGEN_ONLY_USED_FOR_DEBUG(alpha);

				// make the mat vect product
				for (int i=0; i<op.cols();i++)
				{
					for (int j=0; j<m.cols(); j++)
						dst.col(j) += m(i,j) * op.col(i);	
				}
					
			}
		};


	}
}

#endif

