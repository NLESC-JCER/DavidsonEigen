#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Core>

#ifndef _MATRIX_FREE_OP_
#define _MATRIX_FREE_OP_

class MatrixFreeOperator;

namespace Eigen { namespace internal {
		template<>
		struct traits<MatrixFreeOperator> : public Eigen::internal::traits<Eigen::MatrixXd> {};
	}
}

class MatrixFreeOperator : public Eigen::EigenBase<Eigen::MatrixXd>
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

		Index rows() const {return this-> _size;}
		Index cols() const {return this-> _size;}

		template<typename Vtype>
		Eigen::Product<MatrixFreeOperator,Vtype,Eigen::AliasFreeProduct> operator*(const Eigen::MatrixBase<Vtype>& x) const {
			return Eigen::Product<MatrixFreeOperator,Vtype,Eigen::AliasFreeProduct>(*this, x.derived());
		}

		// custom API
		MatrixFreeOperator();

		// convenience function
		Eigen::MatrixXd get_full_mat() const;
		Eigen::VectorXd diagonal() const;
		int get_size() const {return this->_size;}
		void set_size(int N) {this->_size = N;}

		// extract row/col of the operator
		// virtual Eigen::RowVectorXd row(int index) const = 0;
		virtual Eigen::VectorXd col(int index) const = 0;	
		Eigen::VectorXd diag_el;	

	protected:

		int _size;
};

namespace Eigen{
	
	namespace internal{

		// replacement of the mat*vect operation
		template<typename Vtype>
		struct generic_product_impl<MatrixFreeOperator, Vtype, DenseShape, DenseShape, GemvProduct> 
		: generic_product_impl_base<MatrixFreeOperator,Vtype,generic_product_impl<MatrixFreeOperator,Vtype>>
		{

			typedef typename Product<MatrixFreeOperator,Vtype>::Scalar Scalar;

			template<typename Dest>
			static void scaleAndAddTo(Dest& dst, const MatrixFreeOperator& op, const Vtype &v, const Scalar& alpha)
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

		// replacement of the operator*matrix operation
		template<typename Mtype>
		struct generic_product_impl<MatrixFreeOperator, Mtype, DenseShape, DenseShape, GemmProduct> 
		: generic_product_impl_base<MatrixFreeOperator, Mtype, generic_product_impl<MatrixFreeOperator,Mtype>>
		{

			typedef typename Product<MatrixFreeOperator,Mtype>::Scalar Scalar;

			template<typename Dest>
			static void scaleAndAddTo(Dest& dst, const MatrixFreeOperator& op, const Mtype &m, const Scalar& alpha)
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

