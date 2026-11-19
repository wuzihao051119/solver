use nalgebra::allocator::Allocator;
use nalgebra::{
    DMatrixViewMut, DVectorView, DefaultAllocator, DimName, OMatrix, OPoint, OVector, Scalar, min,
};

use crate::traits::Real;

pub mod laplace;

#[derive(Debug, Clone)]
pub enum Symmetry {
    NonSymmetric,
    Symmetric,
}

pub trait Operator {
    type SolutionDim: DimName;
}

pub trait SourceFunction<T, GeometryDim>: Operator
where
    T: Scalar,
    GeometryDim: DimName,
    DefaultAllocator: Allocator<GeometryDim> + Allocator<Self::SolutionDim>,
{
    fn evaluate(&self, coords: &OPoint<T, GeometryDim>) -> OVector<T, Self::SolutionDim>;
}

pub trait EllipticOperator<T, GeometryDim>: Operator
where
    T: Scalar,
    GeometryDim: DimName,
    DefaultAllocator:
        Allocator<GeometryDim, Self::SolutionDim> + Allocator<Self::SolutionDim, GeometryDim>,
{
    fn compute_elliptic_operator(
        &self,
        gradient: &OMatrix<T, GeometryDim, Self::SolutionDim>,
    ) -> OMatrix<T, GeometryDim, Self::SolutionDim>;

    fn compute_elliptic_operator_transpose(
        &self,
        gradient: &OMatrix<T, GeometryDim, Self::SolutionDim>,
    ) -> OMatrix<T, Self::SolutionDim, GeometryDim> {
        self.compute_elliptic_operator(gradient).transpose()
    }
}

pub trait EllipticContraction<T, GeometryDim>: Operator
where
    T: Real,
    GeometryDim: DimName,
    DefaultAllocator: Allocator<GeometryDim>
        + Allocator<GeometryDim, Self::SolutionDim>
        + Allocator<Self::SolutionDim, Self::SolutionDim>,
{
    fn contract(
        &self,
        gradient: &OMatrix<T, GeometryDim, Self::SolutionDim>,
        a: &OVector<T, GeometryDim>,
        b: &OVector<T, GeometryDim>,
    ) -> OMatrix<T, Self::SolutionDim, Self::SolutionDim>;

    fn symmetry(&self) -> Symmetry {
        Symmetry::NonSymmetric
    }

    #[allow(non_snake_case)]
    fn accumulate_contractions_into(
        &self,
        mut output: DMatrixViewMut<T>,
        alpha: T,
        gradient: &OMatrix<T, GeometryDim, Self::SolutionDim>,
        a: DVectorView<T>,
        b: DVectorView<T>,
    ) {
        let d = GeometryDim::dim();
        let s = Self::SolutionDim::dim();
        assert!(
            a.len().is_multiple_of(d),
            "Dimension of a must be divisible by d (GeometryDim)",
        );
        assert!(
            b.len().is_multiple_of(d),
            "Dimension of b must be divisible by d (GeometryDim)",
        );
        #[allow(clippy::manual_is_multiple_of)]
        let M = a.len() / d;
        let N = b.len() / d;
        assert_eq!(
            output.nrows(),
            s * M,
            "Number of rows in output matrix is not consistent with a",
        );
        assert_eq!(
            output.ncols(),
            s * N,
            "Number of columns in output matrix is not consistent with b",
        );
        let s_times_s = (Self::SolutionDim::name(), Self::SolutionDim::name());
        let symmetry = self.symmetry();

        for J in 0..N {
            let row_range = match symmetry {
                Symmetry::Symmetric => min(J + 1, M),
                Symmetry::NonSymmetric => M,
            };
            for I in 0..row_range {
                let a_I = a.rows_generic(d * I, GeometryDim::name()).clone_owned();
                let b_J = b.rows_generic(d * J, GeometryDim::name()).clone_owned();
                let mut c_IJ = output.generic_view_mut((s * I, s * J), s_times_s);
                let contraction = self.contract(gradient, &a_I, &b_J);
                c_IJ += contraction * alpha;
            }
        }
    }
}
