use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, DimName, OVector, U1};

use crate::assembly::operators::{EllipticContraction, EllipticOperator, Operator, Symmetry};
use crate::traits::Real;

#[derive(Debug, Clone)]
pub struct LaplaceOperator;

impl Operator for LaplaceOperator {
    type SolutionDim = U1;
}

impl<T, D> EllipticOperator<T, D> for LaplaceOperator
where
    T: Real,
    D: DimName,
    DefaultAllocator: Allocator<D> + Allocator<Self::SolutionDim, D>,
{
    fn compute_elliptic_operator(
        &self,
        gradient: &nalgebra::OMatrix<T, D, Self::SolutionDim>,
    ) -> nalgebra::OMatrix<T, D, Self::SolutionDim> {
        gradient.clone()
    }
}

impl<T, D> EllipticContraction<T, D> for LaplaceOperator
where
    T: Real,
    D: DimName,
    DefaultAllocator: Allocator<D> + Allocator<Self::SolutionDim, Self::SolutionDim>,
{
    fn contract(
        &self,
        _gradient: &nalgebra::OMatrix<T, D, Self::SolutionDim>,
        a: &nalgebra::OVector<T, D>,
        b: &nalgebra::OVector<T, D>,
    ) -> nalgebra::OMatrix<T, Self::SolutionDim, Self::SolutionDim> {
        OVector::<T, U1>::new(a.dot(b))
    }

    fn symmetry(&self) -> Symmetry {
        Symmetry::Symmetric
    }
}
