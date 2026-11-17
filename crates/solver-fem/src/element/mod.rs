use nalgebra::{DefaultAllocator, DimName, Dyn, MatrixViewMut, OMatrix, OPoint, Scalar};

use crate::traits::{BiDimAllocator, DimAllocator};

pub mod hexahedron;
pub mod quadrilateral;
pub mod segment;
pub mod tetrahedron;
pub mod triangle;

pub trait ReferenceFiniteElement<T>
where
    T: Scalar,
    DefaultAllocator: DimAllocator<Self::ReferenceDim>,
{
    type ReferenceDim: DimName;

    fn num_nodes(&self) -> usize;

    fn populate_basis(
        &self,
        basis_values: &mut [T],
        reference_coords: &OPoint<T, Self::ReferenceDim>,
    );

    fn populate_basis_gradients(
        &self,
        basis_gradients: MatrixViewMut<T, Self::ReferenceDim, Dyn>,
        reference_coords: &OPoint<T, Self::ReferenceDim>,
    );
}

pub trait FiniteElement<T>: ReferenceFiniteElement<T>
where
    T: Scalar,
    DefaultAllocator: BiDimAllocator<Self::GeometryDim, Self::ReferenceDim>,
{
    type GeometryDim: DimName;

    fn reference_jacobian(
        &self,
        reference_coords: &OPoint<T, Self::ReferenceDim>,
    ) -> OMatrix<T, Self::GeometryDim, Self::ReferenceDim>;

    fn map_reference_coords(
        &self,
        reference_coords: &OPoint<T, Self::ReferenceDim>,
    ) -> OPoint<T, Self::GeometryDim>;
}
