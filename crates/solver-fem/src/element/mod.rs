use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, Dyn, MatrixViewMut, OMatrix, OPoint, Scalar};

use crate::mesh::connectivity::Connectivity;
use crate::traits::SmallDim;

pub mod hexahedron;
pub mod quadrilateral;
pub mod segment;
pub mod tetrahedron;
pub mod triangle;

pub trait ReferenceFiniteElement<T>
where
    T: Scalar,
    DefaultAllocator: Allocator<Self::ReferenceDim>,
{
    type ReferenceDim: SmallDim;

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
    DefaultAllocator: Allocator<Self::GeometryDim>
        + Allocator<Self::ReferenceDim>
        + Allocator<Self::GeometryDim, Self::ReferenceDim>,
{
    type GeometryDim: SmallDim;

    fn reference_jacobian(
        &self,
        reference_coords: &OPoint<T, Self::ReferenceDim>,
    ) -> OMatrix<T, Self::GeometryDim, Self::ReferenceDim>;

    fn map_reference_coords(
        &self,
        reference_coords: &OPoint<T, Self::ReferenceDim>,
    ) -> OPoint<T, Self::GeometryDim>;
}

pub trait VolumetricFiniteElement<T>:
    FiniteElement<T, ReferenceDim = <Self as FiniteElement<T>>::GeometryDim>
where
    T: Scalar,
    DefaultAllocator: Allocator<Self::GeometryDim>
        + Allocator<Self::ReferenceDim>
        + Allocator<Self::GeometryDim, Self::ReferenceDim>,
{
}

impl<T, E> VolumetricFiniteElement<T> for E
where
    T: Scalar,
    E: FiniteElement<T, ReferenceDim = <Self as FiniteElement<T>>::GeometryDim>,
    DefaultAllocator: Allocator<Self::GeometryDim>
        + Allocator<Self::ReferenceDim>
        + Allocator<Self::GeometryDim, Self::ReferenceDim>,
{
}

pub trait ElementConnectivity<T>: Connectivity
where
    T: Scalar,
    DefaultAllocator: Allocator<Self::GeometryDim>
        + Allocator<Self::ReferenceDim>
        + Allocator<Self::GeometryDim, Self::ReferenceDim>,
{
    type Element: FiniteElement<T, GeometryDim = Self::GeometryDim, ReferenceDim = Self::ReferenceDim>;
    type GeometryDim: SmallDim;
    type ReferenceDim: SmallDim;

    fn element(&self, all_vertices: &[OPoint<T, Self::GeometryDim>]) -> Option<Self::Element>;
}
