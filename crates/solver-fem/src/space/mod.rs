use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, DimName, Dyn, MatrixViewMut, OMatrix, OPoint, Scalar};

use crate::element::{ElementConnectivity, FiniteElement, ReferenceFiniteElement};
use crate::mesh::Mesh;
use crate::traits::SmallDim;

pub trait FiniteElementConnectivity {
    fn num_elements(&self) -> usize;
    fn num_nodes(&self) -> usize;
    fn element_node_count(&self, element_index: usize) -> usize;
    fn populate_element_nodes(&self, nodes: &mut [usize], element_index: usize);
}

pub trait FiniteElementSpace<T>: FiniteElementConnectivity
where
    T: Scalar,
    DefaultAllocator: Allocator<Self::GeometryDim>
        + Allocator<Self::ReferenceDim>
        + Allocator<Self::GeometryDim, Self::ReferenceDim>,
{
    type GeometryDim: SmallDim;
    type ReferenceDim: SmallDim;

    fn populate_element_basis(
        &self,
        element_index: usize,
        basis_values: &mut [T],
        reference_coords: &OPoint<T, Self::ReferenceDim>,
    );

    fn populate_element_gradients(
        &self,
        element_index: usize,
        gradients: MatrixViewMut<T, Self::ReferenceDim, Dyn>,
        reference_coords: &OPoint<T, Self::ReferenceDim>,
    );

    fn element_reference_jacobian(
        &self,
        element_index: usize,
        reference_coords: &OPoint<T, Self::ReferenceDim>,
    ) -> OMatrix<T, Self::GeometryDim, Self::ReferenceDim>;

    fn map_element_reference_coords(
        &self,
        element_index: usize,
        reference_coords: &OPoint<T, Self::ReferenceDim>,
    ) -> OPoint<T, Self::GeometryDim>;
}

pub trait VolumetricFiniteElementSpace<T>:
    FiniteElementSpace<T, GeometryDim = <Self as FiniteElementSpace<T>>::ReferenceDim>
where
    T: Scalar,
    DefaultAllocator:
        Allocator<Self::ReferenceDim> + Allocator<Self::ReferenceDim, Self::ReferenceDim>,
{
}

impl<T, D, C> FiniteElementConnectivity for Mesh<T, D, C>
where
    T: Scalar,
    D: DimName,
    C: ElementConnectivity<T, GeometryDim = D>,
    DefaultAllocator: Allocator<D> + Allocator<C::ReferenceDim> + Allocator<D, C::ReferenceDim>,
{
    fn num_elements(&self) -> usize {
        self.connectivity().len()
    }

    fn num_nodes(&self) -> usize {
        self.vertices().len()
    }

    fn element_node_count(&self, element_index: usize) -> usize {
        self.connectivity()
            .get(element_index)
            .expect("Element index out of bounds")
            .vertex_indices()
            .len()
    }

    fn populate_element_nodes(&self, nodes: &mut [usize], element_index: usize) {
        let indices = self
            .connectivity()
            .get(element_index)
            .expect("Element index out of bounds")
            .vertex_indices();
        assert_eq!(
            indices.len(),
            nodes.len(),
            "Incompatible slice length for node population",
        );
        nodes.copy_from_slice(indices);
    }
}

impl<T, D, C> FiniteElementSpace<T> for Mesh<T, D, C>
where
    T: Scalar,
    D: SmallDim,
    C: ElementConnectivity<T, GeometryDim = D>,
    DefaultAllocator: Allocator<D> + Allocator<C::ReferenceDim> + Allocator<D, C::ReferenceDim>,
{
    type GeometryDim = D;
    type ReferenceDim = C::ReferenceDim;

    fn populate_element_basis(
        &self,
        element_index: usize,
        basis_values: &mut [T],
        reference_coords: &OPoint<T, Self::ReferenceDim>,
    ) {
        let element = self
            .connectivity()
            .get(element_index)
            .expect("Element index out of bounds")
            .element(self.vertices())
            .unwrap();
        assert_eq!(
            basis_values.len(),
            element.num_nodes(),
            "Incompatible slice shape for basis values",
        );
        element.populate_basis(basis_values, reference_coords);
    }

    fn populate_element_gradients(
        &self,
        element_index: usize,
        gradients: MatrixViewMut<T, Self::ReferenceDim, Dyn>,
        reference_coords: &OPoint<T, Self::ReferenceDim>,
    ) {
        let element = self
            .connectivity()
            .get(element_index)
            .expect("Element index out of bounds")
            .element(self.vertices())
            .unwrap();
        assert_eq!(
            gradients.shape(),
            (C::ReferenceDim::dim(), element.num_nodes()),
            "Incompatible slice shape for basis gradients",
        );
        element.populate_basis_gradients(gradients, reference_coords);
    }

    fn element_reference_jacobian(
        &self,
        element_index: usize,
        reference_coords: &OPoint<T, Self::ReferenceDim>,
    ) -> OMatrix<T, Self::GeometryDim, Self::ReferenceDim> {
        let element = self
            .connectivity()
            .get(element_index)
            .expect("Element index out of bounds")
            .element(self.vertices())
            .unwrap();
        element.reference_jacobian(reference_coords)
    }

    fn map_element_reference_coords(
        &self,
        element_index: usize,
        reference_coords: &OPoint<T, Self::ReferenceDim>,
    ) -> OPoint<T, Self::GeometryDim> {
        let element = self
            .connectivity()
            .get(element_index)
            .expect("Element index out of bounds")
            .element(self.vertices())
            .unwrap();
        element.map_reference_coords(reference_coords)
    }
}
