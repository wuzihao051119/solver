use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, DimName, Dyn, MatrixViewMut, OMatrix, OPoint, Scalar};

use crate::element::{ElementConnectivity, FiniteElement, ReferenceFiniteElement};
use crate::mesh::Mesh;
use crate::traits::SmallDim;

#[derive(Debug, Clone)]
pub struct ElementInSpace<'a, Space> {
    space: &'a Space,
    element_index: usize,
}

impl<'a, Space> ElementInSpace<'a, Space> {
    pub fn from_space_and_element_index(space: &'a Space, element_index: usize) -> Self {
        Self {
            space,
            element_index,
        }
    }
}

impl<'a, T, Space> ReferenceFiniteElement<T> for ElementInSpace<'a, Space>
where
    T: Scalar,
    Space: FiniteElementSpace<T>,
    DefaultAllocator: Allocator<Space::GeometryDim>
        + Allocator<Space::ReferenceDim>
        + Allocator<Space::GeometryDim, Space::ReferenceDim>,
{
    type ReferenceDim = Space::ReferenceDim;

    fn num_nodes(&self) -> usize {
        self.space.element_node_count(self.element_index)
    }

    fn populate_basis(
        &self,
        basis_values: &mut [T],
        reference_coords: &OPoint<T, Self::ReferenceDim>,
    ) {
        self.space
            .populate_element_basis(self.element_index, basis_values, reference_coords);
    }

    fn populate_basis_gradients(
        &self,
        basis_gradients: MatrixViewMut<T, Self::ReferenceDim, Dyn>,
        reference_coords: &OPoint<T, Self::ReferenceDim>,
    ) {
        self.space.populate_element_gradients(
            self.element_index,
            basis_gradients,
            reference_coords,
        );
    }
}

impl<'a, T, Space> FiniteElement<T> for ElementInSpace<'a, Space>
where
    T: Scalar,
    Space: FiniteElementSpace<T>,
    DefaultAllocator: Allocator<Space::GeometryDim>
        + Allocator<Space::ReferenceDim>
        + Allocator<Space::GeometryDim, Space::ReferenceDim>,
{
    type GeometryDim = Space::GeometryDim;

    fn reference_jacobian(
        &self,
        reference_coords: &OPoint<T, Self::ReferenceDim>,
    ) -> OMatrix<T, Self::GeometryDim, Self::ReferenceDim> {
        self.space
            .element_reference_jacobian(self.element_index, reference_coords)
    }

    fn map_reference_coords(
        &self,
        reference_coords: &OPoint<T, Self::ReferenceDim>,
    ) -> OPoint<T, Self::GeometryDim> {
        self.space
            .map_element_reference_coords(self.element_index, reference_coords)
    }
}

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

impl<T, S> VolumetricFiniteElementSpace<T> for S
where
    T: Scalar,
    S: FiniteElementSpace<T, GeometryDim = <Self as FiniteElementSpace<T>>::ReferenceDim>,
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
