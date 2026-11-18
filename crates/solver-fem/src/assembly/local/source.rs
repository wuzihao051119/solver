use std::marker::PhantomData;

use itertools::izip;
use nalgebra::allocator::Allocator;
use nalgebra::{
    DVectorViewMut, DefaultAllocator, DimName, Dyn, MatrixView, MatrixViewMut, OPoint, OVector,
    Scalar, U1,
};

use crate::assembly::local::quadrature_table::QuadratureTable;
use crate::assembly::local::{ElementConnectivityAssembler, ElementVectorAssembler};
use crate::assembly::operators::Operator;
use crate::element::VolumetricFiniteElement;
use crate::space::{ElementInSpace, VolumetricFiniteElementSpace};
use crate::traits::Real;

pub trait SourceFunction<T, GeometryDim>: Operator
where
    T: Scalar,
    GeometryDim: DimName,
    DefaultAllocator: Allocator<GeometryDim> + Allocator<Self::SolutionDim>,
{
    fn evaluate(&self, coords: &OPoint<T, GeometryDim>) -> OVector<T, Self::SolutionDim>;
}

pub struct ElementSourceAssemblerBuilder<T, SpaceRef, SourceRef, QTableRef> {
    space: SpaceRef,
    source: SourceRef,
    qtable: QTableRef,
    marker: PhantomData<T>,
}

#[allow(clippy::new_without_default)]
impl ElementSourceAssemblerBuilder<(), (), (), ()> {
    pub fn new() -> Self {
        Self {
            space: (),
            source: (),
            qtable: (),
            marker: PhantomData,
        }
    }
}

impl<SpaceRef, SourceRef, QTableRef>
    ElementSourceAssemblerBuilder<(), SpaceRef, SourceRef, QTableRef>
{
    pub fn with_finite_element_space<Space>(
        self,
        space: &Space,
    ) -> ElementSourceAssemblerBuilder<(), &Space, SourceRef, QTableRef> {
        ElementSourceAssemblerBuilder {
            space,
            source: self.source,
            qtable: self.qtable,
            marker: PhantomData,
        }
    }

    pub fn with_source<Source>(
        self,
        source: &Source,
    ) -> ElementSourceAssemblerBuilder<(), SpaceRef, &Source, QTableRef> {
        ElementSourceAssemblerBuilder {
            space: self.space,
            source,
            qtable: self.qtable,
            marker: PhantomData,
        }
    }

    pub fn with_quadrature_table<QTable>(
        self,
        qtable: &QTable,
    ) -> ElementSourceAssemblerBuilder<(), SpaceRef, SourceRef, &QTable> {
        ElementSourceAssemblerBuilder {
            space: self.space,
            source: self.source,
            qtable,
            marker: PhantomData,
        }
    }
}

impl<'a, Space, Source, QTable>
    ElementSourceAssemblerBuilder<(), &'a Space, &'a Source, &'a QTable>
{
    pub fn build<T>(self) -> ElementSourceAssembler<'a, T, Space, Source, QTable> {
        ElementSourceAssembler {
            space: self.space,
            source: self.source,
            qtable: self.qtable,
            marker: PhantomData,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ElementSourceAssembler<'a, T, Space, Source, QTable> {
    space: &'a Space,
    source: &'a Source,
    qtable: &'a QTable,
    marker: PhantomData<T>,
}

impl<'a, T, Space, Source, QTable> ElementConnectivityAssembler
    for ElementSourceAssembler<'a, T, Space, Source, QTable>
where
    T: Scalar,
    Space: VolumetricFiniteElementSpace<T>,
    Source: Operator,
    QTable: QuadratureTable<T, Space::ReferenceDim>,
    DefaultAllocator:
        Allocator<Space::ReferenceDim> + Allocator<Space::ReferenceDim, Space::ReferenceDim>,
{
    fn solution_dim(&self) -> usize {
        Source::SolutionDim::dim()
    }

    fn num_elements(&self) -> usize {
        self.space.num_elements()
    }

    fn num_nodes(&self) -> usize {
        self.space.num_nodes()
    }

    fn element_node_count(&self, element_index: usize) -> usize {
        self.space.element_node_count(element_index)
    }

    fn populate_element_nodes(&self, output: &mut [usize], element_index: usize) {
        self.space.populate_element_nodes(output, element_index);
    }
}

impl<'a, T, Space, Source, QTable> ElementVectorAssembler<T>
    for ElementSourceAssembler<'a, T, Space, Source, QTable>
where
    T: Real,
    Space: VolumetricFiniteElementSpace<T>,
    Source: SourceFunction<T, Space::GeometryDim>,
    QTable: QuadratureTable<T, Space::ReferenceDim>,
    DefaultAllocator: Allocator<Space::GeometryDim>
        + Allocator<Source::SolutionDim>
        + Allocator<Space::GeometryDim, Space::GeometryDim>,
{
    fn assemble_element_vector_into(
        &self,
        element_index: usize,
        output: DVectorViewMut<T>,
    ) -> eyre::Result<()> {
        let element = ElementInSpace::from_space_and_element_index(self.space, element_index);
        let quadrature_size = self.qtable.element_quadrature_size(element_index);
        let mut points = vec![OPoint::origin(); quadrature_size];
        let mut weights = vec![T::zero(); quadrature_size];
        self.qtable
            .populate_element_quadrature(element_index, &mut points, &mut weights);

        assemble_element_source_vector(output, &element, self.source, &weights, &points);

        Ok(())
    }
}

pub fn assemble_element_source_vector<T, Element, Source>(
    mut output: DVectorViewMut<T>,
    element: &Element,
    source: &Source,
    quadrature_weights: &[T],
    quadrature_points: &[OPoint<T, Element::ReferenceDim>],
) where
    T: Real,
    Element: VolumetricFiniteElement<T>,
    Source: SourceFunction<T, Element::GeometryDim>,
    DefaultAllocator: Allocator<Element::GeometryDim>
        + Allocator<Source::SolutionDim>
        + Allocator<Element::GeometryDim, Element::GeometryDim>,
{
    assert_eq!(
        quadrature_weights.len(),
        quadrature_points.len(),
        "Number of quadrature weights must be equal to number of points.",
    );

    let n = element.num_nodes();
    assert_eq!(
        output.len(),
        n * Source::SolutionDim::dim(),
        "Length of output vector must be consistent with number of nodes and solution dim",
    );
    let mut output = MatrixViewMut::from_slice_generic(
        output.as_mut_slice(),
        Source::SolutionDim::name(),
        Dyn(n),
    );

    output.fill(T::zero());

    let mut basis_values_buffer = vec![T::zero(); n];

    let quadrature_iter = izip!(quadrature_weights, quadrature_points);
    for (weight, point) in quadrature_iter {
        element.populate_basis(basis_values_buffer.as_mut_slice(), point);

        let x = element.map_reference_coords(point);
        let j = element.reference_jacobian(point);
        let f = source.evaluate(&x);

        let phi = MatrixView::from_slice_generic(&basis_values_buffer, U1::name(), Dyn(n));
        output.gemm(*weight * j.determinant().abs(), &f, &phi, T::one());
    }
}
