use itertools::izip;
use nalgebra::allocator::Allocator;
use nalgebra::{
    DVectorViewMut, DefaultAllocator, DimName, Dyn, MatrixView, MatrixViewMut, OPoint, OVector,
    Scalar, U1,
};

use crate::assembly::operators::Operators;
use crate::element::VolumetricFiniteElement;
use crate::traits::Real;

pub trait SourceFunction<T, GeometryDim>: Operators
where
    T: Scalar,
    GeometryDim: DimName,
    DefaultAllocator: Allocator<GeometryDim> + Allocator<Self::SolutionDim>,
{
    fn evaluate(&self, coords: &OPoint<T, GeometryDim>) -> OVector<T, Self::SolutionDim>;
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
