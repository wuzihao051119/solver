use nalgebra::{DVectorViewMut, OPoint, Scalar, U1};

pub fn assemble_element_source_vector<T, Element, Source>(
    mut _output: DVectorViewMut<T>,
    _element: &Element,
    _source: &Source,
    _quadrature_weights: &[T],
    _quadrature_points: &[OPoint<T, U1>],
) where
    T: Scalar,
{
}
