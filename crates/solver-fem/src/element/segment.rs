use nalgebra::{Dyn, Matrix1, Matrix1x2, MatrixViewMut, Point1, Scalar, U1};
use numeric_literals::replace_float_literals;

use crate::element::{ElementConnectivity, FiniteElement, ReferenceFiniteElement};
use crate::mesh::connectivity::{Connectivity, Segment2d1Connectivity};
use crate::traits::Real;

#[derive(Debug, Clone)]
pub struct Segment2d1Element<T: Scalar> {
    vertices: [Point1<T>; 2],
}

impl<T: Scalar> Segment2d1Element<T> {
    pub fn from_vertices(vertices: [Point1<T>; 2]) -> Self {
        Self { vertices }
    }

    pub fn vertices(&self) -> &[Point1<T>; 2] {
        &self.vertices
    }
}

impl<T: Real> ReferenceFiniteElement<T> for Segment2d1Element<T> {
    type ReferenceDim = U1;

    fn num_nodes(&self) -> usize {
        2
    }

    #[replace_float_literals(T::from_f64(literal).expect("Literal must fit in T"))]
    fn populate_basis(&self, result: &mut [T], xi: &Point1<T>) {
        result.copy_from_slice(&[(1.0 - xi[0]) / 2.0, (1.0 + xi[0]) / 2.0]);
    }

    #[replace_float_literals(T::from_f64(literal).expect("Literal must fit in T"))]
    fn populate_basis_gradients(
        &self,
        mut result: MatrixViewMut<T, Self::ReferenceDim, Dyn>,
        _xi: &Point1<T>,
    ) {
        let gradients = Matrix1x2::from_row_slice(&[-0.5, 0.5]);

        result.copy_from(&gradients);
    }
}

impl<T: Real> FiniteElement<T> for Segment2d1Element<T> {
    type GeometryDim = U1;

    #[allow(non_snake_case)]
    #[replace_float_literals(T::from_f64(literal).expect("Literal must fit in T"))]
    fn reference_jacobian(&self, _xi: &Point1<T>) -> Matrix1<T> {
        let a = &self.vertices[0].coords;
        let b = &self.vertices[1].coords;
        (b - a) / 2.0
    }

    #[allow(non_snake_case)]
    fn map_reference_coords(&self, xi: &Point1<T>) -> Point1<T> {
        let a = &self.vertices[0].coords;
        let b = &self.vertices[1].coords;
        let mut N = Matrix1x2::zeros();
        self.populate_basis(N.as_mut_slice(), xi);
        Point1::from(a * N[0] + b * N[1])
    }
}

impl Connectivity for Segment2d1Connectivity {
    fn num_faces(&self) -> usize {
        0
    }

    fn vertex_indices(&self) -> &[usize] {
        &self.0
    }
}

impl<T: Real> ElementConnectivity<T> for Segment2d1Connectivity {
    type Element = Segment2d1Element<T>;
    type GeometryDim = U1;
    type ReferenceDim = U1;

    fn element(&self, vertices: &[Point1<T>]) -> Option<Self::Element> {
        let a = vertices[self.0[0]];
        let b = vertices[self.0[1]];
        Some(Segment2d1Element::from_vertices([a, b]))
    }
}
