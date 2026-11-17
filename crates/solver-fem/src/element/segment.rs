use nalgebra::{Dyn, Matrix1, Matrix1x2, MatrixViewMut, Point1, Scalar, U1};
use numeric_literals::replace_float_literals;

use crate::element::{FiniteElement, ReferenceFiniteElement};
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
    fn reference_jacobian(&self, xi: &Point1<T>) -> Matrix1<T> {
        let X = Matrix1x2::from_fn(|i, j| self.vertices[j][i]);
        let mut G = Matrix1x2::zeros();
        self.populate_basis_gradients(G.as_view_mut(), xi);
        X * G.transpose()
    }

    #[allow(non_snake_case)]
    fn map_reference_coords(&self, xi: &Point1<T>) -> Point1<T> {
        let X = Matrix1x2::from_fn(|i, j| self.vertices[j][i]);
        let mut N = Matrix1x2::zeros();
        self.populate_basis(N.as_mut_slice(), xi);
        Point1::from(X * N.transpose())
    }
}
