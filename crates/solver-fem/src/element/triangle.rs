use nalgebra::{Dyn, Matrix1x3, Matrix2, Matrix2x3, MatrixViewMut, Point2, Scalar, U2, Vector2};
use numeric_literals::replace_float_literals;

use crate::{
    element::{FiniteElement, ReferenceFiniteElement},
    traits::Real,
};

#[derive(Debug, Clone)]
pub struct Tri3d2Element<T: Scalar> {
    vertices: [Point2<T>; 3],
}

impl<T: Scalar> Tri3d2Element<T> {
    pub fn from_vertices(vertices: [Point2<T>; 3]) -> Self {
        Self { vertices }
    }

    pub fn vertices(&self) -> &[Point2<T>; 3] {
        &self.vertices
    }
}

impl<T: Real> ReferenceFiniteElement<T> for Tri3d2Element<T> {
    type ReferenceDim = U2;

    fn num_nodes(&self) -> usize {
        3
    }

    #[replace_float_literals(T::from_f64(literal).expect("Literal must fit in T"))]
    fn populate_basis(&self, result: &mut [T], xi: &Point2<T>) {
        result.copy_from_slice(&[
            -0.5 * xi[0] - 0.5 * xi[1],
            0.5 * xi[0] + 0.5,
            0.5 * xi[1] + 0.5,
        ])
    }

    #[replace_float_literals(T::from_f64(literal).expect("Literal must fit in T"))]
    fn populate_basis_gradients(
        &self,
        mut result: MatrixViewMut<T, Self::ReferenceDim, Dyn>,
        _xi: &Point2<T>,
    ) {
        let gradients = Matrix2x3::from_columns(&[
            Vector2::new(-0.5, -0.5),
            Vector2::new(0.5, 0.0),
            Vector2::new(0.0, 0.5),
        ]);

        result.copy_from(&gradients);
    }
}

impl<T: Real> FiniteElement<T> for Tri3d2Element<T> {
    type GeometryDim = U2;

    #[allow(non_snake_case)]
    fn reference_jacobian(&self, xi: &Point2<T>) -> Matrix2<T> {
        let X = Matrix2x3::from_fn(|i, j| self.vertices[j][i]);
        let mut G = Matrix2x3::zeros();
        self.populate_basis_gradients(G.as_view_mut(), xi);
        X * G.transpose()
    }

    #[allow(non_snake_case)]
    fn map_reference_coords(&self, xi: &Point2<T>) -> Point2<T> {
        let X = Matrix2x3::from_fn(|i, j| self.vertices[j][i]);
        let mut N = Matrix1x3::zeros();
        self.populate_basis(N.as_mut_slice(), xi);
        Point2::from(X * N.transpose())
    }
}
