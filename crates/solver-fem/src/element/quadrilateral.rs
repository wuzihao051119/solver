use nalgebra::{Dyn, Matrix1x4, Matrix2, Matrix2x4, MatrixViewMut, Point2, Scalar, U2, Vector2};
use numeric_literals::replace_float_literals;

use crate::{
    element::{ElementConnectivity, FiniteElement, ReferenceFiniteElement},
    mesh::connectivity::{Connectivity, Quad4d2Connectivity},
    traits::Real,
};

#[derive(Debug, Clone)]
pub struct Quad4d2Element<T: Scalar> {
    vertices: [Point2<T>; 4],
}

impl<T: Scalar> Quad4d2Element<T> {
    pub fn from_vertices(vertices: [Point2<T>; 4]) -> Self {
        Self { vertices }
    }

    pub fn vertices(&self) -> &[Point2<T>; 4] {
        &self.vertices
    }
}

impl<T: Real> ReferenceFiniteElement<T> for Quad4d2Element<T> {
    type ReferenceDim = U2;

    fn num_nodes(&self) -> usize {
        4
    }

    #[rustfmt::skip]
    #[replace_float_literals(T::from_f64(literal).expect("Literal must fit in T"))]
    fn populate_basis(&self, result: &mut [T], xi: &Point2<T>) {
        let phi = |alpha, beta, xi: &Point2<T>| (1.0 + alpha * xi[0]) * (1.0 + beta * xi[1]) / 4.0;

        result.copy_from_slice(&[
            phi(-1.0, -1.0, xi),
            phi( 1.0, -1.0, xi),
            phi( 1.0,  1.0, xi),
            phi(-1.0,  1.0, xi),
        ]);
    }

    #[rustfmt::skip]
    #[replace_float_literals(T::from_f64(literal).expect("Literal must fit in T"))]
    fn populate_basis_gradients(&self, mut result: MatrixViewMut<T, Self::ReferenceDim, Dyn>, xi: &Point2<T>) {
        let phi_grad = |alpha, beta, xi: &Point2<T>| {
            Vector2::new(
                alpha * (1.0 + beta * xi[1]) / 4.0,
                (1.0 + alpha * xi[0]) * beta / 4.0,
            )
        };

        let gradients = Matrix2x4::from_columns(&[
            phi_grad(-1.0, -1.0, xi),
            phi_grad( 1.0, -1.0, xi),
            phi_grad( 1.0,  1.0, xi),
            phi_grad(-1.0,  1.0, xi),
        ]);

        result.copy_from(&gradients);
    }
}

impl<T: Real> FiniteElement<T> for Quad4d2Element<T> {
    type GeometryDim = U2;

    #[allow(non_snake_case)]
    fn reference_jacobian(&self, xi: &Point2<T>) -> Matrix2<T> {
        let X = Matrix2x4::from_fn(|i, j| self.vertices[j][i]);
        let mut G = Matrix2x4::zeros();
        self.populate_basis_gradients(G.as_view_mut(), xi);
        X * G.transpose()
    }

    #[allow(non_snake_case)]
    fn map_reference_coords(&self, xi: &Point2<T>) -> Point2<T> {
        let X = Matrix2x4::from_fn(|i, j| self.vertices[j][i]);
        let mut N = Matrix1x4::zeros();
        self.populate_basis(N.as_mut_slice(), xi);
        Point2::from(X * N.transpose())
    }
}

impl Connectivity for Quad4d2Connectivity {
    fn num_faces(&self) -> usize {
        4
    }

    fn vertex_indices(&self) -> &[usize] {
        &self.0
    }
}

impl<T: Real> ElementConnectivity<T> for Quad4d2Connectivity {
    type Element = Quad4d2Element<T>;
    type GeometryDim = U2;
    type ReferenceDim = U2;

    fn element(&self, vertices: &[Point2<T>]) -> Option<Self::Element> {
        Some(Quad4d2Element::from_vertices([
            *vertices.get(self.0[0])?,
            *vertices.get(self.0[1])?,
            *vertices.get(self.0[2])?,
            *vertices.get(self.0[3])?,
        ]))
    }
}
