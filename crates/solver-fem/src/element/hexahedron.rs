use nalgebra::{Dyn, Matrix3, MatrixViewMut, Point3, Scalar, U3, Vector3};
use numeric_literals::replace_float_literals;

use crate::element::{ElementConnectivity, FiniteElement, ReferenceFiniteElement};
use crate::mesh::connectivity::{Connectivity, Hex8d3Connectivity};
use crate::traits::Real;
use crate::types::{Matrix1x8, Matrix3x8};

#[derive(Debug, Clone)]
pub struct Hex8d3Element<T: Scalar> {
    vertices: [Point3<T>; 8],
}

impl<T: Scalar> Hex8d3Element<T> {
    pub fn from_vertices(vertices: [Point3<T>; 8]) -> Self {
        Self { vertices }
    }

    pub fn vertices(&self) -> &[Point3<T>; 8] {
        &self.vertices
    }
}

impl<T: Real> ReferenceFiniteElement<T> for Hex8d3Element<T> {
    type ReferenceDim = U3;

    fn num_nodes(&self) -> usize {
        8
    }

    #[rustfmt::skip]
    #[replace_float_literals(T::from_f64(literal).expect("Literal must fit in T"))]
    fn populate_basis(&self, result: &mut [T], xi: &Point3<T>) {
        let phi = |alpha, beta, gamma, xi: &Point3<T>| {
            (1.0 + alpha * xi[0]) * (1.0 + beta * xi[1]) * (1.0 + gamma * xi[2]) / 8.0
        };

        result.copy_from_slice(&[
            phi(-1.0, -1.0, -1.0, xi),
            phi( 1.0, -1.0, -1.0, xi),
            phi( 1.0,  1.0, -1.0, xi),
            phi(-1.0,  1.0, -1.0, xi),
            phi(-1.0, -1.0,  1.0, xi),
            phi( 1.0, -1.0,  1.0, xi),
            phi( 1.0,  1.0,  1.0, xi),
            phi(-1.0,  1.0,  1.0, xi),
        ]);
    }

    #[rustfmt::skip]
    #[replace_float_literals(T::from_f64(literal).expect("Literal must fit in T"))]
    fn populate_basis_gradients(&self, mut result: MatrixViewMut<T, Self::ReferenceDim, Dyn>, xi: &Point3<T>) {
        let phi_grad = |alpha, beta, gamma, xi: &Point3<T>| Vector3::new(
            alpha * (1.0 + beta * xi[1]) * (1.0 + gamma * xi[2]) / 8.0,
            (1.0 + alpha * xi[0]) * beta * (1.0 + gamma * xi[2]) / 8.0,
            (1.0 + alpha * xi[0]) * (1.0 + beta * xi[1]) * gamma / 8.0,
        );

        let gradients = Matrix3x8::from_columns(&[
            phi_grad(-1.0, -1.0, -1.0, xi),
            phi_grad( 1.0, -1.0, -1.0, xi),
            phi_grad( 1.0,  1.0, -1.0, xi),
            phi_grad(-1.0,  1.0, -1.0, xi),
            phi_grad(-1.0, -1.0,  1.0, xi),
            phi_grad( 1.0, -1.0,  1.0, xi),
            phi_grad( 1.0,  1.0,  1.0, xi),
            phi_grad(-1.0,  1.0,  1.0, xi),
        ]);

        result.copy_from(&gradients);
    }
}

impl<T: Real> FiniteElement<T> for Hex8d3Element<T> {
    type GeometryDim = U3;

    #[allow(non_snake_case)]
    fn reference_jacobian(&self, xi: &Point3<T>) -> Matrix3<T> {
        let X = Matrix3x8::from_fn(|i, j| self.vertices[j][i]);
        let mut G = Matrix3x8::zeros();
        self.populate_basis_gradients(G.as_view_mut(), xi);
        X * G.transpose()
    }

    #[allow(non_snake_case)]
    fn map_reference_coords(&self, xi: &Point3<T>) -> Point3<T> {
        let X = Matrix3x8::from_fn(|i, j| self.vertices[j][i]);
        let mut N = Matrix1x8::zeros();
        self.populate_basis(N.as_mut_slice(), xi);
        Point3::from(X * N.transpose())
    }
}

impl Connectivity for Hex8d3Connectivity {
    fn num_faces(&self) -> usize {
        6
    }

    fn vertex_indices(&self) -> &[usize] {
        &self.0
    }
}

impl<T: Real> ElementConnectivity<T> for Hex8d3Connectivity {
    type Element = Hex8d3Element<T>;
    type GeometryDim = U3;
    type ReferenceDim = U3;

    fn element(&self, vertices: &[Point3<T>]) -> Option<Self::Element> {
        Some(Hex8d3Element::from_vertices([
            *vertices.get(self.0[0])?,
            *vertices.get(self.0[1])?,
            *vertices.get(self.0[2])?,
            *vertices.get(self.0[3])?,
            *vertices.get(self.0[4])?,
            *vertices.get(self.0[5])?,
            *vertices.get(self.0[6])?,
            *vertices.get(self.0[7])?,
        ]))
    }
}
