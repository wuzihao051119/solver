use nalgebra::{Dyn, Matrix1x4, Matrix3, Matrix3x4, MatrixViewMut, Point3, Scalar, U3, Vector3};
use numeric_literals::replace_float_literals;

use crate::element::{ElementConnectivity, FiniteElement, ReferenceFiniteElement};
use crate::mesh::connectivity::{Connectivity, Tet4d3Connectivity};
use crate::traits::Real;

#[derive(Debug, Clone)]
pub struct Tet4d3Element<T: Scalar> {
    vertices: [Point3<T>; 4],
}

impl<T: Scalar> Tet4d3Element<T> {
    pub fn from_vertices(vertices: [Point3<T>; 4]) -> Self {
        Self { vertices }
    }

    pub fn vertices(&self) -> &[Point3<T>; 4] {
        &self.vertices
    }
}

impl<T: Real> ReferenceFiniteElement<T> for Tet4d3Element<T> {
    type ReferenceDim = U3;

    fn num_nodes(&self) -> usize {
        4
    }

    #[replace_float_literals(T::from_f64(literal).expect("Literal must fit in T"))]
    fn populate_basis(&self, result: &mut [T], xi: &Point3<T>) {
        result.copy_from_slice(&[
            -0.5 * xi[0] - 0.5 * xi[1] - 0.5 * xi[2] - 0.5,
            0.5 * xi[0] + 0.5,
            0.5 * xi[1] + 0.5,
            0.5 * xi[2] + 0.5,
        ])
    }

    #[replace_float_literals(T::from_f64(literal).expect("Literal must fit in T"))]
    fn populate_basis_gradients(
        &self,
        mut result: MatrixViewMut<T, Self::ReferenceDim, Dyn>,
        _xi: &Point3<T>,
    ) {
        let gradients = Matrix3x4::from_columns(&[
            Vector3::new(-0.5, -0.5, -0.5),
            Vector3::new(0.5, 0.0, 0.0),
            Vector3::new(0.0, 0.5, 0.0),
            Vector3::new(0.0, 0.0, 0.5),
        ]);

        result.copy_from(&gradients);
    }
}

impl<T: Real> FiniteElement<T> for Tet4d3Element<T> {
    type GeometryDim = U3;

    #[allow(non_snake_case)]
    fn reference_jacobian(&self, xi: &Point3<T>) -> Matrix3<T> {
        let X = Matrix3x4::from_fn(|i, j| self.vertices[j][i]);
        let mut G = Matrix3x4::zeros();
        self.populate_basis_gradients(G.as_view_mut(), xi);
        X * G.transpose()
    }

    #[allow(non_snake_case)]
    fn map_reference_coords(&self, xi: &Point3<T>) -> Point3<T> {
        let X = Matrix3x4::from_fn(|i, j| self.vertices[j][i]);
        let mut N = Matrix1x4::zeros();
        self.populate_basis(N.as_mut_slice(), xi);
        Point3::from(X * N.transpose())
    }
}

impl Connectivity for Tet4d3Connectivity {
    fn num_faces(&self) -> usize {
        4
    }
}

impl<T: Real> ElementConnectivity<T> for Tet4d3Connectivity {
    type Element = Tet4d3Element<T>;
    type GeometryDim = U3;
    type ReferenceDim = U3;
}
