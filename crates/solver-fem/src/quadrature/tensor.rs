use crate::traits::Real;
use nalgebra::{Point2, Point3, convert};

use crate::quadrature::gauss::gauss;
use crate::types::{QuadraturePair2d, QuadraturePair3d, Rule};

pub fn quadrilateral_gauss<T: Real>(n: usize) -> QuadraturePair2d<T> {
    let (weights, points) = quadrilateral_gauss_(n);
    convert_quadrature_rule_from_2d_f64((weights, points))
}

pub fn hexahedron_gauss_gauss<T: Real>(n: usize) -> QuadraturePair3d<T> {
    let (weights, points) = hexahedron_gauss_(n);
    convert_quadrature_rule_from_3d_f64((weights, points))
}

fn quadrilateral_gauss_(n: usize) -> Rule<2> {
    let (weights1d, points1d) = gauss(n);
    let mut weights2d = Vec::with_capacity(n * n);
    let mut points2d = Vec::with_capacity(n * n);

    let rule1d_iter = || weights1d.iter().zip(&points1d);

    for (&wx, &[x]) in rule1d_iter() {
        for (&wy, &[y]) in rule1d_iter() {
            let w = wx * wy;
            weights2d.push(w);
            points2d.push([x, y]);
        }
    }

    (weights2d, points2d)
}

fn hexahedron_gauss_(n: usize) -> Rule<3> {
    let (weights1d, points1d) = gauss(n);
    let mut weights3d = Vec::with_capacity(n * n * n);
    let mut points3d = Vec::with_capacity(n * n * n);

    let rule1d_iter = || weights1d.iter().zip(&points1d);

    for (&wx, &[x]) in rule1d_iter() {
        for (&wy, &[y]) in rule1d_iter() {
            for (&wz, &[z]) in rule1d_iter() {
                let w = wx * wy * wz;
                weights3d.push(w);
                points3d.push([x, y, z]);
            }
        }
    }

    (weights3d, points3d)
}

fn convert_quadrature_rule_from_2d_f64<T: Real>(quadrature: Rule<2>) -> QuadraturePair2d<T> {
    let (weights, points) = quadrature;
    let weights = weights.into_iter().map(convert).collect();
    let points = points.into_iter().map(Point2::from).map(convert).collect();
    (weights, points)
}

fn convert_quadrature_rule_from_3d_f64<T: Real>(quadrature: Rule<3>) -> QuadraturePair3d<T> {
    let (weights, points) = quadrature;
    let weights = weights.into_iter().map(convert).collect();
    let points = points.into_iter().map(Point3::from).map(convert).collect();
    (weights, points)
}
