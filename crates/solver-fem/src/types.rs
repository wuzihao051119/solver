use nalgebra::{ArrayStorage, Matrix, OPoint, U1, U2, U3, U8};

pub type Matrix1x8<T> = Matrix<T, U1, U8, ArrayStorage<T, 1, 8>>;

pub type Matrix3x8<T> = Matrix<T, U3, U8, ArrayStorage<T, 3, 8>>;

pub type Point<const D: usize> = [f64; D];

pub type Rule<const D: usize> = (Vec<f64>, Vec<Point<D>>);

pub type QuadraturePair<T, D> = (Vec<f64>, Vec<OPoint<T, D>>);

pub type QuadraturePair1d<T> = QuadraturePair<T, U1>;
pub type QuadraturePair2d<T> = QuadraturePair<T, U2>;
pub type QuadraturePair3d<T> = QuadraturePair<T, U3>;
