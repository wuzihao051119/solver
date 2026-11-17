use nalgebra::{ArrayStorage, Matrix, U1, U3, U8};

pub type Matrix1x8<T> = Matrix<T, U1, U8, ArrayStorage<T, 1, 8>>;

pub type Matrix3x8<T> = Matrix<T, U3, U8, ArrayStorage<T, 3, 8>>;

pub type Point<const D: usize> = [f64; D];

pub type Rule<const D: usize> = (Vec<f64>, Vec<Point<D>>);
