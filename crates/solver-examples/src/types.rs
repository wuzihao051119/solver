use nalgebra::{Dyn, Matrix, U1, VecStorage};

pub type Matrix9x1<T> = Matrix<T, Dyn, U1, VecStorage<T, Dyn, U1>>;
