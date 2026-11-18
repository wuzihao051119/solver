use nalgebra::RealField;

pub trait Real: RealField + Copy {}

impl<T: RealField + Copy> Real for T {}
