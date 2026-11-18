use nalgebra::{DimMin, DimName, RealField};

pub trait Real: RealField + Copy {}

impl<T: RealField + Copy> Real for T {}

pub trait SmallDim: DimName + DimMin<Self, Output = Self> {}

impl<D> SmallDim for D where D: DimName + DimMin<Self, Output = Self> {}
