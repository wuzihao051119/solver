use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, DimName, OPoint, Scalar};

#[derive(Debug, Clone)]
pub struct QuadratureBuffer<T, D>
where
    T: Scalar,
    D: DimName,
    DefaultAllocator: Allocator<D>,
{
    quad_weights: Vec<T>,
    quad_points: Vec<OPoint<T, D>>,
}

impl<T, D> QuadratureBuffer<T, D>
where
    T: Scalar,
    D: DimName,
    DefaultAllocator: Allocator<D>,
{
    pub fn weights(&self) -> &[T] {
        &self.quad_weights
    }

    pub fn points(&self) -> &[OPoint<T, D>] {
        &self.quad_points
    }
}

impl<T, D> Default for QuadratureBuffer<T, D>
where
    T: Scalar,
    D: DimName,
    DefaultAllocator: Allocator<D>,
{
    fn default() -> Self {
        Self {
            quad_weights: Vec::new(),
            quad_points: Vec::new(),
        }
    }
}
