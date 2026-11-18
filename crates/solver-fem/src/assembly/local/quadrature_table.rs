use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, DimName, OPoint, Scalar};

pub trait QuadratureTable<T, GeometryDim>
where
    T: Scalar,
    GeometryDim: DimName,
    DefaultAllocator: Allocator<GeometryDim>,
{
    fn element_quadrature_size(&self, element_index: usize) -> usize;

    fn populate_element_quadrature(
        &self,
        element_index: usize,
        points: &mut [OPoint<T, GeometryDim>],
        weights: &mut [T],
    );
}

#[derive(Debug, Clone)]
pub struct UniformQuadratureTable<T, GeometryDim>
where
    T: Scalar,
    GeometryDim: DimName,
    DefaultAllocator: Allocator<GeometryDim>,
{
    points: Vec<OPoint<T, GeometryDim>>,
    weights: Vec<T>,
}

impl<T, GeometryDim> UniformQuadratureTable<T, GeometryDim>
where
    T: Scalar,
    GeometryDim: DimName,
    DefaultAllocator: Allocator<GeometryDim>,
{
    pub fn from_points_and_weights(points: Vec<OPoint<T, GeometryDim>>, weights: Vec<T>) -> Self {
        let msg = "Points, weights and data must have the same length.";
        assert_eq!(points.len(), weights.len(), "{}", msg);
        Self { points, weights }
    }
}

impl<T, GeometryDim> QuadratureTable<T, GeometryDim> for UniformQuadratureTable<T, GeometryDim>
where
    T: Scalar,
    GeometryDim: DimName,
    DefaultAllocator: Allocator<GeometryDim>,
{
    fn element_quadrature_size(&self, _element_index: usize) -> usize {
        self.points.len()
    }

    fn populate_element_quadrature(
        &self,
        _element_index: usize,
        points: &mut [OPoint<T, GeometryDim>],
        weights: &mut [T],
    ) {
        assert_eq!(points.len(), self.points.len());
        assert_eq!(weights.len(), self.weights.len());
        points.clone_from_slice(&self.points);
        weights.clone_from_slice(&self.weights);
    }
}
