use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, DimName, RealField};

pub trait Real: RealField + Copy {}

pub trait DimAllocator<D: DimName>: Allocator<D> {}

impl<D> DimAllocator<D> for DefaultAllocator
where
    D: DimName,
    DefaultAllocator: Allocator<D>,
{
}

pub trait BiDimAllocator<D1: DimName, D2: DimName>:
    DimAllocator<D1> + DimAllocator<D2> + Allocator<D1, D2>
{
}

impl<D1, D2> BiDimAllocator<D1, D2> for DefaultAllocator
where
    D1: DimName,
    D2: DimName,
    DefaultAllocator: DimAllocator<D1> + DimAllocator<D2> + Allocator<D1, D2>,
{
}
