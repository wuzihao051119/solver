use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, DimName, OPoint, Scalar, U2, U3};

use crate::mesh::connectivity::{
    Hex8d3Connectivity, Quad4d2Connectivity, Tet4d3Connectivity, Tri3d2Connectivity,
};

pub mod connectivity;

#[derive(Debug, Clone)]
pub struct Mesh<T, D, Connectivity>
where
    T: Scalar,
    D: DimName,
    DefaultAllocator: Allocator<D>,
{
    vertices: Vec<OPoint<T, D>>,
    connectivity: Vec<Connectivity>,
}

pub type Mesh2d<T, Connectivity> = Mesh<T, U2, Connectivity>;
pub type Mesh3d<T, Connectivity> = Mesh<T, U3, Connectivity>;

pub type TriMesh2d<T> = Mesh2d<T, Tri3d2Connectivity>;
pub type QuadMesh2d<T> = Mesh2d<T, Quad4d2Connectivity>;
pub type TetMesh3d<T> = Mesh3d<T, Tet4d3Connectivity>;
pub type HexMesh3d<T> = Mesh3d<T, Hex8d3Connectivity>;

impl<T, D, Connectivity> Mesh<T, D, Connectivity>
where
    T: Scalar,
    D: DimName,
    DefaultAllocator: Allocator<D>,
{
    pub fn vertices(&self) -> &[OPoint<T, D>] {
        &self.vertices
    }

    pub fn connectivity(&self) -> &[Connectivity] {
        &self.connectivity
    }

    pub fn from_vertices_and_connectivity(
        vertices: Vec<OPoint<T, D>>,
        connectivity: Vec<Connectivity>,
    ) -> Self {
        Self {
            vertices,
            connectivity,
        }
    }
}
