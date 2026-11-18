pub trait Connectivity {
    fn num_faces(&self) -> usize;
    fn vertex_indices(&self) -> &[usize];
}

#[derive(Debug, Clone)]
pub struct Segment2d1Connectivity(pub [usize; 2]);

#[derive(Debug, Clone)]
pub struct Tri3d2Connectivity(pub [usize; 3]);

#[derive(Debug, Clone)]
pub struct Quad4d2Connectivity(pub [usize; 4]);

#[derive(Debug, Clone)]
pub struct Tet4d3Connectivity(pub [usize; 4]);

#[derive(Debug, Clone)]
pub struct Hex8d3Connectivity(pub [usize; 8]);
