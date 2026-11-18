use std::path::Path;

use eyre::{Context, eyre};
use mshio::{MshFloatT, MshIntT};
use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, DimName, OPoint, RealField};
use num_traits::cast::ToPrimitive;

use crate::mesh::Mesh;
use crate::mesh::connectivity::{
    Hex8d3Connectivity, Quad4d2Connectivity, Tet4d3Connectivity, Tri3d2Connectivity,
};

pub fn load_msh_from_file<T, D, C, P>(file_path: P) -> eyre::Result<Mesh<T, D, C>>
where
    T: RealField,
    D: DimName,
    C: MshConnectivity,
    P: AsRef<Path>,
    DefaultAllocator: Allocator<D>,
{
    let msh_bytes = std::fs::read(file_path).wrap_err("failed to read file")?;
    load_msh_from_bytes(&msh_bytes).wrap_err("failed to load mesh from msh file")
}

pub fn load_msh_from_bytes<T, D, C>(bytes: &[u8]) -> eyre::Result<Mesh<T, D, C>>
where
    T: RealField,
    D: DimName,
    C: MshConnectivity,
    DefaultAllocator: Allocator<D>,
{
    let mut msh_file =
        mshio::parse_msh_bytes(bytes).map_err(|e| eyre!("failed to parse msh file: {}", e))?;

    let msh_nodes = msh_file
        .data
        .nodes
        .take()
        .ok_or(eyre!("MSH file does not contain nodes"))?;
    let msh_elements = msh_file
        .data
        .elements
        .take()
        .ok_or(eyre!("MSH file does not contain elements"))?;

    let mut vertices = Vec::new();
    let mut connectivity = Vec::new();

    if !msh_elements
        .element_blocks
        .iter()
        .any(|block| element_block_matches_connectivity::<C, _>(block))
    {
        return Err(eyre!(
            "MSH file does not contain an element block of the requested type (type {:?} of with reference/entity dim {})",
            C::msh_element_type(),
            C::reference_dim(),
        ));
    }

    for node_block in &msh_nodes.node_blocks {
        let block_vertices = vertices_from_node_block(node_block)?;
        vertices.extend(block_vertices);
    }

    if vertices.len()
        != msh_nodes
            .num_nodes
            .to_usize()
            .expect("failed to convert num_nodes to usize")
    {
        return Err(eyre!(
            "only {} vertices were read but msh file claims to contain {} nodes",
            vertices.len(),
            msh_nodes.num_nodes,
        ));
    }

    for element_block in &msh_elements.element_blocks {
        let block_connectivity = connectivity_from_element_block(element_block)?;
        connectivity.extend(block_connectivity);
    }

    Ok(Mesh::from_vertices_and_connectivity(vertices, connectivity))
}

fn vertices_from_node_block<T, D, F, I>(
    node_block: &mshio::NodeBlock<u64, I, F>,
) -> eyre::Result<Vec<OPoint<T, D>>>
where
    T: RealField,
    D: DimName,
    F: mshio::MshFloatT,
    I: mshio::MshIntT,
    DefaultAllocator: Allocator<D>,
{
    if node_block.node_tags.is_some() {
        return Err(eyre!(
            "node block tags are not consecutive in msh file (sparse tags are not supported)",
        ));
    }

    let mut vertices = Vec::with_capacity(node_block.nodes.len());

    for node in &node_block.nodes {
        vertices.push(point_from_msh_node(node)?);
    }

    Ok(vertices)
}

fn connectivity_from_element_block<C, I>(
    element_block: &mshio::ElementBlock<u64, I>,
) -> eyre::Result<Vec<C>>
where
    C: MshConnectivity,
    I: mshio::MshIntT,
{
    if element_block.element_tags.is_some() {
        return Err(eyre!(
            "element block tags are not consecutive in msh file (sparse tags are not supported)",
        ));
    }

    if !element_block_matches_connectivity::<C, _>(element_block) {
        Ok(Vec::new())
    } else {
        let mut connectivity = Vec::with_capacity(element_block.elements.len());
        let requested_nodes = C::msh_element_type()
            .nodes()
            .map_err(|_| eyre!("unimplemented element type requested"))?;

        for element in &element_block.elements {
            if element.nodes.len() < requested_nodes {
                return Err(eyre!("not enough nodes to initialize connectivity"));
            }
            connectivity.push(C::try_connectivity_from_msh_element(element)?);
        }

        Ok(connectivity)
    }
}

fn element_block_matches_connectivity<C, I>(element_block: &mshio::ElementBlock<u64, I>) -> bool
where
    I: MshIntT,
    C: MshConnectivity,
{
    element_block.element_type == C::msh_element_type()
        && element_block
            .entity_dim
            .to_usize()
            .expect("failed to convert element block dimension to usize")
            == C::reference_dim()
}

macro_rules! f_to_t {
    ($component:expr) => {
        T::from_f64(
            $component
                .to_f64()
                .ok_or_else(|| eyre!("failed to convert coordinate to f64"))?,
        )
        .ok_or_else(|| {
            eyre!("failed to convert node coordinate from f64 to target mesh real type")
        })?
    };
}

fn point_from_msh_node<T, D, F>(node: &mshio::Node<F>) -> eyre::Result<OPoint<T, D>>
where
    T: RealField,
    D: DimName,
    F: MshFloatT,
    DefaultAllocator: Allocator<D>,
{
    let mut point = OPoint::origin();
    point[0] = f_to_t!(node.x);
    if D::dim() > 1 {
        point[1] = f_to_t!(node.y);
    }
    if D::dim() > 2 {
        point[2] = f_to_t!(node.z);
    }

    Ok(point)
}

pub trait MshConnectivity
where
    Self: Sized,
{
    fn msh_element_type() -> mshio::ElementType;
    fn reference_dim() -> usize;
    fn try_connectivity_from_msh_element(element: &mshio::Element<u64>) -> eyre::Result<Self>;
}

macro_rules! impl_msh_connectivity {
    ($connectivity:ident, $msh_type:ident, num_nodes = $num_nodes:literal) => {
        impl MshConnectivity for $connectivity {
            fn msh_element_type() -> mshio::ElementType {
                mshio::ElementType::$msh_type
            }

            fn reference_dim() -> usize {
                use crate::element::ElementConnectivity;
                <$connectivity as ElementConnectivity<f64>>::ReferenceDim::dim()
            }

            fn try_connectivity_from_msh_element(
                element: &mshio::Element<u64>,
            ) -> eyre::Result<Self> {
                assert_eq!(
                    element.nodes.len(),
                    $num_nodes,
                    "number of msh element nodes have to match with connectivity",
                );
                let mut nodes = [0; $num_nodes];
                for i in 0..$num_nodes {
                    nodes[i] = element.nodes[i] as usize - 1;
                }
                Ok(Self(nodes))
            }
        }
    };
}

impl_msh_connectivity!(Tri3d2Connectivity, Tri3, num_nodes = 3);
impl_msh_connectivity!(Quad4d2Connectivity, Qua4, num_nodes = 4);
impl_msh_connectivity!(Tet4d3Connectivity, Tet4, num_nodes = 4);
impl_msh_connectivity!(Hex8d3Connectivity, Hex8, num_nodes = 8);
