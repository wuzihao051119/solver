use std::cell::RefCell;
use std::ops::AddAssign;

use nalgebra::{DVector, DVectorView, DVectorViewMut, Scalar};

use crate::assembly::local::ElementVectorAssembler;
use crate::traits::Real;

#[derive(Debug, Clone)]
struct VectorAssemblerWorkspace<T: Scalar> {
    vector: DVector<T>,
    nodes: Vec<usize>,
}

impl<T: Real> Default for VectorAssemblerWorkspace<T> {
    fn default() -> Self {
        Self {
            vector: DVector::zeros(0),
            nodes: vec![],
        }
    }
}

#[derive(Debug, Clone)]
pub struct VectorAssembler<T: Scalar> {
    workspace: RefCell<VectorAssemblerWorkspace<T>>,
}

impl<T: Real> Default for VectorAssembler<T> {
    fn default() -> Self {
        Self {
            workspace: RefCell::new(VectorAssemblerWorkspace::default()),
        }
    }
}

impl<T: Real> VectorAssembler<T> {
    pub fn assemble_vector(
        &self,
        element_assembler: &(impl ElementVectorAssembler<T> + Sync),
    ) -> eyre::Result<DVector<T>> {
        let n = element_assembler.num_nodes();
        let mut result = DVector::zeros(element_assembler.solution_dim() * n);
        self.assemble_vector_into(&mut result, element_assembler)?;
        Ok(result)
    }

    pub fn assemble_vector_into<'a>(
        &self,
        output: impl Into<DVectorViewMut<'a, T>>,
        element_assembler: &impl ElementVectorAssembler<T>,
    ) -> eyre::Result<()> {
        let mut output = output.into();
        let num_elements = element_assembler.num_elements();
        let n = element_assembler.num_nodes();
        let s = element_assembler.solution_dim();
        assert_eq!(output.len(), s * n, "Output dimensions mismatch");

        let mut workspace = self.workspace.borrow_mut();

        for i in 0..num_elements {
            let element_node_count = element_assembler.element_node_count(i);
            workspace.nodes.resize(element_node_count, usize::MAX);
            workspace
                .vector
                .resize_vertically_mut(s * element_node_count, T::zero());
            element_assembler.populate_element_nodes(&mut workspace.nodes, i);
            element_assembler.assemble_element_vector_into(i, (&mut workspace.vector).into())?;
            add_local_to_global(&workspace.vector, &mut output, &workspace.nodes, s);
        }

        Ok(())
    }
}

pub fn add_local_to_global<'a, T: Real>(
    local: impl Into<DVectorView<'a, T>>,
    global: impl Into<DVectorViewMut<'a, T>>,
    indices: &[usize],
    solution_dim: usize,
) {
    add_local_to_global_(local.into(), global.into(), indices, solution_dim);
}

fn add_local_to_global_<'a, T: Real>(
    local: DVectorView<'a, T>,
    mut global: DVectorViewMut<'a, T>,
    indices: &[usize],
    solution_dim: usize,
) {
    assert_eq!(
        local.len(),
        indices.len() * solution_dim,
        "Size of local vector must be compatible with solution dim and index count",
    );
    let s = solution_dim;
    for (i_local, i_global) in indices.iter().enumerate() {
        global
            .index_mut((s * i_global..s * i_global + s, 0))
            .add_assign(&local.index((s * i_local..s * i_local + s, 0)));
    }
}
