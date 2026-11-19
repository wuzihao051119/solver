use std::cell::RefCell;
use std::ops::AddAssign;

use nalgebra::{
    DMatrix, DMatrixViewMut, DVector, DVectorView, DVectorViewMut, Dyn, Matrix, Scalar, Storage, U1,
};
use nalgebra_sparse::CsrMatrix;
use nalgebra_sparse::pattern::SparsityPattern;
use rustc_hash::FxHashSet;

use crate::assembly::local::{
    ElementConnectivityAssembler, ElementMatrixAssembler, ElementVectorAssembler,
};
use crate::traits::Real;

#[derive(Debug, Clone)]
struct CsrAssemblerWorkspace<T: Scalar> {
    element_matrix: DMatrix<T>,
    element_global_nodes: Vec<usize>,
    connectivity_permutation: Vec<usize>,
}

impl<T: Scalar> Default for CsrAssemblerWorkspace<T> {
    fn default() -> Self {
        Self {
            element_matrix: DMatrix::from_row_slice(0, 0, &[]),
            element_global_nodes: Vec::new(),
            connectivity_permutation: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CsrAssembler<T: Scalar> {
    workspace: RefCell<CsrAssemblerWorkspace<T>>,
}

impl<T: Scalar> Default for CsrAssembler<T> {
    fn default() -> Self {
        Self {
            workspace: RefCell::new(CsrAssemblerWorkspace::default()),
        }
    }
}

impl<T: Scalar> CsrAssembler<T> {
    pub fn assemble_pattern(
        &self,
        element_assembler: &impl ElementConnectivityAssembler,
    ) -> SparsityPattern {
        let sdim = element_assembler.solution_dim();
        let num_nodes = element_assembler.num_nodes();
        let num_rows = sdim * num_nodes;
        let mut node_sets: Vec<FxHashSet<usize>> = vec![FxHashSet::default(); num_nodes];
        let mut element_global_nodes = Vec::new();
        for i in 0..element_assembler.num_elements() {
            let element_node_count = element_assembler.element_node_count(i);
            element_global_nodes.resize(element_node_count, usize::MAX);
            element_assembler.populate_element_nodes(&mut element_global_nodes, i);

            for &node_i in &element_global_nodes {
                for &node_j in &element_global_nodes {
                    node_sets[node_i].insert(node_j);
                }
            }
        }

        let mut offsets = Vec::with_capacity(num_rows + 1);
        offsets.push(0);
        let mut current_offset = 0;
        for node_set in &node_sets {
            for _ in 0..sdim {
                let count = sdim * node_set.len();
                offsets.push(current_offset + count);
                current_offset += count;
            }
        }
        assert_eq!(offsets.len(), num_rows + 1);

        let mut col_indices = Vec::with_capacity(*offsets.last().unwrap());
        let mut node_buffer: Vec<usize> = Vec::new();
        for node_set in &node_sets {
            node_buffer.clear();
            node_buffer.extend(node_set);
            node_buffer.sort_unstable();
            for _ in 0..sdim {
                for node_j in &node_buffer {
                    for j in 0..sdim {
                        let col_idx = sdim * node_j + j;
                        col_indices.push(col_idx);
                    }
                }
            }
        }

        assert_eq!(*offsets.last().unwrap(), col_indices.len());

        debug_assert!(
            SparsityPattern::try_from_offsets_and_indices(
                num_rows,
                num_rows,
                offsets.clone(),
                col_indices.clone()
            )
            .is_ok(),
            "Internal error: constructed sparsity pattern is not valid. This is a bug!",
        );
        unsafe {
            SparsityPattern::from_offset_and_indices_unchecked(
                num_rows,
                num_rows,
                offsets,
                col_indices,
            )
        }
    }
}

impl<T: Real> CsrAssembler<T> {
    pub fn assemble(
        &self,
        element_assembler: &impl ElementMatrixAssembler<T>,
    ) -> eyre::Result<CsrMatrix<T>> {
        let pattern = self.assemble_pattern(element_assembler);
        let initial_matrix_values = vec![T::zero(); pattern.nnz()];
        let mut matrix = CsrMatrix::try_from_pattern_and_values(pattern, initial_matrix_values)
            .expect("CSR data must be valid by definition");
        self.assemble_into_csr(&mut matrix, element_assembler)?;
        Ok(matrix)
    }

    pub fn assemble_into_csr(
        &self,
        csr: &mut CsrMatrix<T>,
        element_assembler: &impl ElementMatrixAssembler<T>,
    ) -> eyre::Result<()> {
        let ws = &mut *self.workspace.borrow_mut();
        let element_matrix = &mut ws.element_matrix;
        let element_global_nodes = &mut ws.element_global_nodes;
        let connectivity_permutation = &mut ws.connectivity_permutation;

        let sdim = element_assembler.solution_dim();

        for i in 0..element_assembler.num_elements() {
            let element_node_count = element_assembler.element_node_count(i);
            let element_matrix_dim = sdim * element_node_count;

            element_global_nodes.resize(element_node_count, 0);
            element_matrix.resize_mut(element_matrix_dim, element_matrix_dim, T::zero());

            let matrix_slice = DMatrixViewMut::from(&mut *element_matrix);
            element_assembler.assemble_element_matrix_into(i, matrix_slice)?;
            element_assembler.populate_element_nodes(element_global_nodes, i);

            connectivity_permutation.clear();
            connectivity_permutation.extend(0..element_node_count);
            connectivity_permutation.sort_unstable_by_key(|i| element_global_nodes[*i]);

            for (local_node_idx, global_node_idx) in element_global_nodes.iter().enumerate() {
                for i in 0..sdim {
                    let local_row_index = sdim * local_node_idx + i;
                    let global_row_index = sdim * global_node_idx + i;
                    let mut csr_row = csr.row_mut(global_row_index);
                    let (cols, values) = csr_row.cols_and_values_mut();

                    let a_row = element_matrix.row(local_row_index);
                    add_element_row_to_csr_row(
                        values,
                        cols,
                        element_global_nodes,
                        connectivity_permutation,
                        sdim,
                        &a_row,
                    );
                }
            }
        }

        Ok(())
    }
}

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

pub fn gather_global_to_local<'a, T: Scalar>(
    global: impl Into<DVectorView<'a, T>>,
    local: impl Into<DVectorViewMut<'a, T>>,
    indices: &[usize],
    solution_dim: usize,
) {
    gather_global_to_local_(global.into(), local.into(), indices, solution_dim);
}

fn gather_global_to_local_<T: Scalar>(
    global: DVectorView<T>,
    mut local: DVectorViewMut<T>,
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
        local
            .index_mut((s * i_local..s * i_local + s, 0))
            .copy_from(&global.index((s * i_global..s * i_global + s, 0)));
    }
}

fn add_element_row_to_csr_row<T, S>(
    row_values: &mut [T],
    row_col_indices: &[usize],
    node_connectivity: &[usize],
    sorted_permutation: &[usize],
    dim: usize,
    local_row: &Matrix<T, U1, Dyn, S>,
) where
    T: Real,
    S: Storage<T, U1, Dyn>,
{
    assert_eq!(node_connectivity.len(), sorted_permutation.len());
    assert_eq!(node_connectivity.len() * dim, local_row.ncols());
    assert!(dim >= 1);

    let (col_indices, values) = (row_col_indices, row_values);
    let mut csr_col_idx_iter = col_indices.iter().copied().enumerate();

    for &node_local_idx in sorted_permutation {
        let node_global_idx = node_connectivity[node_local_idx];

        for i in 0..dim {
            let local_col_idx = dim * node_local_idx + i;
            let global_col_idx = dim * node_global_idx + i;

            let (local_csr_col_idx, _) = csr_col_idx_iter
                .find(|(_, csr_col_idx)| *csr_col_idx == global_col_idx)
                .expect("Could not find column index associated with node in CSR row");
            values[local_csr_col_idx] += local_row[local_col_idx];
        }
    }
}

pub fn apply_homogeneous_dirichlet_bc_csr<T: Real>(
    matrix: &mut CsrMatrix<T>,
    nodes: &[usize],
    solution_dim: usize,
) {
    let d = solution_dim;

    let scale = matrix
        .triplet_iter()
        .filter(|(i, j, _)| i == j)
        .map(|(_, _, v)| v)
        .skip_while(|&x| x == &T::zero())
        .map(|x| x.abs())
        .next()
        .unwrap_or(T::one());

    let mut dirichlet_membership = vec![false; matrix.nrows()];
    let mut rows_to_visit = vec![false; matrix.nrows()];

    for &node in nodes {
        for i in 0..d {
            let row_idx = d * node + i;
            dirichlet_membership[row_idx] = true;
            let mut row = matrix.row_mut(row_idx);
            let (cols, values) = row.cols_and_values_mut();

            for (&col_idx, val) in cols.iter().zip(values) {
                if col_idx == row_idx {
                    *val = scale;
                } else {
                    *val = T::zero();
                    rows_to_visit[col_idx] = true;
                }
            }
        }
    }

    let row_visit_iter = rows_to_visit
        .iter()
        .enumerate()
        .filter_map(|(index, &should_visit)| if should_visit { Some(index) } else { None });
    for row_index in row_visit_iter {
        let row_is_dirichlet = dirichlet_membership[row_index];
        if !row_is_dirichlet {
            let mut row = matrix.row_mut(row_index);
            let (cols, values) = row.cols_and_values_mut();
            for (local_idx, &global_idx) in cols.iter().enumerate() {
                let col_is_dirichlet = dirichlet_membership[global_idx];
                if col_is_dirichlet {
                    values[local_idx] = T::zero();
                }
            }
        }
    }
}

pub fn apply_homogeneous_dirichlet_bc_rhs<'a, T: Real>(
    rhs: impl Into<DVectorViewMut<'a, T>>,
    nodes: &[usize],
    solution_dim: usize,
) {
    let mut rhs = rhs.into();
    let d = solution_dim;

    for node in nodes {
        for i in 0..d {
            let idx = d * node + i;
            *rhs.index_mut(idx) = T::zero();
        }
    }
}
