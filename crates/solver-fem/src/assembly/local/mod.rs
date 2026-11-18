use nalgebra::{DMatrix, DMatrixViewMut, DVector, DVectorViewMut, Scalar};

use crate::traits::Real;

pub mod quadrature_table;
pub mod source;

pub trait ElementConnectivityAssembler {
    fn solution_dim(&self) -> usize;

    fn num_elements(&self) -> usize;

    fn num_nodes(&self) -> usize;

    fn element_node_count(&self, element_index: usize) -> usize;

    fn populate_element_nodes(&self, output: &mut [usize], element_index: usize);
}

pub trait ElementMatrixAssembler<T: Scalar>: ElementConnectivityAssembler {
    fn assemble_element_matrix_into(
        &self,
        element_index: usize,
        output: DMatrixViewMut<T>,
    ) -> eyre::Result<()>;

    fn assemble_element_matrix(&self, element_index: usize) -> eyre::Result<DMatrix<T>>
    where
        T: Real,
    {
        let ndof = self.solution_dim() * self.element_node_count(element_index);
        let mut output = DMatrix::zeros(ndof, ndof);
        self.assemble_element_matrix_into(element_index, DMatrixViewMut::from(&mut output))?;
        Ok(output)
    }
}

pub trait ElementVectorAssembler<T: Scalar>: ElementConnectivityAssembler {
    fn assemble_element_vector_into(
        &self,
        element_index: usize,
        output: DVectorViewMut<T>,
    ) -> eyre::Result<()>;

    fn assemble_element_vector(&self, element_index: usize) -> eyre::Result<DVector<T>>
    where
        T: Real,
    {
        let ndof = self.solution_dim() * self.element_node_count(element_index);
        let mut output = DVector::zeros(ndof);
        self.assemble_element_vector_into(element_index, DVectorViewMut::from(&mut output))?;
        Ok(output)
    }
}

pub trait ElementScalarAssembler<T: Scalar>: ElementConnectivityAssembler {
    fn assemble_element_scalar(&self, element_index: usize) -> eyre::Result<T>;
}
