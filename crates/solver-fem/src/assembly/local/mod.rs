use nalgebra::{DMatrix, DMatrixViewMut, DVector, DVectorViewMut, Scalar};

use crate::traits::Real;

pub mod source;

pub trait ElementMatrixAssembler<T: Scalar> {
    fn assemble_element_matrix_into(
        &self,
        element_index: usize,
        output: DMatrixViewMut<T>,
    ) -> eyre::Result<()>;

    fn assemble_element_matrix(&self, element_index: usize) -> eyre::Result<DMatrix<T>>
    where
        T: Real,
    {
        let ndof = 3usize;
        let mut output = DMatrix::zeros(ndof, ndof);
        self.assemble_element_matrix_into(element_index, DMatrixViewMut::from(&mut output))?;
        Ok(output)
    }
}

pub trait ElementVectorAssembler<T: Scalar> {
    fn assemble_element_vector_into(
        &self,
        element_index: usize,
        output: DVectorViewMut<T>,
    ) -> eyre::Result<()>;

    fn assemble_element_vector(&self, element_index: usize) -> eyre::Result<DVector<T>>
    where
        T: Real,
    {
        let ndof = 3usize;
        let mut output = DVector::zeros(ndof);
        self.assemble_element_vector_into(element_index, DVectorViewMut::from(&mut output))?;
        Ok(output)
    }
}

pub trait ElementScalarAssembler<T: Scalar> {
    fn assemble_element_scalar(&self, element_index: usize) -> eyre::Result<T>;
}
