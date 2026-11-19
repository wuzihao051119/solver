use nalgebra::{
    Dim, DimMul, DimName, DimProd, Matrix, MatrixView, Scalar, Storage, StorageMut, U1,
    ViewStorage,
};
use nalgebra::constraint::{DimEq, ShapeConstraint};

pub fn clone_upper_to_lower<T, R, C, S>(matrix: &mut Matrix<T, R, C, S>)
where
    T: Scalar,
    R: Dim,
    C: Dim,
    S: StorageMut<T, R, C>,
{
    for j in 0..matrix.ncols() {
        for i in (j + 1)..matrix.nrows() {
            matrix[(i, j)] = matrix[(j, i)].clone();
        }
    }
}

pub fn reshape_to_slice<'a, T, R, C, S, R2, C2>(
    matrix: &Matrix<T, R, C, S>,
    shape: (R2, C2),
) -> MatrixView<'a, T, R2, C2, U1, R2>
where
    T: Scalar,
    R: DimMul<C>,
    C: Dim,
    S: Storage<T, R, C>,
    R2: DimMul<C2>,
    C2: Dim,
    ShapeConstraint: DimEq<DimProd<R, C>, DimProd<R2, C2>>,
{
    let (r2, c2) = shape;
    assert_eq!(
        matrix.nrows() * matrix.ncols(),
        r2.value() * c2.value(),
        "Cannot reshape with different number of elements",
    );
    let strides = (U1::name(), r2);
    let storage = unsafe { ViewStorage::from_raw_parts(matrix.data.ptr(), shape, strides) };
    Matrix::from_data(storage)
}
