use eyre::eyre;
use itertools::izip;
use nalgebra::allocator::Allocator;
use nalgebra::{
    DMatrixViewMut, DVector, DVectorView, DefaultAllocator, Dim, DimName, Dyn, MatrixView,
    MatrixViewMut, OMatrix, OPoint, Scalar, U1,
};

use crate::assembly::global::gather_global_to_local;
use crate::assembly::local::quadrature_table::QuadratureTable;
use crate::assembly::local::{ElementConnectivityAssembler, ElementMatrixAssembler};
use crate::assembly::operators::Symmetry;
use crate::assembly::operators::{EllipticContraction, Operator};
use crate::assembly::util::{clone_upper_to_lower, reshape_to_slice};
use crate::element::VolumetricFiniteElement;
use crate::space::{ElementInSpace, VolumetricFiniteElementSpace};
use crate::traits::Real;

pub struct ElementEllipticAssemblerBuilder<Space, Op, QTable, U> {
    space: Space,
    op: Op,
    qtable: QTable,
    u: U,
}

#[allow(clippy::new_without_default)]
impl ElementEllipticAssemblerBuilder<(), (), (), ()> {
    pub fn new() -> Self {
        Self {
            space: (),
            op: (),
            qtable: (),
            u: (),
        }
    }
}

impl<Op, QTable, U> ElementEllipticAssemblerBuilder<(), Op, QTable, U> {
    pub fn with_finite_element_space<Space>(
        self,
        space: &Space,
    ) -> ElementEllipticAssemblerBuilder<&Space, Op, QTable, U> {
        ElementEllipticAssemblerBuilder {
            space,
            op: self.op,
            qtable: self.qtable,
            u: self.u,
        }
    }
}

impl<Space, QTable, U> ElementEllipticAssemblerBuilder<Space, (), QTable, U> {
    pub fn with_operator<Op>(
        self,
        op: &Op,
    ) -> ElementEllipticAssemblerBuilder<Space, &Op, QTable, U> {
        ElementEllipticAssemblerBuilder {
            space: self.space,
            op,
            qtable: self.qtable,
            u: self.u,
        }
    }
}

impl<Space, Op, U> ElementEllipticAssemblerBuilder<Space, Op, (), U> {
    pub fn with_quadrature_table<QTable>(
        self,
        qtable: &QTable,
    ) -> ElementEllipticAssemblerBuilder<Space, Op, &QTable, U> {
        ElementEllipticAssemblerBuilder {
            space: self.space,
            op: self.op,
            qtable,
            u: self.u,
        }
    }
}

impl<Space, Op, QTable> ElementEllipticAssemblerBuilder<Space, Op, QTable, ()> {
    pub fn with_u<'a, T: Scalar>(
        self,
        u: impl Into<DVectorView<'a, T>>,
    ) -> ElementEllipticAssemblerBuilder<Space, Op, QTable, DVectorView<'a, T>> {
        ElementEllipticAssemblerBuilder {
            space: self.space,
            op: self.op,
            qtable: self.qtable,
            u: u.into(),
        }
    }
}

impl<'a, T: Scalar, Space, Op, QTable>
    ElementEllipticAssemblerBuilder<&'a Space, &'a Op, &'a QTable, DVectorView<'a, T>>
{
    pub fn build(self) -> ElementEllipticAssembler<'a, T, Space, Op, QTable> {
        ElementEllipticAssembler {
            space: self.space,
            op: self.op,
            qtable: self.qtable,
            u: self.u,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ElementEllipticAssembler<'a, T: Scalar, Space, Op, QTable> {
    space: &'a Space,
    op: &'a Op,
    qtable: &'a QTable,
    u: DVectorView<'a, T>,
}

impl<'a, T, Space, Op, QTable> ElementConnectivityAssembler
    for ElementEllipticAssembler<'a, T, Space, Op, QTable>
where
    T: Scalar,
    Space: VolumetricFiniteElementSpace<T>,
    Op: Operator,
    DefaultAllocator:
        Allocator<Space::ReferenceDim> + Allocator<Space::ReferenceDim, Space::ReferenceDim>,
{
    fn solution_dim(&self) -> usize {
        Op::SolutionDim::dim()
    }

    fn num_elements(&self) -> usize {
        self.space.num_elements()
    }

    fn num_nodes(&self) -> usize {
        self.space.num_nodes()
    }

    fn element_node_count(&self, element_index: usize) -> usize {
        self.space.element_node_count(element_index)
    }

    fn populate_element_nodes(&self, output: &mut [usize], element_index: usize) {
        self.space.populate_element_nodes(output, element_index);
    }
}

impl<'a, T, Space, Op, QTable> ElementMatrixAssembler<T>
    for ElementEllipticAssembler<'a, T, Space, Op, QTable>
where
    T: Real,
    Space: VolumetricFiniteElementSpace<T>,
    Op: EllipticContraction<T, Space::GeometryDim>,
    QTable: QuadratureTable<T, Space::ReferenceDim>,
    DefaultAllocator: Allocator<Space::ReferenceDim>
        + Allocator<Space::ReferenceDim, Space::ReferenceDim>
        + Allocator<Space::ReferenceDim, Op::SolutionDim>
        + Allocator<Op::SolutionDim, Op::SolutionDim>,
{
    fn assemble_element_matrix_into(
        &self,
        element_index: usize,
        output: DMatrixViewMut<T>,
    ) -> eyre::Result<()> {
        let s = self.solution_dim();
        let n = self.element_node_count(element_index);
        assert_eq!(output.nrows(), s * n, "Output matrix dimension mismatch");
        assert_eq!(output.ncols(), s * n, "Output matrix dimension mismatch");

        let element = ElementInSpace::from_space_and_element_index(self.space, element_index);

        let mut node_indices = vec![usize::MAX; n];
        self.populate_element_nodes(&mut node_indices, element_index);

        let mut u_element = DVector::zeros(s * n);
        gather_global_to_local(self.u, &mut u_element, &node_indices, s);

        let quadrature_size = self.qtable.element_quadrature_size(element_index);
        let mut weights = vec![T::zero(); quadrature_size];
        let mut points = vec![OPoint::origin(); quadrature_size];
        self.qtable
            .populate_element_quadrature(element_index, &mut points, &mut weights);

        assemble_element_elliptic_matrix(
            output,
            &element,
            self.op,
            DVectorView::from(&u_element),
            &weights,
            &points,
        )
    }
}

#[allow(non_snake_case)]
fn compute_volume_u_grad<'a, T, GeometryDim, SolutionDim>(
    jacobian_inv_t: &OMatrix<T, GeometryDim, GeometryDim>,
    phi_grad_ref: impl Into<MatrixView<'a, T, GeometryDim, Dyn>>,
    u: impl Into<MatrixView<'a, T, SolutionDim, Dyn>>,
) -> OMatrix<T, GeometryDim, SolutionDim>
where
    T: Real,
    GeometryDim: DimName,
    SolutionDim: DimName,
    DefaultAllocator: Allocator<GeometryDim, GeometryDim> + Allocator<GeometryDim, SolutionDim>,
{
    let phi_grad_ref = phi_grad_ref.into();
    let u = u.into();
    let mut u_grad = OMatrix::<_, GeometryDim, SolutionDim>::zeros();
    for (phi_I_grad_ref, u_I) in phi_grad_ref.column_iter().zip(u.column_iter()) {
        u_grad.ger(T::one(), &phi_I_grad_ref, &u_I, T::one());
    }
    jacobian_inv_t * u_grad
}

fn assemble_element_elliptic_matrix<T, Element, Contraction>(
    mut output: DMatrixViewMut<T>,
    element: &Element,
    operator: &Contraction,
    u_element: DVectorView<T>,
    quadrature_weights: &[T],
    quadrature_points: &[OPoint<T, Element::ReferenceDim>],
) -> eyre::Result<()>
where
    T: Real,
    Element: VolumetricFiniteElement<T>,
    Contraction: EllipticContraction<T, Element::GeometryDim>,
    DefaultAllocator: Allocator<Element::GeometryDim>
        + Allocator<Element::GeometryDim, Element::GeometryDim>
        + Allocator<Element::GeometryDim, Contraction::SolutionDim>
        + Allocator<Contraction::SolutionDim, Contraction::SolutionDim>,
{
    assert_eq!(quadrature_weights.len(), quadrature_points.len());

    let d = Element::GeometryDim::dim();
    let s = Contraction::SolutionDim::name();
    let n = element.num_nodes();
    assert_eq!(
        u_element.len(),
        s.value() * n,
        "Local element dofs (u_element) dimension mismatch",
    );
    assert_eq!(
        output.nrows(),
        s.value() * n,
        "Output matrix dimension mismatch"
    );
    assert_eq!(
        output.ncols(),
        s.value() * n,
        "Output matrix dimension mismatch"
    );

    output.fill(T::zero());

    let mut phi_grad = OMatrix::<_, Element::ReferenceDim, Dyn>::zeros(n);

    let quadrature_iter = izip!(quadrature_weights, quadrature_points);

    for (&weight, point) in quadrature_iter {
        let j = element.reference_jacobian(point);
        let j_det = j.determinant();
        let j_inv = j
            .try_inverse()
            .ok_or_else(|| eyre!("Singular element Jacobian encountered"))?;
        let j_inv_t = j_inv.transpose();

        element.populate_basis_gradients(MatrixViewMut::from(&mut phi_grad), point);

        let u_element = reshape_to_slice(&u_element, (s, Dyn(n)));
        let u_grad = compute_volume_u_grad(&j_inv_t, &phi_grad, u_element);

        for mut phi_grad in phi_grad.column_iter_mut() {
            let new_phi_grad = &j_inv_t * &phi_grad;
            phi_grad.copy_from(&new_phi_grad);
        }

        let scale = weight * j_det.abs();
        let phi_grad = reshape_to_slice(&phi_grad, (Dyn(d * n), U1::name()));
        operator.accumulate_contractions_into(
            DMatrixViewMut::from(&mut output),
            scale,
            &u_grad,
            phi_grad,
            phi_grad,
        );
    }

    if matches!(operator.symmetry(), Symmetry::Symmetric) {
        clone_upper_to_lower(&mut output);
    }

    Ok(())
}
