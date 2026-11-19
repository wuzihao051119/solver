use eyre::eyre;
use nalgebra::{DMatrix, DVector, Point2, U1, U2, Vector1};
use nalgebra_sparse::CsrMatrix;
use solver_fem::assembly::global::{
    CsrAssembler, VectorAssembler, apply_homogeneous_dirichlet_bc_csr,
    apply_homogeneous_dirichlet_bc_rhs,
};
use solver_fem::assembly::local::elliptic::ElementEllipticAssemblerBuilder;
use solver_fem::assembly::local::quadrature_table::UniformQuadratureTable;
use solver_fem::assembly::local::source::ElementSourceAssemblerBuilder;
use solver_fem::assembly::operators::laplace::LaplaceOperator;
use solver_fem::assembly::operators::{Operator, SourceFunction};
use solver_fem::io::msh::load_msh_from_file;
use solver_fem::mesh::QuadMesh2d;
use solver_fem::quadrature::tensor::quadrilateral_gauss;

pub fn poisson2d() -> eyre::Result<()> {
    let mesh = load_msh_from_file("assets/poisson2d20x20.msh")?;

    let (a, b) = assemble_linear_system(&mesh)?;
    let u = solve_linear_system(&a, &b)?;

    println!("{}", u);

    Ok(())
}

fn assemble_linear_system(mesh: &QuadMesh2d<f64>) -> eyre::Result<(CsrMatrix<f64>, DVector<f64>)> {
    let (weights, points) = quadrilateral_gauss(2);
    let quadrature = UniformQuadratureTable::from_points_and_weights(points, weights);

    let u = DVector::<f64>::zeros(mesh.vertices().len());

    let matrix_assembler = CsrAssembler::default();
    let vector_assembler = VectorAssembler::<f64>::default();

    let laplace_assembler = ElementEllipticAssemblerBuilder::new()
        .with_finite_element_space(mesh)
        .with_operator(&LaplaceOperator)
        .with_quadrature_table(&quadrature)
        .with_u(&u)
        .build();

    let mut a_global = matrix_assembler.assemble(&laplace_assembler)?;

    let source_assembler = ElementSourceAssemblerBuilder::new()
        .with_finite_element_space(mesh)
        .with_quadrature_table(&quadrature)
        .with_source(&PoissonProblemSourceFunction)
        .build();

    let mut b_global = vector_assembler.assemble_vector(&source_assembler)?;
    
    let dirichlet_nodes: Vec<_> = mesh
        .vertices()
        .iter()
        .enumerate()
        .filter(|(_, v)| {
            (v.y + 1.0).abs() < 1e-12
                || (v.y - 1.0).abs() < 1e-12
                || (v.x - 1.0).abs() < 1e-12
                || (v.x + 1.0).abs() < 1e-12
        })
        .map(|(idx, _)| idx)
        .collect();

    apply_homogeneous_dirichlet_bc_csr(&mut a_global, &dirichlet_nodes, 1);
    apply_homogeneous_dirichlet_bc_rhs(&mut b_global, &dirichlet_nodes, 1);

    Ok((a_global, b_global))
}

fn solve_linear_system(matrix: &CsrMatrix<f64>, rhs: &DVector<f64>) -> eyre::Result<DVector<f64>> {
    let matrix = DMatrix::from(matrix);
    let cholesky = matrix
        .cholesky()
        .ok_or_else(|| eyre!("Failed to solve linear system"))?;
    Ok(cholesky.solve(rhs))
}

struct PoissonProblemSourceFunction;

impl Operator for PoissonProblemSourceFunction {
    type SolutionDim = U1;
}

impl SourceFunction<f64, U2> for PoissonProblemSourceFunction {
    fn evaluate(&self, _coords: &Point2<f64>) -> Vector1<f64> {
        Vector1::new(1.0)
    }
}
