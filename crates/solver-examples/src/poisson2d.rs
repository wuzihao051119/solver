use nalgebra::{DVector, Point2, U1, U2, Vector1};
use solver_fem::assembly::global::VectorAssembler;
use solver_fem::assembly::local::quadrature_table::UniformQuadratureTable;
use solver_fem::assembly::local::source::{ElementSourceAssemblerBuilder, SourceFunction};
use solver_fem::assembly::operators::Operator;
use solver_fem::io::msh::load_msh_from_file;
use solver_fem::mesh::QuadMesh2d;
use solver_fem::quadrature::tensor::quadrilateral_gauss;

use crate::types::Matrix9x1;

pub fn poisson2d() -> eyre::Result<()> {
    let mesh = load_msh_from_file("assets/poisson2d.msh")?;

    let b = assemble_linear_system(&mesh)?;

    let expected_b =
        Matrix9x1::from_column_slice(&[0.25, 0.5, 0.25, 0.5, 1.0, 0.5, 0.25, 0.5, 0.25]);

    matrixcompare::assert_matrix_eq!(b, expected_b, comp = abs, tol = 1e-14);

    Ok(())
}

fn assemble_linear_system(mesh: &QuadMesh2d<f64>) -> eyre::Result<DVector<f64>> {
    let (weights, points) = quadrilateral_gauss(2);
    let quadrature = UniformQuadratureTable::from_points_and_weights(points, weights);

    let vector_assembler = VectorAssembler::<f64>::default();

    let source_assembler = ElementSourceAssemblerBuilder::new()
        .with_finite_element_space(mesh)
        .with_quadrature_table(&quadrature)
        .with_source(&PoissonProblemSourceFunction)
        .build();

    let b_global = vector_assembler.assemble_vector(&source_assembler)?;

    Ok(b_global)
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
