use nalgebra::DimName;

pub trait Operator {
    type SolutionDim: DimName;
}
