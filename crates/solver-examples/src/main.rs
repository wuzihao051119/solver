use crate::poisson2d::poisson2d;

pub mod poisson2d;
pub mod types;

fn main() -> eyre::Result<()> {
    poisson2d()?;
    Ok(())
}
