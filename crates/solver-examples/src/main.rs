use crate::poisson2d::poisson2d;

pub mod poisson2d;

fn main() -> eyre::Result<()> {
    poisson2d()?;
    Ok(())
}
