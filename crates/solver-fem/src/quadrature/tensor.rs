use crate::quadrature::gauss::gauss;
use crate::types::Rule;

pub fn quadrilateral_gauss(n: usize) -> Rule<2> {
    let (weights1d, points1d) = gauss(n);
    let mut weights2d = Vec::with_capacity(n * n);
    let mut points2d = Vec::with_capacity(n * n);

    let rule1d_iter = || weights1d.iter().zip(&points1d);

    for (&wx, &[x]) in rule1d_iter() {
        for (&wy, &[y]) in rule1d_iter() {
            let w = wx * wy;
            weights2d.push(w);
            points2d.push([x, y]);
        }
    }

    (weights2d, points2d)
}

pub fn hexahedron_gauss(n: usize) -> Rule<3> {
    let (weights1d, points1d) = gauss(n);
    let mut weights3d = Vec::with_capacity(n * n * n);
    let mut points3d = Vec::with_capacity(n * n * n);

    let rule1d_iter = || weights1d.iter().zip(&points1d);

    for (&wx, &[x]) in rule1d_iter() {
        for (&wy, &[y]) in rule1d_iter() {
            for (&wz, &[z]) in rule1d_iter() {
                let w = wx * wy * wz;
                weights3d.push(w);
                points3d.push([x, y, z]);
            }
        }
    }

    (weights3d, points3d)
}
