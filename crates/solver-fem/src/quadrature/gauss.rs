use std::f64::consts::PI;

use crate::types::Rule;

#[derive(Debug, Clone)]
struct LegendreRecurrence {
    n: usize,
    x: f64,
    p1: f64,
    p2: f64,
}

impl LegendreRecurrence {
    pub fn evaluate(n: usize, x: f64) -> Self {
        let mut p1 = 1.0;
        let mut p2 = 0.0;
        let mut p3;
        for m in 1..=n {
            let m = m as f64;
            p3 = p2;
            p2 = p1;
            p1 = ((2.0 * m - 1.0) * x * p2 - (m - 1.0) * p3) / m;
        }

        Self { n, x, p1, p2 }
    }

    fn value(&self) -> f64 {
        self.p1
    }

    fn derivative(&self) -> f64 {
        let Self { n, x, p1, p2 } = self;
        let n = *n as f64;
        n * (x * p1 - p2) / (x * x - 1.0)
    }

    fn value_and_derivative(&self) -> (f64, f64) {
        (self.value(), self.derivative())
    }
}

pub fn gauss(n: usize) -> Rule<1> {
    assert!(n > 0, "number of points must be positive");

    #[allow(clippy::manual_div_ceil)]
    let m = (n + 1) / 2;

    let mut points = Vec::with_capacity(n);
    let mut weights = Vec::with_capacity(n);

    for i in 0..m {
        let mut x = (PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        let (mut p, mut dp) = LegendreRecurrence::evaluate(n, x).value_and_derivative();

        'newton: loop {
            let dx = -p / dp;
            x += dx;
            let (p_new, dp_new) = LegendreRecurrence::evaluate(n, x).value_and_derivative();
            p = p_new;
            dp = dp_new;
            if dx.abs() <= 1e-15 {
                break 'newton;
            }
        }

        let w = 2.0 / ((1.0 - x * x) * dp * dp);

        points.push([x]);
        weights.push(w);
    }

    for i in m..n {
        let mirror_idx = n - i - 1;

        points.push([-points[mirror_idx][0]]);
        weights.push(weights[mirror_idx]);
    }

    assert_eq!(points.len(), weights.len());
    assert_eq!(
        points.len(),
        n,
        "Internal error: incorrect number of points produced"
    );

    (weights, points)
}

#[cfg(test)]
mod tests {
    use matrixcompare::assert_scalar_eq;

    use crate::quadrature::gauss::LegendreRecurrence;

    #[test]
    fn legendre_recurrence() {
        let num_samples = 10;

        let p: Vec<fn(f64) -> f64> = vec![
            |_| 1.0,
            |x| x,
            |x| 0.5 * (3.0 * x.powi(2) - 1.0),
            |x| 0.5 * (5.0 * x.powi(3) - 3.0 * x),
            |x| (1.0 / 8.0) * (35.0 * x.powi(4) - 30.0 * x.powi(2) + 3.0),
        ];
        let dp: Vec<fn(f64) -> f64> = vec![
            |_| 0.0,
            |_| 1.0,
            |x| 3.0 * x,
            |x| 0.5 * (15.0 * x.powi(2) - 3.0),
            |x| (1.0 / 8.0) * (35.0 * 4.0 * x.powi(3) - 60.0 * x),
        ];

        for n in 0..p.len() {
            for i in 1..num_samples + 1 {
                let x_i = -1.0 + (i as f64) * 2.0 / (num_samples as f64 + 1.0);
                let recurrence = LegendreRecurrence::evaluate(n, x_i);
                assert_scalar_eq!(recurrence.value(), p[n](x_i), comp = abs, tol = 1e-14);
                assert_scalar_eq!(recurrence.derivative(), dp[n](x_i), comp = abs, tol = 1e-14);
            }
        }
    }
}
