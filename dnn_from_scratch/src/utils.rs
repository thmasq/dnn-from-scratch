use nd::{Array1, Array2, ArrayD};

pub struct Classification {}
pub struct Regression {}

impl Classification {
    pub fn argmax(arr: &Array2<f64>, axis: usize) -> Array1<usize> {
        let shape = arr.shape();
        let (nrows, ncols) = (shape[0], shape[1]);
        let (upper_bound_i, upper_bound_j) = match axis {
            0 => (ncols, nrows),
            1 => (nrows, ncols),
            _ => panic!("Axis must be 0 or 1"),
        };
        let mut result = Vec::with_capacity(upper_bound_i);
        for i in 0..upper_bound_i {
            let mut max_val = f64::NEG_INFINITY;
            let mut max_idx = 0;
            for j in 0..upper_bound_j {
                let val = arr[[i, j]];
                if val > max_val {
                    max_val = val;
                    max_idx = j;
                }
            }
            result.push(max_idx);
        }
        Array1::from_vec(result)
    }

    pub fn compute_accuracy<T>(output: &ArrayD<T>, target: &ArrayD<T>) -> f64
    where
        T: PartialEq,
    {
        let mut total_elements = 0;
        let mut equal_elements = 0;
        output
            .iter()
            .zip(target.iter())
            .for_each(|(predicted, expected)| {
                total_elements += 1;
                if *predicted == *expected {
                    equal_elements += 1;
                }
            });
        (equal_elements as f64) / (total_elements as f64)
    }
}

impl Regression {
    pub fn cumulative_distribution(
        predictions: &Array2<f64>,
        expected: &Array2<f64>,
    ) -> (Vec<f64>, Vec<f64>) {
        let mut errors = Vec::new();
        let (mut error_x, mut error_y, mut rmse);
        let top = predictions.shape()[0];
        for i in 0..top {
            error_x = predictions[[i, 0]] - expected[[i, 0]];
            error_y = predictions[[i, 1]] - expected[[i, 1]];
            rmse = (error_x.powi(2) + error_y.powi(2)).sqrt();
            errors.push(rmse);
        }
        // Sort the errors
        let mut sorted_errors = errors.clone();
        sorted_errors.sort_by(|&a, b| a.partial_cmp(b).unwrap());
        // Calculate the CDF
        let len = sorted_errors.len() as f64;
        let cdf = Vec::from_iter((1..=(sorted_errors.len() + 1)).map(|i| i as f64 / len));
        (sorted_errors, cdf)
    }
}
