use nd::{Array1, Array2, ArrayD};

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
