use nd::{Array2, Axis};

enum ActivationType {
    Softmax,
    ReLU,
    None,
}

pub struct Activation {
    activation_type: ActivationType,
}

impl Activation {
    pub fn new(activation_type: &str) -> Activation {
        let activation_type = match activation_type {
            "relu" => ActivationType::ReLU,
            "softmax" => ActivationType::Softmax,
            "none" => ActivationType::None,
            _ => panic!("Invalid activation type."),
        };
        Activation { activation_type }
    }

    pub fn forward(&self, mut z: Array2<f64>) -> Array2<f64> {
        match self.activation_type {
            ActivationType::ReLU => z.mapv(|v| v.max(0.)),
            ActivationType::Softmax => {
                for mut row in z.rows_mut() {
                    let z_max = row.iter().fold(f64::NEG_INFINITY, |max, &v| max.max(v));
                    row.iter_mut().for_each(|v| *v -= z_max);
                }
                let mut exp_values = z.mapv(|v| v.exp());
                for mut row in exp_values.rows_mut() {
                    let sum: f64 = row.iter().sum();
                    row.iter_mut().for_each(|v| *v /= sum);
                }
                exp_values
            }
            ActivationType::None => z.to_owned(),
        }
    }

    pub fn backward(&self, d_values: &Array2<f64>, z: &Array2<f64>) -> Array2<f64> {
        let mut d_values = d_values.clone();
        match self.activation_type {
            ActivationType::ReLU => d_values *= &z.mapv(|x| if x > 0. { 1. } else { 0. }),
            ActivationType::Softmax => {
                for i in 0..d_values.nrows() {
                    let gradient = d_values.row(i).to_owned();
                    let diagonal = Array2::from_diag(&gradient);
                    let outer_product = gradient
                        .clone()
                        .insert_axis(Axis(1))
                        .dot(&gradient.clone().insert_axis(Axis(0)));
                    let jacobian_matrix = diagonal - outer_product;
                    let transformed_gradient =
                        jacobian_matrix.dot(&z.row(i).to_owned().insert_axis(Axis(1)));
                    let result = transformed_gradient.index_axis(Axis(1), 0);
                    d_values.row_mut(i).assign(&result);
                }
            }
            ActivationType::None => {}
        }
        d_values
    }
}
