use nd::Array2;

enum LossType {
    CrossEntropy,
    MSE,
    RMSE,
}

pub struct Loss {
    loss: LossType,
}

impl Loss {
    pub fn new(loss_type: &str) -> Loss {
        match loss_type {
            "cross_entropy" => Loss {
                loss: LossType::CrossEntropy,
            },
            "mse" => Loss {
                loss: LossType::MSE,
            },
            "rmse" => Loss {
                loss: LossType::RMSE,
            },
            _ => panic!("Invalid loss passed."),
        }
    }

    pub fn compute_loss(&self, output: &Array2<f64>, target: &Array2<f64>) -> f64 {
        match self.loss {
            LossType::CrossEntropy => self.categorical_cross_entropy(output, target),
            LossType::MSE => self.mse(output, target),
            LossType::RMSE => self.rmse(output, target),
        }
    }

    pub fn categorical_cross_entropy(&self, output: &Array2<f64>, target: &Array2<f64>) -> f64 {
        let mut total_sum = 0.;
        let mut total_elements = 0.;
        const EPSILON: f64 = 1e-10;
        output
            .iter()
            .zip(target.iter())
            .for_each(|(predicted, expected)| {
                total_sum += *predicted * f64::ln(*expected + EPSILON);
                total_elements += 1.;
            });
        -1. * (total_sum / total_elements)
    }

    pub fn mse(&self, output: &Array2<f64>, target: &Array2<f64>) -> f64 {
        let diff = output - target;
        let squared_diff = diff.mapv(|x| x.powi(2));
        let mse = squared_diff.mean().unwrap_or(0.0);
        mse
    }

    pub fn rmse(&self, output: &Array2<f64>, target: &Array2<f64>) -> f64 {
        let diff = output - target;
        let squared_diff = diff.mapv(|x| x.powi(2));
        let mse = squared_diff.mean().unwrap_or(0.0);
        mse.sqrt()
    }
}
