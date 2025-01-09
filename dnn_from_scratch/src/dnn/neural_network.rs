use crate::fully_connected::{Activation, FullyConnected};
use nd::{Array1, Array2, ArrayD};

pub struct NeuralNetwork {
    layers: Vec<FullyConnected>,
}

impl NeuralNetwork {
    pub fn new() -> NeuralNetwork {
        NeuralNetwork { layers: Vec::new() }
    }

    pub fn add_layer(&mut self, input_size: usize, output_size: usize, activation: &str) {
        let activation = match activation {
            "relu" => Activation::ReLU,
            "softmax" => Activation::Softmax,
            "none" => Activation::None,
            _ => panic!("Invalid activation function passed. Use either relu or softmax."),
        };
        let new_layer = FullyConnected::new(input_size, output_size, activation);
        self.layers.push(new_layer);
    }

    pub fn forward(&mut self, inputs: &Array2<f64>) -> Array2<f64> {
        let mut layers_output = self.layers[0].forward(inputs.to_owned());
        for i in 1..self.layers.len() {
            layers_output = self.layers[i].forward(layers_output);
        }
        layers_output
    }

    pub fn backward(&mut self, gradient: &Array2<f64>, learning_rate: f64, time_step: u32) {
        let mut layers_gradient =
            self.layers
                .last_mut()
                .unwrap()
                .backward(gradient.to_owned(), learning_rate, time_step);
        let range = (0..self.layers.len() - 1).rev();
        for i in range {
            layers_gradient = self.layers[i].backward(layers_gradient, learning_rate, time_step);
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

    pub fn argmax(&self, arr: &Array2<f64>, axis: usize) -> Array1<usize> {
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

    pub fn compute_accuracy<T>(&self, output: &ArrayD<T>, target: &ArrayD<T>) -> f64
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
