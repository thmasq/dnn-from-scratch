use crate::activation::Activation;
use crate::fully_connected::FullyConnected;
use nd::Array2;

pub struct NeuralNetwork {
    layers: Vec<FullyConnected>,
    random_seed: Option<u64>,
}

impl NeuralNetwork {
    pub fn new(random_seed: Option<u64>) -> NeuralNetwork {
        NeuralNetwork {
            layers: Vec::new(),
            random_seed,
        }
    }

    pub fn add_layer(&mut self, input_size: usize, output_size: usize, activation: &str) {
        let activation = Activation::new(activation);
        let new_layer = FullyConnected::new(input_size, output_size, activation, self.random_seed);
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
}
