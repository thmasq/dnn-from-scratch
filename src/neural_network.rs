use nd::{s, Array3, Axis};

use crate::fully_connected::FullyConnected;

pub struct NeuralNetwork<'a> {
    layer_1: FullyConnected<'a>,
    layer_2: FullyConnected<'a>,
    layer_3: FullyConnected<'a>,
}

impl NeuralNetwork<'_> {
    pub fn new<'a>(
        input_size: usize,
        output_size: usize,
        hidden_sizes: [usize; 2],
    ) -> NeuralNetwork<'a> {
        NeuralNetwork {
            layer_1: FullyConnected::new(input_size, hidden_sizes[0], "relu"),
            layer_2: FullyConnected::new(hidden_sizes[0], hidden_sizes[1], "relu"),
            layer_3: FullyConnected::new(hidden_sizes[1], output_size, "softmax"),
        }
    }

    fn forward(&mut self, inputs: Array3<f64>) -> Array3<f64> {
        let shape = (inputs.shape()[0], inputs.shape()[1], inputs.shape()[2]);
        let mut output = Array3::zeros(shape);
        inputs.axis_iter(Axis(0)).enumerate().for_each(|(i, row)| {
            let layer1_output = self.layer_1.forward(row.to_owned());
            let layer2_output = self.layer_2.forward(layer1_output);
            let layer3_output = self.layer_3.forward(layer2_output);
            output.slice_mut(s![i, .., ..]).assign(&layer3_output);
        });
        output
    }

    /// This function does the training process of the model.
    /// Firstly, forward propagation is done,
    /// then the loss and accuracy are calculated,
    /// after that the backpropagation is done.
    pub fn train(
        &self,
        x_train: Array3<f64>, // inputs
        y_train: Array3<f64>, //targets
        x_test: Array3<f64>,
        y_test: Array3<f64>,
        n_epochs: u32,
        initial_learning_rate: f32,
        decay: f32,
    ) {
        unimplemented!("Train function is not yet implemented")
    }

    /// This function will plot Loss over Epochs
    /// and Accuracy over Epochs.
    pub fn plot(&self) {
        unimplemented!("Plot function is not yet implemented")
    }
}
