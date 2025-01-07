use nd::{s, Array1, Array2, Array3, Axis};

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

    fn forward(&mut self, inputs: &Array2<f64>) -> Array2<f64> {
        let layer1_output = self.layer_1.forward(inputs.to_owned());
        let layer2_output = self.layer_2.forward(layer1_output);
        let layer3_output = self.layer_3.forward(layer2_output);
        layer3_output
    }

    fn backward(&mut self, gradient: &Array2<f64>, learning_rate: f64, time_step: u32) {
        let layer3_gradient = self
            .layer_3
            .backward(gradient.to_owned(), learning_rate, time_step);
        let layer2_gradient = self
            .layer_2
            .backward(layer3_gradient, learning_rate, time_step);
        let layer1_gradient = self
            .layer_1
            .backward(layer2_gradient, learning_rate, time_step);
    }

    fn categorical_cross_entropy(&self, output: &Array2<f64>, target: &Array2<f64>) -> f64 {
        let mut total_sum = 0.;
        let mut total_elements = 0.;
        const EPSILON: f64 = 1e-10;
        output
            .iter()
            .zip(target.iter())
            .for_each(|(predicted, expected)| {
                total_sum += *expected * f64::ln(*expected + EPSILON);
                total_elements += 1.;
            });
        total_sum / total_elements
    }

    fn argmax(&self, array: &Array2<f64>, axis: usize) -> Array1<usize> {
        let axis = Axis(axis);
        // Validate the axis value
        assert!(
            axis.index() < array.ndim(),
            "Invalid axis: {} for array with {} dimensions",
            axis.index(),
            array.ndim()
        );
        // Compute the argmax along the specified axis
        array.map_axis(axis, |subview| {
            subview
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0) // Default to 0 if subview is empty
        })
    }

    fn compute_accuracy<T>(&self, output: &Array1<T>, target: &Array1<T>) -> f64
    where
        T: Eq,
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

    /// This function does the training process of the model.
    /// Firstly, forward propagation is done,
    /// then the loss and accuracy are calculated,
    /// after that the backpropagation is done.
    pub fn train(
        &mut self,
        x_train: Array2<f64>, // inputs
        y_train: Array2<f64>, //targets
        x_test: Array2<f64>,
        y_test: Array2<f64>,
        n_epochs: u32,
        initial_learning_rate: f64,
        decay: f64,
    ) {
        let epsilon = 1e-10;
        let mut losses = Vec::new();
        let mut accuracies = Vec::new();
        let mut learning_rate;
        for epoch in 0..n_epochs {
            let output = self.forward(&x_train);
            let loss = self.categorical_cross_entropy(&output, &y_train);
            let predicted_labels = self.argmax(&output, 1);
            let true_labels = self.argmax(&y_train, 1);
            let accuracy = self.compute_accuracy(&predicted_labels, &true_labels);
            let scaling_factor = 6. / output.shape()[0] as f64;
            let output_gradient = (output - y_train.clone()).map(|&v| v * scaling_factor);
            learning_rate = initial_learning_rate / (1. + decay * epoch as f64);
            self.backward(&output_gradient, learning_rate, epoch);
            losses.push(loss);
            accuracies.push(accuracy);
            println!("Epoch: {}, loss: {}, accuracy: {}", epoch, loss, accuracy);
        }
    }

    /// This function will plot Loss over Epochs
    /// and Accuracy over Epochs.
    pub fn plot(&self) {
        unimplemented!("Plot function is not yet implemented")
    }
}
