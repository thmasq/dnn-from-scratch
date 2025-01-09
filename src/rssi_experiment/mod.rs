use dnn_from_scratch::neural_network::NeuralNetwork;
use dnn_from_scratch::report::ReportData;
use nd::Array2;

mod dataset_setup;

#[allow(dead_code)]
pub trait Training {
    fn train(
        &mut self,
        x_train: Array2<f64>,
        y_train: Array2<f64>,
        x_test: Array2<f64>,
        y_test: Array2<f64>,
        n_epochs: u32,
        initial_learning_rate: f64,
        decay: f64,
    );
}

impl Training for NeuralNetwork {
    fn train(
        &mut self,
        x_train: Array2<f64>,
        y_train: Array2<f64>,
        x_test: Array2<f64>,
        y_test: Array2<f64>,
        n_epochs: u32,
        initial_learning_rate: f64,
        decay: f64,
    ) {
        let mut learning_rate;
        let mut report_data = ReportData::new(n_epochs);
        for epoch in 1..=n_epochs {
            // Training pipeline
            let output = self.forward(&x_train);
            let train_loss = self.categorical_cross_entropy(&output, &y_train);
            let predicted_labels = self.argmax(&output, 1);
            let true_labels = self.argmax(&y_train, 1);
            let train_accuracy = self.compute_accuracy(&predicted_labels, &true_labels);
            let scaling_factor = 6. / output.shape()[0] as f64;
            let output_gradient = (output - y_train.clone()).map(|&v| v * scaling_factor);
            learning_rate = initial_learning_rate / (1. + decay * epoch as f64);
            self.backward(&output_gradient, learning_rate, epoch);
            // Testing pipeline
            let output = self.forward(&x_test);
            let test_loss = self.categorical_cross_entropy(&output, &y_test);
            let predicted_labels = self.argmax(&output, 1);
            let true_labels = self.argmax(&y_test, 1);
            let test_accuracy = self.compute_accuracy(&predicted_labels, &true_labels);
            // Report
            report_data.add(train_loss, train_accuracy, test_loss, test_accuracy);
            report_data.print_report(epoch);
        }
        report_data.save_report(true, true);
    }
}

#[allow(dead_code)]
pub fn run_rssi_experiment() {
    // MNIST Architecture
    const INPUT_SIZE: usize = 13;
    const OUTPUT_SIZE: usize = 2;
    const HIDDEN_SIZES: [usize; 2] = [200, 200];
    // Load dataset
    let (x_train, y_train, x_test, y_test) =
        dataset_setup::load_rssi_dataset("assets/rssi/rssi-dataset.csv", 0.15);
    println!("RSSI dataset successfully loaded, shapes:");
    println!(
        "{:?} {:?} {:?} {:?}",
        x_train.shape(),
        y_train.shape(),
        x_test.shape(),
        y_test.shape()
    );
    // Neural Network pipeline
    // let mut neural_network = NeuralNetwork::new();
    // neural_network.add_layer(INPUT_SIZE, HIDDEN_SIZES[0], "relu");
    // neural_network.add_layer(HIDDEN_SIZES[0], HIDDEN_SIZES[1], "relu");
    // neural_network.add_layer(HIDDEN_SIZES[1], OUTPUT_SIZE, "none");
    // neural_network.train(x_train, y_train, x_test, y_test, 100, 0.001, 0.001);
}
