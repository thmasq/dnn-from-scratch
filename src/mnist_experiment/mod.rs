extern crate dnn_from_scratch;
use dnn_from_scratch::loss::Loss;
use dnn_from_scratch::neural_network::NeuralNetwork;
use dnn_from_scratch::report::ReportData;
use dnn_from_scratch::utils::Classification;
use nd::Array2;

mod dataset_setup;
mod plot;

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
        let mut report_data = ReportData::new(n_epochs, "accuracy");
        let loss = Loss::new("cross_entropy");
        for epoch in 1..=n_epochs {
            // Training pipeline
            let output = self.forward(&x_train);
            let train_loss = loss.compute_loss(&output, &y_train);
            let predicted_labels = Classification::argmax(&output, 1);
            let true_labels = Classification::argmax(&y_train, 1);
            let train_accuracy = Classification::compute_accuracy(
                &predicted_labels.into_dyn(),
                &true_labels.into_dyn(),
            );
            let scaling_factor = 6. / output.shape()[0] as f64;
            let output_gradient = (output - y_train.clone()).map(|&v| v * scaling_factor);
            learning_rate = initial_learning_rate / (1. + decay * epoch as f64); // Learning rate step decay
            self.backward(&output_gradient, learning_rate, epoch);
            // Testing pipeline
            let output = self.forward(&x_test);
            let test_loss = loss.compute_loss(&output, &y_test);
            let predicted_labels = Classification::argmax(&output, 1);
            let true_labels = Classification::argmax(&y_test, 1);
            let test_accuracy = Classification::compute_accuracy(
                &predicted_labels.into_dyn(),
                &true_labels.into_dyn(),
            );
            // Report
            report_data.add(train_loss, train_accuracy, test_loss, test_accuracy);
            report_data.print_report(epoch);
        }
        // Save training history and plot losses
        report_data.save_report("mnist_experiment_training_history.txt");
        let (train_accuracy, test_accuracy) = report_data.get_errors();
        plot::plot_error(
            n_epochs,
            train_accuracy,
            test_accuracy,
            "output/mnist_experiment_plot.png",
        );
    }
}

pub fn run_mnist_experiment() {
    // MNIST Architecture
    const INPUT_SIZE: usize = 784;
    const OUTPUT_SIZE: usize = 10;
    const HIDDEN_SIZES: [usize; 2] = [512, 512];
    // Load dataset
    let (x_train, y_train, x_test, y_test) = dataset_setup::load_mnist_dataset("assets/mnist");
    println!("MNIST dataset successfully loaded");
    // Neural Network pipeline
    let random_seed = Some(42); // For reproducibility
    let mut neural_network = NeuralNetwork::new(random_seed);
    neural_network.add_layer(INPUT_SIZE, HIDDEN_SIZES[0], "relu");
    neural_network.add_layer(HIDDEN_SIZES[0], HIDDEN_SIZES[1], "relu");
    neural_network.add_layer(HIDDEN_SIZES[1], OUTPUT_SIZE, "softmax");
    neural_network.train(x_train, y_train, x_test, y_test, 100, 0.001, 0.001);
}
