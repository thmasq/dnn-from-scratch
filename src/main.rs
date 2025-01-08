extern crate charming;
extern crate nalgebra as na;
extern crate ndarray as nd;
extern crate ndarray_npy as npy;
extern crate polars;
extern crate rand;
extern crate rand_distr;

mod dataset_handler;
mod fully_connected;
mod neural_network;
mod report;

fn main() {
    // MNIST constants
    const INPUT_SIZE: usize = 784;
    const OUTPUT_SIZE: usize = 10;
    const HIDDEN_SIZES: [usize; 2] = [512, 512];
    // Load dataset
    let (x_train, y_train, x_test, y_test) = dataset_handler::load_mnist_dataset("assets/mnist");
    println!(
        "MNIST dataset shapes: {:?} {:?} {:?} {:?}",
        x_train.shape(),
        y_train.shape(),
        x_test.shape(),
        y_test.shape()
    );
    // NN-related
    let mut neural_network =
        neural_network::NeuralNetwork::new(INPUT_SIZE, OUTPUT_SIZE, HIDDEN_SIZES);
    neural_network.train(x_train, y_train, x_test, y_test, 100, 0.001, 0.001);
    // Further processing
    // let (x_train, y_train, x_test, y_test) =
    //     dataset_handler::load_rssi_dataset("assets/rssi/rssi-dataset.csv", 0.10);
    // println!(
    //     "RSSI dataset shapes: {:?} {:?} {:?} {:?}",
    //     x_train.shape(),
    //     y_train.shape(),
    //     x_test.shape(),
    //     y_test.shape()
    // );
}
