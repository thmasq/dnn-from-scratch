// Temporary
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]

extern crate nalgebra as na;
extern crate ndarray as nd;
extern crate ndarray_npy as npy;
extern crate polars;
extern crate rand;
extern crate rand_distr;

mod dataset_handler;
mod fully_connected;
mod neural_network;

use std::env;

fn main() {
    // Debug env var, uncomment if necessary
    // env::set_var("RUST_BACKTRACE", "1");

    // Project constants
    const INPUT_SIZE: usize = 784;
    const OUTPUT_SIZE: usize = 10;
    const HIDDEN_SIZES: [usize; 2] = [512, 512];
    // Main routine
    let (x_train, y_train, x_test, y_test) =
        dataset_handler::load_rssi_dataset("assets/rssi-dataset.csv", 0.10);
    println!(
        "RSSI dataset shapes: {:?} {:?} {:?} {:?}",
        x_train.shape(),
        y_train.shape(),
        x_test.shape(),
        y_test.shape()
    );
    let (x_train, y_train, x_test, y_test) = dataset_handler::load_mnist_dataset("assets/mnist");
    println!(
        "MNIST dataset shapes: {:?} {:?} {:?} {:?}",
        x_train.shape(),
        y_train.shape(),
        x_test.shape(),
        y_test.shape()
    );
    // NN-related (not yet implemented)
    let neural_network = neural_network::NeuralNetwork::new(INPUT_SIZE, OUTPUT_SIZE, HIDDEN_SIZES);
    neural_network.train(x_train, y_train, x_test, y_test, 100, 0.001, 0.001);
    // neural_network.plot();

    // Further processing
    // ...
}
