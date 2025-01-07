extern crate nalgebra as na;
extern crate ndarray as nd;
extern crate polars;
extern crate rand;
extern crate rand_distr;

mod dataset_handler;
mod fully_connected;
mod neural_network;

fn main() {
    const INPUT_SIZE: usize = 784;
    const OUTPUT_SIZE: usize = 10;
    const HIDDEN_SIZES: [usize; 2] = [512, 512];
    let (x_train, y_train, x_test, y_test) =
        dataset_handler::load_rssi_dataset("static/rssi-dataset.csv", 0.10);
    let neural_network = neural_network::NeuralNetwork::new(INPUT_SIZE, OUTPUT_SIZE, HIDDEN_SIZES);
    neural_network.train(x_train, y_train, x_test, y_test, 100, 0.001, 0.001);
    neural_network.plot();
}
