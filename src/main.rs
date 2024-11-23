extern crate nalgebra as na;
extern crate rand;
extern crate rand_distr;

mod dataset_handler;
mod fully_connected;
mod neural_network;

fn main() {
    const INPUT_SIZE: usize = 784;
    const OUTPUT_SIZE: usize = 10;
    const HIDDEN_SIZES: [usize; 2] = [512, 512];
    // Load the dataset
    let (x_train, y_train, x_test, y_test) = dataset_handler::load_dataset();
    // Further processing
    // ...
    // Instance and initialize the NN
    let nn = neural_network::NeuralNetwork::new(INPUT_SIZE, OUTPUT_SIZE, HIDDEN_SIZES);
    // nn.train();
    // Further processing
    // ...
}
