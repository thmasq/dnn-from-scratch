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
    // Load the dataset
    let (x_train, y_train, x_test, y_test) =
        dataset_handler::load_rssi_dataset("static/rssi-dataset.csv", 0.10);
    println!("Dataset loaded. Shape of matrices:");
    println!(
        "{:?} {:?} {:?} {:?}\n",
        x_train.shape(),
        y_train.shape(),
        x_test.shape(),
        y_test.shape()
    );
    println!("y_train first row");
    println!("{}", y_train.row(0));
    println!("x_train first row");
    println!("{}", x_train.row(0));
    // Further processing
    // ...
    // Instance and initialize the NN
    // let nn = neural_network::NeuralNetwork::new(INPUT_SIZE, OUTPUT_SIZE, HIDDEN_SIZES);
    // nn.train();
    // Further processing
    // ...
}
