extern crate ndarray as nd;
extern crate ndarray_npy as npy;

mod mnist_experiment;
mod rssi_experiment;

fn main() {
    rssi_experiment::run_rssi_experiment();
    mnist_experiment::run_mnist_experiment();
}
