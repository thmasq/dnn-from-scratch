extern crate charming;
extern crate ndarray as nd;
extern crate rand;
extern crate rand_distr;

pub mod dnn;
pub mod utils;

pub use crate::dnn::fully_connected;
pub use crate::dnn::neural_network;
pub use crate::utils::report;
