use na::DMatrix;
use rand::thread_rng;
use rand_distr::{Distribution, Normal};

/// Generate weights following the HE-Initialization method
fn generate_weights(input_size: usize, output_size: usize) -> DMatrix<f64> {
    let scale = (2.0 / input_size as f64).sqrt();
    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut rng = thread_rng();
    let mut weights = DMatrix::zeros(input_size, output_size);
    weights.row_iter_mut().for_each(|mut row| {
        row.iter_mut().for_each(|element| {
            *element = normal.sample(&mut rng) * scale;
        });
    });
    weights
}

pub struct FullyConnected<'a> {
    // Weights and biases
    weights: DMatrix<f64>,
    biases: DMatrix<f64>,
    // Activation-related
    activation: &'a str,
    // Optimizer-related (Adam)
    m_weights: DMatrix<f64>,
    v_weights: DMatrix<f64>,
    m_biases: DMatrix<f64>,
    v_biases: DMatrix<f64>,
    // Hyper-parameters (Adam)
    beta_1: f64,
    beta_2: f64,
    epsilon: f64,
}

impl FullyConnected<'_> {
    /// Initialize a fully-connected layer
    ///
    /// Args:
    ///
    /// input_size (int): Input shape of the layer
    ///
    /// output_size (int): Output of the layer
    ///
    /// activation (str): Activation function
    pub fn new<'a>(input_size: usize, output_size: usize, activation: &'a str) -> FullyConnected {
        FullyConnected {
            weights: generate_weights(input_size, output_size),
            biases: DMatrix::zeros(1, input_size),
            activation: &activation,
            m_weights: DMatrix::zeros(input_size, output_size),
            v_weights: DMatrix::zeros(input_size, output_size),
            m_biases: DMatrix::zeros(1, input_size),
            v_biases: DMatrix::zeros(1, input_size),
            beta_1: 0.9,
            beta_2: 0.999,
            epsilon: 1e-8,
        }
    }

    /// Forward pass
    ///
    /// Args:
    ///
    /// x: Numerical values of the data (Tensor)
    pub fn forward(x: DMatrix<f64>) {}

    /// Backpropagation
    ///
    /// Args:
    ///
    /// d_values: Derivative of the output
    ///
    /// learning_rate: Learning rate for gradient descent
    ///
    /// t: Timestep
    pub fn backward(d_values: f64, learning_rate: f64, t: i32) {}
}
