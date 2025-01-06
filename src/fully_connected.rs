use na::{DMatrix, RowDVector};
use rand::thread_rng;
use rand_distr::{Distribution, Normal};

/// Generate weights following the HE-Initialization method
fn generate_weights(input_size: usize, output_size: usize) -> DMatrix<f64> {
    let scale = (2.0 / input_size as f64).sqrt();
    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut rng = thread_rng();
    let mut weights = DMatrix::zeros(input_size, output_size);
    weights
        .iter_mut()
        .for_each(|element| *element = normal.sample(&mut rng) * scale);
    weights
}

pub struct FullyConnected<'a> {
    // Weights and biases
    weights: DMatrix<f64>,
    biases: RowDVector<f64>,
    // Activation-related
    activation: &'a str,
    // Optimizer-related (Adam)
    m_weights: DMatrix<f64>,
    v_weights: DMatrix<f64>,
    m_biases: RowDVector<f64>,
    v_biases: RowDVector<f64>,
    // Hyper-parameters (Adam)
    beta_1: f64,
    beta_2: f64,
    epsilon: f64,
    // Input and Output (for the forward pass)
    input: DMatrix<f64>,
    output: DMatrix<f64>,
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
            biases: RowDVector::zeros(input_size),
            activation: &activation,
            m_weights: DMatrix::zeros(input_size, output_size),
            v_weights: DMatrix::zeros(input_size, output_size),
            m_biases: RowDVector::zeros(input_size),
            v_biases: RowDVector::zeros(input_size),
            beta_1: 0.9,
            beta_2: 0.999,
            epsilon: 1e-8,
            input: DMatrix::zeros(1, 1),
            output: DMatrix::zeros(1, 1),
        }
    }

    /// Forward pass
    ///
    /// Args:
    ///
    /// x: Numerical values of the data (Tensor)
    pub fn forward(&mut self, x: DMatrix<f64>) -> DMatrix<f64> {
        self.input = x.clone();
        let mut z = &self.input * &self.weights;
        z.row_iter_mut().for_each(|mut row| row += &self.biases);
        match self.activation {
            "relu" => {
                self.output = z.map(|v| if v > 0.0 { v } else { 0.0 });
            }
            "softmax" => {
                unimplemented!("softmax not yet implemented.")
            }
            _ => {
                panic!("Invalid activation function passed. Use either relu or softmax.")
            }
        }
        self.output.clone()
    }

    /// Backpropagation
    ///
    /// Args:
    ///
    /// d_values: Derivative of the output
    ///
    /// learning_rate: Learning rate for gradient descent
    ///
    /// t: Timestep
    pub fn backward(&mut self, d_values: DMatrix<f64>, learning_rate: f64, t: i32) -> DMatrix<f64> {
        let mut d_values = d_values.clone();
        // Calculate the derivative of the activation function
        if self.activation == "softmax" {
            unimplemented!("softmax not yet implemented.")
        } else if self.activation == "relu" {
            d_values.component_mul_assign(&self.output.map(|x| if x > 0.0 { 1.0 } else { 0.0 }));
        }
        // Calculate the derivative with respect to weights and biases
        let d_weights = self.output.transpose() * &d_values;
        let d_biases = d_values.row_sum();
        // Clip derivatives to avoid extreme values
        let d_weights_clipped = d_weights.map(|x| x.max(-1.0).min(1.0));
        let d_biases_clipped = d_biases.map(|x| x.max(-1.0).min(1.0));
        // Calculate gradient with respect to the input
        let d_inputs = d_values * self.weights.transpose();
        // Update weights using gradient descent
        self.weights -= learning_rate * &d_weights_clipped;
        self.biases -= RowDVector::from(d_biases_clipped.clone()) * learning_rate;
        // Adam optimizer for weights
        self.m_weights = self.beta_1 * &self.m_weights + (1.0 - self.beta_1) * &d_weights_clipped;
        self.v_weights =
            self.beta_2 * &self.v_weights + (1.0 - self.beta_2) * d_weights_clipped.map(|x| x * x);
        let m_hat_weights = &self.m_weights / (1.0 - self.beta_1.powi(t));
        let v_hat_weights = &self.v_weights / (1.0 - self.beta_2.powi(t));
        self.weights -= learning_rate
            * m_hat_weights.component_div(&v_hat_weights.map(|x| x.sqrt() + self.epsilon));
        // Adam optimizer for biases
        self.m_biases = self.beta_1 * &self.m_biases + (1.0 - self.beta_1) * &d_biases_clipped;
        self.v_biases =
            self.beta_2 * &self.v_biases + (1.0 - self.beta_2) * d_biases_clipped.map(|x| x * x);
        let m_hat_biases = &self.m_biases / (1.0 - self.beta_1.powi(t));
        let v_hat_biases = &self.v_biases / (1.0 - self.beta_2.powi(t));
        self.biases -= RowDVector::from(
            m_hat_biases.component_div(&v_hat_biases.map(|x| x.sqrt() + self.epsilon)),
        ) * learning_rate;
        d_inputs
    }
}
