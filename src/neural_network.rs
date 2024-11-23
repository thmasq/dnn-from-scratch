use crate::fully_connected::FullyConnected;

pub struct NeuralNetwork<'a> {
    layer_1: FullyConnected<'a>,
    layer_2: FullyConnected<'a>,
    layer_3: FullyConnected<'a>,
}

impl NeuralNetwork<'_> {
    pub fn new<'a>(
        input_size: usize,
        output_size: usize,
        hidden_size: [usize; 2],
    ) -> NeuralNetwork<'a> {
        NeuralNetwork {
            layer_1: FullyConnected::new(input_size, hidden_size[0], "relu"),
            layer_2: FullyConnected::new(hidden_size[0], hidden_size[1], "relu"),
            layer_3: FullyConnected::new(hidden_size[1], output_size, "softmax"),
        }
    }

    pub fn train(self) {}

    pub fn plot(self) {}
}
