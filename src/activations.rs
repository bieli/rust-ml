#[derive(Clone, Copy)]
pub enum Activation {
	Sigmoid,
	Relu,
	Tanh,
	Linear,
}

impl Activation {
	pub fn apply(self, x: f64) -> f64 {
		match self {
			Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
			Activation::Relu => {
				if x > 0.0 {
					x
				} else {
					0.0
				}
			}
			Activation::Tanh => x.tanh(),
			Activation::Linear => x,
		}
	}

	// derivative - necessary to improve the loss function and model predictions
	pub fn derivative(self, y: f64) -> f64 {
		match self {
			Activation::Sigmoid => y * (1.0 - y),
			Activation::Relu => {
				if y > 0.0 {
					1.0
				} else {
					0.0
				}
			}
			Activation::Tanh => 1.0 - y.powi(2),
			Activation::Linear => 1.0,
		}
	}
}
