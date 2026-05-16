use std::{
	fs::File,
	io::{Read, Write},
};

use serde::{Deserialize, Serialize};
use serde_json::{from_str, json};

use super::matrix::Matrix;
use crate::activations::Activation;

pub struct Network {
	layers: Vec<usize>,
	weights: Vec<Matrix>,
	biases: Vec<Matrix>,
	data: Vec<Matrix>, // we store a copy of the intermediate layers (A)
	learning_rate: f64,
	activation: Activation,
}

#[derive(Serialize, Deserialize)]
struct SaveData {
	weights: Vec<Vec<Vec<f64>>>,
	biases: Vec<Vec<Vec<f64>>>,
}

impl Network {
	pub fn new(layers: Vec<usize>, learning_rate: f64, activation: Activation) -> Self {
		let mut weights = vec![];
		let mut biases = vec![];

		for i in 0..layers.len() - 1 {
			weights.push(Matrix::random(layers[i + 1], layers[i]));
			biases.push(Matrix::random(layers[i + 1], 1));
		}

		Self {
			layers,
			weights,
			biases,
			data: vec![],
			learning_rate,
			activation,
		}
	}

	// Feed forward: A = f(W*A_prev + b)
	pub fn feed_forward(&mut self, inputs: Vec<f64>) -> Vec<f64> {
		assert_eq!(inputs.len(), self.layers[0]);

		let mut current = Matrix::from(vec![inputs]).transpose();
		self.data = vec![current.clone()]; // store the input layer

		for i in 0..self.layers.len() - 1 {
			let z = self.weights[i].multiply(&current).add(&self.biases[i]);
			current = z.map(|x| self.activation.apply(x));
			self.data.push(current.clone()); // store the activations
		}

		current.transpose().data[0].clone()
	}

	pub fn inference(&mut self, inputs: Vec<f64>) -> Vec<f64> {
		self.feed_forward(inputs)
	}

	// Backpropagation the classic way (no complications)
	pub fn back_propagate(&mut self, outputs: Vec<f64>, targets: Vec<f64>) {
		assert_eq!(targets.len(), *self.layers.last().unwrap());

		let output = Matrix::from(vec![outputs]).transpose();
		let target = Matrix::from(vec![targets]).transpose();

		// Output layer error
		let mut errors = target.subtract(&output);

		for i in (0..self.layers.len() - 1).rev() {
			// gradient = errors ⊙ f'(A)
			let mut gradients = self.data[i + 1]
				.map(|y| self.activation.derivative(y))
				.dot_multiply(&errors)
				.map(|x| x * self.learning_rate);

			//println!("self.learning_rate: {}", self.learning_rate);

			// W[i] += gradients * A_prev^T
			self.weights[i] = self.weights[i].add(&gradients.multiply(&self.data[i].transpose()));

			// Biases
			self.biases[i] = self.biases[i].add(&gradients);

			// New error for previous layer
			errors = self.weights[i].transpose().multiply(&errors);
		}
	}

	pub fn train(&mut self, inputs: Vec<Vec<f64>>, targets: Vec<Vec<f64>>, epochs: u16) {
		println!("Start training for {} epochs", epochs);
		for epoch in 1..=epochs {
			if epochs < 100 || epoch % (epochs / 100) == 0 {
				//println!("Epoch {} of {}", epoch, epochs);
				print!(".");
			}

			for i in 0..inputs.len() {
				let outputs = self.feed_forward(inputs[i].clone());
				self.back_propagate(outputs, targets[i].clone());
			}

			if epoch % 100 == 0 {
				println!("Epoch: {}", epoch);
				self.visualize_ascii(20);
			}
		}
		println!("\nTraining finished for epochs {}", epochs);
	}

	pub fn save(&self, file_name: String) {
		let mut file = File::create(&file_name).expect("Unable to create save file");

		file.write_all(
			json!({
				"weights": self.weights.iter().map(|m| &m.data).collect::<Vec<_>>(),
				"biases": self.biases.iter().map(|m| &m.data).collect::<Vec<_>>(),
			})
			.to_string()
			.as_bytes(),
		)
		.expect("Unable to write save file");
		println!(
			"\nML model weights & biases saved to file: {} ...",
			&file_name
		);
	}

	pub fn load(&mut self, file_name: String) {
		let mut file = File::open(&file_name).expect("Unable to open save file");
		let mut buffer = String::new();

		file.read_to_string(&mut buffer)
			.expect("Unable to read save file");

		let save_data: SaveData = from_str(&buffer).expect("Unable to parse save data");

		let mut weights = vec![];
		let mut biases = vec![];

		for i in 0..self.layers.len() - 1 {
			weights.push(Matrix::from(save_data.weights[i].clone()));
			biases.push(Matrix::from(save_data.biases[i].clone()));
		}

		self.weights = weights;
		self.biases = biases;
		println!(
			"\nML model weights & biases loaded from file: {} ...",
			&file_name
		);
	}

	pub fn visualize_ascii(&self, delay_ms: u64) {
		print!("\x1b[2J\x1b[H");

		const RESET: &str = "\x1b[0m";
		const RED: &str = "\x1b[31m";
		const GREEN: &str = "\x1b[32m";
		const YELLOW: &str = "\x1b[33m";
		const CYAN: &str = "\x1b[36m";

		println!("=== Network Visualization In Training Phase (weights & biases) ===\n");

		println!("Input Layer ({} neurons)", self.layers[0]);
		println!("  |");
		for i in 0..self.layers[0] {
			print!("  x{} ", i + 1);
		}
		println!("\n  |\n  v");

		println!("Hidden Layer ({} neurons)", self.layers[1]);
		for h in 0..self.layers[1] {
			print!("  H{}: ", h + 1);
			for w in 0..self.layers[0] {
				let weight_val = self.weights[0].get(h, w);
				print!("{}{:>5.2}{}", CYAN, weight_val, RESET);
				if w < self.layers[0] - 1 {
					print!(" ");
				}
			}
			let bias_val = self.biases[0].get(h, 0);
			print!(" | bias: {}{:.2}{}", YELLOW, bias_val, RESET);
			println!();
		}

		println!("\nOutput Layer ({} neuron)", self.layers[2]);
		for o in 0..self.layers[2] {
			print!("  O{}: ", o + 1);
			for w in 0..self.layers[1] {
				let weight_val = self.weights[1].get(o, w);
				print!("{}{:>5.2}{}", RED, weight_val, RESET);
				if w < self.layers[1] - 1 {
					print!(" ");
				}
			}
			let bias_val = self.biases[1].get(o, 0);
			print!(" | bias: {}{:.2}{}", YELLOW, bias_val, RESET);
			println!();
		}

		println!("\n=============================================");
		std::thread::sleep(std::time::Duration::from_millis(delay_ms));
	}
}
