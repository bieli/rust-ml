mod activations;
mod matrix;
mod network;

use activations::Activation;
use network::Network;

const XOR_PRED_MODEL_FILE: &str = "xor_model.json";

fn main() {
	let inputs = vec![
		vec![0.0, 0.0],
		vec![0.0, 1.0],
		vec![1.0, 0.0],
		vec![1.0, 1.0],
	];

	let targets = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];

	let mut net = Network::new(
		vec![2, 3, 1],          // - a little worse results after training
		//vec![2, 4, 3, 1],     // - the best results after training
		0.5,
		Activation::Sigmoid,
	);

	println!("Before training:");
	println!("{:?}", net.inference(vec![0.0, 0.0]));
	println!("{:?}", net.inference(vec![0.0, 1.0]));
	println!("{:?}", net.inference(vec![1.0, 0.0]));
	println!("{:?}", net.inference(vec![1.0, 1.0]));
	std::thread::sleep(std::time::Duration::from_millis(5000));

	//net.train(inputs, targets, 100);
	//net.train(inputs, targets, 500);
	net.train(inputs, targets, 1000);
	//net.train(inputs, targets, 5000);
	net.save(XOR_PRED_MODEL_FILE.to_string());

	println!("After training:");
	println!("{:?}", net.inference(vec![0.0, 0.0]));
	println!("{:?}", net.inference(vec![0.0, 1.0]));
	println!("{:?}", net.inference(vec![1.0, 0.0]));
	println!("{:?}", net.inference(vec![1.0, 1.0]));
}
