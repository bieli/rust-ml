# Basic Neural Networks implementation in RUST with examples

## Motivation
This is example upgraded and working verion of Neural Networks forked from other github repo.
Intention of this forked code it to present NN basics on meetups and conferences, when I have a chance to share my knowladge about Machine Learning.

### Where I presented ML NN concepts

- Polish language YT recording you can watch here on: [IoT, Hardware and Robotics meetup #7](https://www.youtube.com/watch?v=cnhuSjFtERs)
- English language version -> `comming soon ... from RUSTMEET 2026`

## How to run?
```bash
cargo run
```

## Working example of NN for XOR gate with important network settings
```rust
mod matrix;
mod network;
mod activations;

use network::Network;
use activations::Activation;

fn main() {
    let inputs = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];

    let targets = vec![
        vec![0.0],
        vec![1.0],
        vec![1.0],
        vec![0.0],
    ];

    let mut net = Network::new(
        // NOTE: SHOW ON MEETUP
        vec![2, 3, 1],           // - a little lower accuracy with this network architecture
        //vec![2, 4, 3, 1],      // - better accuracy with this network architecture
        0.5,
        Activation::Sigmoid,
    );

    println!("{:?}", net.feed_forward(vec![0.0, 0.0]));
    println!("{:?}", net.feed_forward(vec![0.0, 1.0]));
    println!("{:?}", net.feed_forward(vec![1.0, 0.0]));
    println!("{:?}", net.feed_forward(vec![1.0, 1.0]));

    // NOTE: SHOW ON MEETUP
    //net.train(inputs, targets, 100);
    net.train(inputs, targets, 1000);
    //net.train(inputs, targets, 5000);

    println!("After training:");
    println!("{:?}", net.feed_forward(vec![0.0, 0.0]));
    println!("{:?}", net.feed_forward(vec![0.0, 1.0]));
    println!("{:?}", net.feed_forward(vec![1.0, 0.0]));
    println!("{:?}", net.feed_forward(vec![1.0, 1.0]));
}
```

### What you can see after run?
```bash
$ cargo run

Before training:
[0.6099642504817715]
[0.6683502676140888]
[0.6248547167472169]
[0.6816736568692362]



=== Network Visualization In Training Phase (weights & biases) ===

Input Layer (2 neurons)
  |
  x1   x2 
  |
  v
Hidden Layer (3 neurons)
  H1:  0.05  0.54 | bias: 0.12
  H2: -0.36 -0.11 | bias: -0.84
  H3: -0.16  0.42 | bias: -0.61

Output Layer (1 neuron)
  O1: -0.56  0.41  0.04 | bias: 0.21

=============================================

Training finished for epochs 100 - FAIL AFTER FIRST TRAINING!
After training:
Predicted: [0.51317241733852]	| Target: [0.0]
Predicted: [0.49376105236113843]	| Target: [1.0]
Predicted: [0.5036966522356805]	| Target: [1.0]
Predicted: [0.48480635178676607]	| Target: [0.0]
```

### Side effects

Biases and weights will be saved in output file `xor_model.json`. You can see something like that after running below code:

```bash
{
  "biases": [
    [
      [
        -2.734447934684026
      ],
      [
        -1.9794427870206694
      ],
      [
        1.8436379060136001
      ]
    ],
    [
      [
        -0.864748292576498
      ]
    ]
  ],
  "weights": [
    [
      [
        -8.648698117566433,
        6.62219354571712
      ],
      [
        6.998102132468941,
        6.702193001346195
      ],
      [
        -5.486457038382822,
        8.201541955878316
      ]
    ],
    [
      [
        5.399112499074607,
        3.665876868522426,
        -5.292558224023659
      ]
    ]
  ]
}
```

### Special thanks
Thanks for author - Open Source contributor - of forked [repo.](https://github.com/neowsl/rust-ml) for inspiration and baseline to start this working implementation (original repository code doesn't work - it was PoC only!).
