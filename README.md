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
  H1:  6.39 -8.48 | bias: -2.64
  H2:  6.24  6.07 | bias: -1.76
  H3:  8.86 -7.11 | bias: 3.08

Output Layer (1 neuron)
  O1:  5.65  2.79 -5.54 | bias: 0.13

=============================================

Training finished for epochs 1000

ML model weights & biases saved to file: xor_model.json ...
After training:
[0.012376023928445052]
[0.941951257070452]
[0.946291342673547]
[0.07426817688525245]
```

### Side effects

Biases and weights will be saved in output file `xor_model.json`. You can see something like that after running below code:

```bash
{
  "biases": [
    [
      [
        -2.636330932999572
      ],
      [
        -1.7594490804794753
      ],
      [
        3.0830092937660716
      ]
    ],
    [
      [
        0.12700452870944304
      ]
    ]
  ],
  "weights": [
    [
      [
        6.385657890999971,
        -8.475612531245003
      ],
      [
        6.235605000994422,
        6.066682810794803
      ],
      [
        8.860546273906039,
        -7.110813074781834
      ]
    ],
    [
      [
        5.646351512598119,
        2.7936163870997057,
        -5.536782865352909
      ]
    ]
  ]
}
```

### Special thanks
Thanks for author - Open Source contributor - of forked [repo.](https://github.com/neowsl/rust-ml) for inspiration and baseline to start this working implementation (original repository code doesn't work - it was PoC only!).
