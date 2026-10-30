# `markovka`

a word-based markov chain for generating text

```rust
use markovka::MarkovChain;
use std::num::NonZeroUsize;

let mut chain = MarkovChain::new(NonZeroUsize::new(2).unwrap());
chain.train_text("the cat sleeps on the sofa");
chain.train_text("the cat sleeps on the windowsill");

println!("{}", chain.generate(&mut rand::rng(), 20).join(" "));
```

to continue training later, use `save_to_path`, then `load_from_path` and `train_text`