use markovka::MarkovChain;
use rand::{SeedableRng, rngs::StdRng};
use std::hint::black_box;
use std::num::NonZeroUsize;
use std::time::{Duration, Instant};

fn measure(name: &str, mut run: impl FnMut() -> usize) {
    let mut iterations = 1;
    loop {
        let start = Instant::now();
        for _ in 0..iterations {
            black_box(run());
        }
        if start.elapsed() >= Duration::from_millis(100) {
            break;
        }
        iterations *= 2;
    }
    let mut samples = [(0.0, 0.0); 5];
    for sample in &mut samples {
        let start = Instant::now();
        let mut words = 0;
        for _ in 0..iterations {
            words += black_box(run());
        }
        let elapsed = start.elapsed().as_nanos() as f64;
        *sample = (elapsed / iterations as f64, words as f64 / iterations as f64);
    }
    samples.sort_by(|a, b| a.0.total_cmp(&b.0));
    let (ns_per_op, words_per_op) = samples[2];
    println!(
        "{name:26} {ns_per_op:>10.1} ns/op  {words_per_op:>6.1} words/op  {:>8.1} ns/word",
        ns_per_op / words_per_op
    );
}

fn main() {
    let order = NonZeroUsize::new(2).unwrap();
    let repeated: Vec<_> = (0..1_000).map(|i| format!("word{}", i % 16)).collect();
    let diverse: Vec<_> = (0..1_000).map(|i| format!("word{i}")).collect();

    measure("train repeated 1000", || {
        let mut chain = MarkovChain::new(order);
        chain.train(black_box(repeated.iter().map(String::as_str)));
        black_box(chain);
        repeated.len()
    });
    measure("train diverse 1000", || {
        let mut chain = MarkovChain::new(order);
        chain.train(black_box(diverse.iter().map(String::as_str)));
        black_box(chain);
        diverse.len()
    });
    measure("train order 32 1000", || {
        let mut chain = MarkovChain::new(NonZeroUsize::new(32).unwrap());
        chain.train(black_box(repeated.iter().map(String::as_str)));
        black_box(chain);
        repeated.len()
    });

    let mut repeated_chain = MarkovChain::new(order);
    repeated_chain.train(repeated.iter().map(String::as_str));
    let mut unique_chain = MarkovChain::new(order);
    unique_chain.train(diverse.iter().map(String::as_str));
    let mut branching_chain = MarkovChain::new(NonZeroUsize::new(1).unwrap());
    for word in &diverse {
        branching_chain.train(["root", word]);
    }
    let mut rng = StdRng::seed_from_u64(42);
    measure("generate repeated max 100", || {
        black_box(repeated_chain.generate(&mut rng, black_box(100))).len()
    });
    measure("generate unique 100", || {
        black_box(unique_chain.generate(&mut rng, black_box(100))).len()
    });
    measure("generate 1000-way choice", || {
        black_box(branching_chain.generate(&mut rng, black_box(1_000))).len()
    });
}
