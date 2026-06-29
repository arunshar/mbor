//! Criterion micro-benchmark for the label-setting core on the toy network.
//! Scaled BAY benchmarks live in `mbor-online` (Phase G3).

use criterion::{criterion_group, criterion_main, Criterion};
use mbor_core::graph::Graph;
use mbor_core::label_setting::pareto_costs;

const TOY: &str = include_str!("../tests/data/toy.txt");

fn bench_toy(c: &mut Criterion) {
    let g = Graph::from_dimacs_str(TOY).unwrap();
    c.bench_function("pareto_search_toy_n0_to_n7", |b| {
        b.iter(|| pareto_costs(&g, 0, 7))
    });
}

criterion_group!(benches, bench_toy);
criterion_main!(benches);
