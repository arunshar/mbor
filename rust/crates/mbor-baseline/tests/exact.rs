//! Both baselines must return the exact Pareto frontier (== mbor-core), and
//! agree with each other.

use mbor_baseline::{boa_star, bod};
use mbor_core::graph::{Cost, Graph};
use mbor_core::label_setting::pareto_costs;

const TOY: &str = "\
8 20
1 2 1 3
1 3 7 8
1 4 3 6
2 1 1 3
2 4 3 2
3 1 7 8
3 5 1 2
4 1 3 6
4 2 3 2
4 6 5 6
5 3 1 2
5 7 9 7
6 4 5 6
6 7 2 3
6 8 2 5
7 5 9 7
7 6 2 3
7 8 1 3
8 6 2 5
8 7 1 3
";

fn grid(w: usize, h: usize) -> Graph {
    let id = |r: usize, c: usize| (r * w + c) as u32;
    let cost = |a: u32, b: u32| {
        let s = a as i64 + b as i64;
        Cost::new((s * 7 + 3) % 9 + 1, (s * 5 + 2) % 11 + 1)
    };
    let mut edges = Vec::new();
    for r in 0..h {
        for c in 0..w {
            let u = id(r, c);
            if c + 1 < w {
                let v = id(r, c + 1);
                edges.push((u, v, cost(u, v)));
                edges.push((v, u, cost(v, u)));
            }
            if r + 1 < h {
                let v = id(r + 1, c);
                edges.push((u, v, cost(u, v)));
                edges.push((v, u, cost(v, u)));
            }
        }
    }
    Graph::from_edges(w * h, edges)
}

#[test]
fn toy_baselines_match_exact_and_paper() {
    let g = Graph::from_dimacs_str(TOY).unwrap();
    assert_eq!(
        boa_star(&g, 0, 7),
        vec![Cost::new(10, 17), Cost::new(11, 16)]
    );
    assert_eq!(bod(&g, 0, 7), vec![Cost::new(10, 17), Cost::new(11, 16)]);
}

#[test]
fn baselines_match_exact_on_all_pairs() {
    for g in [Graph::from_dimacs_str(TOY).unwrap(), grid(6, 6), grid(8, 4)] {
        let n = g.num_nodes();
        for o in 0..n {
            for d in 0..n {
                if o == d {
                    continue;
                }
                let exact = pareto_costs(&g, o, d);
                assert_eq!(boa_star(&g, o, d), exact, "boa* o={o} d={d}");
                assert_eq!(bod(&g, o, d), exact, "bod o={o} d={d}");
            }
        }
    }
}
