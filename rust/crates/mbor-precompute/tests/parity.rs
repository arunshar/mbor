//! Parity: the partition-based MEPFV online retrieval must reproduce, exactly,
//! the Pareto frontier from `mbor-core`'s exact full-graph bi-objective search,
//! across many partition counts and on both the paper toy and synthetic grids.

use mbor_core::graph::{Cost, Graph};
use mbor_core::label_setting::pareto_costs;
use mbor_precompute::Mepfv;

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

/// Build a `w x h` grid road-graph with bidirectional edges and two
/// deterministic non-negative costs per edge.
fn grid(w: usize, h: usize) -> Graph {
    let n = w * h;
    let id = |r: usize, c: usize| (r * w + c) as u32;
    let mut edges = Vec::new();
    let cost = |a: u32, b: u32| {
        let s = a as i64 + b as i64;
        Cost::new((s * 7 + 3) % 9 + 1, (s * 5 + 2) % 11 + 1)
    };
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
    Graph::from_edges(n, edges)
}

fn assert_full_parity(g: &Graph, k: usize) {
    let mepfv = Mepfv::build_contiguous(g, k);
    let n = g.num_nodes();
    for o in 0..n {
        for d in 0..n {
            let got = mepfv.query(o, d);
            let want = pareto_costs(g, o, d);
            assert_eq!(
                got, want,
                "mismatch o={o} d={d} k={k}: mepfv={got:?} exact={want:?}"
            );
        }
    }
}

#[test]
fn toy_query_matches_paper_and_exact() {
    let g = Graph::from_dimacs_str(TOY).unwrap();
    let mepfv = Mepfv::build_contiguous(&g, 3);
    // Paper Figure 1.
    assert_eq!(
        mepfv.query(0, 7),
        vec![Cost::new(10, 17), Cost::new(11, 16)]
    );
}

#[test]
fn toy_full_parity_across_partitions() {
    let g = Graph::from_dimacs_str(TOY).unwrap();
    for k in [1, 2, 3, 4, 5, 8] {
        assert_full_parity(&g, k);
    }
}

#[test]
fn grid_full_parity_across_partitions() {
    let g = grid(6, 6); // 36 nodes
    for k in [2, 3, 5, 7, 12] {
        assert_full_parity(&g, k);
    }
}

#[test]
fn grid_rectangular_parity() {
    let g = grid(8, 4); // 32 nodes, asymmetric
    for k in [3, 6, 10] {
        assert_full_parity(&g, k);
    }
}

#[test]
fn boundary_count_is_sane() {
    let g = grid(6, 6);
    let mepfv = Mepfv::build_contiguous(&g, 4);
    let nb = mepfv.num_boundary();
    assert!(nb > 0 && nb < 36, "boundary nodes: {nb}");
}

#[test]
fn single_fragment_has_no_boundary() {
    // k = 1: everything in one fragment, no cross-fragment edges.
    let g = grid(5, 5);
    let mepfv = Mepfv::build_contiguous(&g, 1);
    assert_eq!(mepfv.num_boundary(), 0);
    // Still answers correctly via the within-fragment direct path.
    assert_eq!(mepfv.query(0, 24), pareto_costs(&g, 0, 24));
}

fn assert_adv_parity(g: &Graph, k: usize) {
    let mepfv = Mepfv::build_contiguous(g, k);
    let n = g.num_nodes();
    for o in 0..n {
        for d in 0..n {
            let basic = mepfv.query(o, d);
            let (adv, _) = mepfv.query_adv(o, d);
            let exact = pareto_costs(g, o, d);
            assert_eq!(adv, basic, "adv != basic o={o} d={d} k={k}");
            assert_eq!(adv, exact, "adv != exact o={o} d={d} k={k}");
        }
    }
}

#[test]
fn adv_matches_basic_and_exact() {
    let toy = Graph::from_dimacs_str(TOY).unwrap();
    for k in [2, 3, 5, 8] {
        assert_adv_parity(&toy, k);
    }
    let g = grid(6, 6);
    for k in [3, 5, 9] {
        assert_adv_parity(&g, k);
    }
}

#[test]
fn adv_actually_prunes() {
    // On a larger grid with many boundary pairs, 2DCI pruning must skip some.
    let g = grid(7, 7); // 49 nodes
    let mepfv = Mepfv::build_contiguous(&g, 9);
    let (mut pruned, mut total) = (0usize, 0usize);
    for o in 0..49 {
        for d in 0..49 {
            let (_, s) = mepfv.query_adv(o, d);
            pruned += s.pairs_pruned;
            total += s.pairs_total;
        }
    }
    assert!(total > 0);
    assert!(pruned > 0, "expected 2DCI pruning, got {pruned}/{total}");
}
