//! Parity tests against the paper's Figure 1 toy network (the upstream `test`
//! map). The Pareto frontier for n0 -> n7 is `{(10,17), (11,16)}`.

use mbor_core::graph::{Cost, Graph};
use mbor_core::label_setting::{pareto_costs, pareto_search};
use mbor_core::pareto::{insert_nondominated, minkowski_sum, pareto_filter};

const TOY: &str = include_str!("data/toy.txt");

#[test]
fn toy_n0_to_n7_matches_paper_figure1() {
    let g = Graph::from_dimacs_str(TOY).unwrap();
    assert_eq!(g.num_nodes(), 8);
    assert_eq!(g.num_edges(), 20);
    // File node i == paper node n_{i-1}; query "1 8" is n0 -> n7.
    let costs = pareto_costs(&g, 0, 7);
    assert_eq!(costs, vec![Cost::new(10, 17), Cost::new(11, 16)]);
}

#[test]
fn toy_paths_are_the_paper_paths() {
    let g = Graph::from_dimacs_str(TOY).unwrap();
    let sols = pareto_search(&g, 0, 7);
    let paths: Vec<Vec<u32>> = sols.iter().map(|p| p.path.clone()).collect();
    // [n0,n3,n5,n7] = 0,3,5,7 ; [n0,n1,n3,n5,n7] = 0,1,3,5,7
    assert!(paths.contains(&vec![0, 3, 5, 7]), "paths: {paths:?}");
    assert!(paths.contains(&vec![0, 1, 3, 5, 7]), "paths: {paths:?}");
}

#[test]
fn path_costs_sum_their_edges() {
    let g = Graph::from_dimacs_str(TOY).unwrap();
    for sol in pareto_search(&g, 0, 7) {
        // Recompute the cost from the path and confirm it matches the label.
        let mut acc = Cost::ZERO;
        for w in sol.path.windows(2) {
            let (from, to) = (w[0] as usize, w[1]);
            let (_, edge) = g
                .neighbors(from)
                .find(|&(v, _)| v as u32 == to)
                .expect("path edge must exist");
            acc += edge;
        }
        assert_eq!(acc, sol.cost);
    }
}

#[test]
fn same_node_query_is_trivial() {
    let g = Graph::from_dimacs_str(TOY).unwrap();
    assert_eq!(pareto_costs(&g, 2, 2), vec![Cost::new(0, 0)]);
}

#[test]
fn dominance_rules() {
    assert!(Cost::new(10, 17).dominates(Cost::new(12, 18)));
    assert!(Cost::new(10, 17).dominates(Cost::new(10, 18)));
    assert!(!Cost::new(10, 17).dominates(Cost::new(11, 16)));
    assert!(!Cost::new(10, 17).dominates(Cost::new(10, 17))); // not strict
}

#[test]
fn pareto_filter_drops_dominated_and_duplicates() {
    let f = pareto_filter(vec![
        Cost::new(10, 17),
        Cost::new(11, 16),
        Cost::new(12, 18), // dominated by (10,17)
        Cost::new(11, 16), // duplicate
    ]);
    assert_eq!(f, vec![Cost::new(10, 17), Cost::new(11, 16)]);
}

#[test]
fn insert_nondominated_maintains_frontier() {
    let mut f = Vec::new();
    assert!(insert_nondominated(&mut f, Cost::new(10, 17)));
    assert!(insert_nondominated(&mut f, Cost::new(11, 16)));
    assert!(!insert_nondominated(&mut f, Cost::new(12, 18))); // dominated
    assert!(!insert_nondominated(&mut f, Cost::new(11, 16))); // duplicate
    assert!(insert_nondominated(&mut f, Cost::new(9, 20))); // new non-dominated
    f.sort_by_key(|c| c.c1);
    assert_eq!(
        f,
        vec![Cost::new(9, 20), Cost::new(10, 17), Cost::new(11, 16)]
    );
}

#[test]
fn minkowski_combines_two_frontiers() {
    let a = vec![Cost::new(1, 3), Cost::new(2, 1)];
    let b = vec![Cost::new(2, 2), Cost::new(0, 5)];
    // sums: (3,5),(1,8),(4,3),(2,6); all mutually non-dominated.
    let m = minkowski_sum(&a, &b);
    assert_eq!(
        m,
        vec![
            Cost::new(1, 8),
            Cost::new(2, 6),
            Cost::new(3, 5),
            Cost::new(4, 3),
        ]
    );
}
