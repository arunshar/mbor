//! Compressed-sparse-row (CSR) directed graph with two non-negative integer
//! costs per edge, plus a loader for the MBOR `*-road-d.txt` map format.

use std::fs;
use std::path::Path;

/// A pair of non-negative path/edge costs `(c1, c2)` (e.g. distance, travel
/// time). The whole bi-objective problem is defined over these pairs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Cost {
    pub c1: i64,
    pub c2: i64,
}

impl Cost {
    pub const ZERO: Cost = Cost { c1: 0, c2: 0 };

    #[inline]
    pub fn new(c1: i64, c2: i64) -> Self {
        Cost { c1, c2 }
    }

    /// Weak Pareto dominance per the paper (Def. 2.3): `self` dominates `other`
    /// when both components are `<=` and the pair is not identical.
    #[inline]
    pub fn dominates(self, other: Cost) -> bool {
        self.c1 <= other.c1 && self.c2 <= other.c2 && self != other
    }
}

impl std::ops::Add for Cost {
    type Output = Cost;

    /// Component-wise addition: a path cost is the sum of its edge costs.
    #[inline]
    fn add(self, other: Cost) -> Cost {
        Cost {
            c1: self.c1 + other.c1,
            c2: self.c2 + other.c2,
        }
    }
}

impl std::ops::AddAssign for Cost {
    #[inline]
    fn add_assign(&mut self, other: Cost) {
        self.c1 += other.c1;
        self.c2 += other.c2;
    }
}

/// Directed graph in CSR layout. Nodes are stored 0-indexed internally; the
/// DIMACS loader converts the file's 1-indexed ids on the way in.
#[derive(Clone, Debug)]
pub struct Graph {
    n: usize,
    /// `head[u]..head[u + 1]` is the slice of out-edges for node `u`.
    head: Vec<u32>,
    to: Vec<u32>,
    cost: Vec<Cost>,
}

impl Graph {
    pub fn num_nodes(&self) -> usize {
        self.n
    }

    pub fn num_edges(&self) -> usize {
        self.to.len()
    }

    /// Iterate the outgoing edges of `u` as `(neighbor, cost)`.
    #[inline]
    pub fn neighbors(&self, u: usize) -> impl Iterator<Item = (usize, Cost)> + '_ {
        let lo = self.head[u] as usize;
        let hi = self.head[u + 1] as usize;
        (lo..hi).map(move |e| (self.to[e] as usize, self.cost[e]))
    }

    /// Build from a 0-indexed edge list `(from, to, cost)` over `n` nodes.
    pub fn from_edges(n: usize, mut edges: Vec<(u32, u32, Cost)>) -> Graph {
        edges.sort_by_key(|&(u, _, _)| u);
        let mut head = vec![0u32; n + 1];
        for &(u, _, _) in &edges {
            head[u as usize + 1] += 1;
        }
        for i in 0..n {
            head[i + 1] += head[i];
        }
        let mut to = vec![0u32; edges.len()];
        let mut cost = vec![Cost::ZERO; edges.len()];
        for (i, &(_, v, c)) in edges.iter().enumerate() {
            to[i] = v;
            cost[i] = c;
        }
        Graph { n, head, to, cost }
    }

    /// Parse the MBOR `*-road-d.txt` format: the first line is
    /// `<num_nodes> <num_edges>`, then one edge per line as
    /// `<from> <to> <c1> <c2>` with 1-indexed node ids. The returned graph is
    /// 0-indexed. The header edge count is advisory and not enforced.
    pub fn from_dimacs_str(s: &str) -> Result<Graph, String> {
        let mut lines = s.lines().filter(|l| !l.trim().is_empty());
        let header = lines.next().ok_or("empty input")?;
        let mut h = header.split_whitespace();
        let n: usize = h
            .next()
            .ok_or("missing node count")?
            .parse()
            .map_err(|e| format!("bad node count: {e}"))?;
        let _m: usize = h
            .next()
            .ok_or("missing edge count")?
            .parse()
            .map_err(|e| format!("bad edge count: {e}"))?;

        let mut edges = Vec::new();
        for (ln, line) in lines.enumerate() {
            let mut f = line.split_whitespace();
            let mut next = |what: &str| -> Result<i64, String> {
                f.next()
                    .ok_or_else(|| format!("line {ln}: missing {what}"))?
                    .parse()
                    .map_err(|e| format!("line {ln}: bad {what}: {e}"))
            };
            let u = next("from")?;
            let v = next("to")?;
            let c1 = next("c1")?;
            let c2 = next("c2")?;
            if u < 1 || v < 1 {
                return Err(format!("line {ln}: node ids must be 1-indexed"));
            }
            if (u as usize) > n || (v as usize) > n {
                return Err(format!("line {ln}: node id exceeds declared count {n}"));
            }
            if c1 < 0 || c2 < 0 {
                return Err(format!("line {ln}: costs must be non-negative"));
            }
            edges.push(((u - 1) as u32, (v - 1) as u32, Cost::new(c1, c2)));
        }
        Ok(Graph::from_edges(n, edges))
    }

    pub fn from_dimacs_file<P: AsRef<Path>>(path: P) -> Result<Graph, String> {
        let s = fs::read_to_string(path).map_err(|e| e.to_string())?;
        Graph::from_dimacs_str(&s)
    }
}
