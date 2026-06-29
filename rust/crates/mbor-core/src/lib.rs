//! mbor-core: the algorithmic core of Multi-level Bi-objective Routing (MBOR).
//!
//! This crate is an independent Rust reimplementation of the bi-objective
//! routing primitives from "Towards Pareto-optimality with Multi-level
//! Bi-objective Routing: A Summary of Results" (Yang, Zeng, Sharma, Sawamura,
//! Northrop, Shekhar; IWCTS'24). It provides a CSR graph, Pareto-frontier
//! utilities, and the lexicographic label-setting search (Algorithm 2 of the
//! paper; the BOA* / Bi-Objective-Dijkstra family).
//!
//! Upstream reference implementation (C++/C): https://github.com/yang-mingzhou/MBOR

pub mod graph;
pub mod label_setting;
pub mod pareto;

pub use graph::{Cost, Graph};
pub use label_setting::{pareto_costs, pareto_search, ParetoPath};
pub use pareto::{insert_nondominated, minkowski_sum, pareto_filter};
