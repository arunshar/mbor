//! Speedup benchmark for MBOR online retrieval vs compute-on-demand baselines.
//!
//! Usage:
//!   mbor-bench <map.txt> <queries> [k=50] [reps=3] [partition_file]
//!
//! Builds the MEPFV (timed), then for each query runs BOD, BOA*, MBOR-Basic, and
//! MBOR-Adv, verifies all four return the identical Pareto frontier (exactness),
//! and reports average per-query time and the MBOR speedups, plus the Adv 2DCI
//! pruning rate. Partition: a KaHIP `kaffpaIndex.txt` if given, else a BFS
//! region-growing partition.

use std::time::Instant;

use mbor_baseline::{boa_star, bod};
use mbor_core::Graph;
use mbor_precompute::{bfs_partition, load_partition, Mepfv};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: mbor-bench <map> <queries> [k=50] [reps=3] [partition_file]");
        std::process::exit(2);
    }
    let map = &args[1];
    let queries_path = &args[2];
    let k: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(50);
    let reps: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(3);

    let g = Graph::from_dimacs_file(map).expect("load map");

    let (part, part_kind) = match args.get(5) {
        Some(pf) => (load_partition(pf).expect("load partition"), "file"),
        None => (bfs_partition(&g, k), "bfs"),
    };
    let real_k = (*part.iter().max().unwrap_or(&0) as usize) + 1;

    let t = Instant::now();
    let mepfv = Mepfv::build(&g, part);
    let precompute_ms = t.elapsed().as_secs_f64() * 1000.0;

    let qtext = std::fs::read_to_string(queries_path).expect("load queries");
    let queries: Vec<(usize, usize)> = qtext
        .lines()
        .filter_map(|l| {
            let mut it = l.split_whitespace();
            let o: usize = it.next()?.parse().ok()?;
            let d: usize = it.next()?.parse().ok()?;
            Some((o - 1, d - 1)) // queries are 1-indexed
        })
        .collect();

    // Optional: export the batched boundary-pair workload (A,B,C cost-sets) and
    // the largest fragment for the GPU kernels / precompute, then exit.
    if let Ok(dir) = std::env::var("MBOR_EXPORT") {
        export_workload(&mepfv, &queries, &dir);
        return;
    }

    // Exactness check (once): all four methods must agree on every query.
    let mut mismatches = 0usize;
    let mut sol_total = 0usize;
    let (mut pairs, mut pruned) = (0usize, 0usize);
    for &(o, d) in &queries {
        let r_bod = bod(&g, o, d);
        let r_boa = boa_star(&g, o, d);
        let r_basic = mepfv.query(o, d);
        let (r_adv, st) = mepfv.query_adv(o, d);
        sol_total += r_adv.len();
        pairs += st.pairs_total;
        pruned += st.pairs_pruned;
        if !(r_bod == r_boa && r_boa == r_basic && r_basic == r_adv) {
            mismatches += 1;
        }
    }

    // Timing: take the min total over `reps` passes for each method.
    let mut best = [f64::MAX; 4]; // bod, boa, basic, adv
    for _ in 0..reps {
        let mut acc = [0f64; 4];
        for &(o, d) in &queries {
            let t0 = Instant::now();
            let _ = bod(&g, o, d);
            acc[0] += t0.elapsed().as_secs_f64();
            let t1 = Instant::now();
            let _ = boa_star(&g, o, d);
            acc[1] += t1.elapsed().as_secs_f64();
            let t2 = Instant::now();
            let _ = mepfv.query(o, d);
            acc[2] += t2.elapsed().as_secs_f64();
            let t3 = Instant::now();
            let _ = mepfv.query_adv(o, d);
            acc[3] += t3.elapsed().as_secs_f64();
        }
        for i in 0..4 {
            best[i] = best[i].min(acc[i]);
        }
    }
    let nq = queries.len() as f64;
    let ms = |s: f64| s / nq * 1000.0;

    println!("== MBOR speedup benchmark ==");
    println!(
        "map={map}  nodes={}  edges={}  partition={part_kind}(k={real_k})  boundary_nodes={}",
        g.num_nodes(),
        g.num_edges(),
        mepfv.num_boundary()
    );
    println!("precompute_time = {precompute_ms:.1} ms");
    println!(
        "queries={}  avg_pareto_solutions={:.1}  exactness_mismatches={mismatches}",
        queries.len(),
        sol_total as f64 / nq
    );
    println!("--- avg per-query online time (ms), min of {reps} passes ---");
    println!("  BOD (bi-objective Dijkstra) : {:.4}", ms(best[0]));
    println!("  BOA*                        : {:.4}", ms(best[1]));
    println!("  MBOR-Basic                  : {:.4}", ms(best[2]));
    println!("  MBOR-Adv                    : {:.4}", ms(best[3]));
    println!("--- speedup (online) ---");
    println!("  MBOR-Basic vs BOA* : {:.1}x", best[1] / best[2]);
    println!("  MBOR-Adv   vs BOA* : {:.1}x", best[1] / best[3]);
    println!("  MBOR-Adv   vs BOD  : {:.1}x", best[0] / best[3]);
    println!("  MBOR-Adv   vs MBOR-Basic : {:.2}x", best[2] / best[3]);
    println!(
        "Adv 2DCI pruning: {pruned}/{pairs} boundary-pair combinations pruned ({:.0}%)",
        if pairs > 0 {
            100.0 * pruned as f64 / pairs as f64
        } else {
            0.0
        }
    );
}

fn costs_json(out: &mut String, costs: &[mbor_core::graph::Cost]) {
    out.push('[');
    for (i, c) in costs.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&format!("[{},{}]", c.c1, c.c2));
    }
    out.push(']');
}

/// Write `pairs.json` (all boundary-pair A,B,C cost-sets across the queries) and
/// `frag.json` (the largest fragment's edge list + boundary source locals) for
/// the GPU kernels and the speculative GPU precompute.
fn export_workload(mepfv: &Mepfv, queries: &[(usize, usize)], dir: &str) {
    std::fs::create_dir_all(dir).expect("mkdir export");
    let mut all = Vec::new();
    for &(o, d) in queries {
        all.extend(mepfv.export_pairs(o, d));
    }
    let mut s = String::from("[");
    for (i, (a, b, c)) in all.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str("{\"a\":");
        costs_json(&mut s, a);
        s.push_str(",\"b\":");
        costs_json(&mut s, b);
        s.push_str(",\"c\":");
        costs_json(&mut s, c);
        s.push('}');
    }
    s.push(']');
    std::fs::write(format!("{dir}/pairs.json"), s).expect("write pairs");

    let (n, edges, srcs) = mepfv.largest_fragment();
    let mut fs = format!("{{\"n\":{n},\"edges\":[");
    for (i, (u, v, c1, c2)) in edges.iter().enumerate() {
        if i > 0 {
            fs.push(',');
        }
        fs.push_str(&format!("[{u},{v},{c1},{c2}]"));
    }
    fs.push_str("],\"sources\":[");
    for (i, sc) in srcs.iter().enumerate() {
        if i > 0 {
            fs.push(',');
        }
        fs.push_str(&sc.to_string());
    }
    fs.push_str("]}");
    std::fs::write(format!("{dir}/frag.json"), fs).expect("write frag");
    eprintln!(
        "exported {} boundary-pair triples + fragment (n={}, {} edges, {} sources) to {}",
        all.len(),
        n,
        edges.len(),
        srcs.len(),
        dir
    );
}
