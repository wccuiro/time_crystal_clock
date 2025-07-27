use ndarray::{array, Array1, Array2};

use rayon::prelude::*;


use std::io::{BufReader, Read, ErrorKind};

use std::{fs::File, path::Path};
use zstd::stream::read::Decoder;




fn bin_width(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 1.0; // Default bin size if no data
    }

    fn quantile(data: &[f64], prob: f64) -> f64 {
        let n = data.len();
        let idx = prob * (n - 1) as f64;
        let lo = idx.floor() as usize;
        let hi = idx.ceil() as usize;
        if lo == hi {
            data[lo]
        } else {
            let frac = idx - lo as f64;
            data[lo] * (1.0 - frac) + data[hi] * frac
        }
    }

    let q1 = quantile(data, 0.25);
    let q3 = quantile(data, 0.75);
    let iqr = q3 - q1;
    let n = data.len() as f64;

    (2.0 * iqr) / n.cbrt()
}    

fn counts_per_bin(
    data: &[f64],
    bin_width: f64,
    min: f64,
    max: f64,
) -> Vec<f64> {
    // how many bins?
    let num_bins = ((max - min) / bin_width).ceil() as usize;
    let mut counts = vec![0usize; num_bins];

    for &v in data {
        if v >= min && v <= max {
            // floor to get bin index
            let mut idx = ((v - min) / bin_width).floor() as isize;
            // clamp exact‐max into the last bin
            if idx == num_bins as isize {
                idx = num_bins as isize - 1;
            }
            if (0..num_bins as isize).contains(&idx) {
                counts[idx as usize] += 1;
            }
        }
    }

    let total_area = counts.iter().sum::<usize>() as f64 * bin_width;
    counts
        .into_iter()
        .map(|c| c as f64 / total_area)
        .collect()
}


// Results struct to organize outputs
#[derive(Debug)]
struct SimulationResults {
    counts_n: Vec<f64>,
    counts_k: Vec<f64>,
    counts_q: Vec<f64>,
    bin_width_n: [f64;3],
    bin_width_k: [f64;3],
    bin_width_q: [f64;3],
    num_ticks_n: usize,
    num_ticks_k: usize,
    num_ticks_q: usize,
    entropy_tick_n: f64,
    entropy_tick_k: f64,
    entropy_tick_q: f64,
    exp_entropy_tick_n: f64,
    exp_entropy_tick_k: f64,
    exp_entropy_tick_q: f64,
    exp_entropy_mar_n: f64,
    exp_entropy_mar_k: f64,
    exp_entropy_mar_q: f64,
    accuracy_n: f64,
    accuracy_k: f64,
    accuracy_q: f64,
    resolution_n: f64,
    resolution_k: f64,
    resolution_q: f64,
    activity_tick_n: f64,
    activity_tick_k: f64,
    activity_tick_q: f64,
}


fn find_first_peak(y: &[f64], window: usize, min_prominence: f64) -> Option<usize> {
    if y.len() < 2 * window + 1 {
        return None; // not enough data
    }

    for i in window..y.len() - window {
        let center = y[i];

        let left_max = y[i - window..i].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let right_max = y[i + 1..=i + window].iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        if center > left_max && center > right_max {
            let prominence = center - f64::max(left_max, right_max);
            if prominence >= min_prominence {
                return Some(i);
            }
        }
    }
    None
}



fn optimal_threshold(
    vec_lambda: &Vec<f64>,
    vec_s: &Vec<f64>,
    max_files: usize,
) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let n_pts = 1000;
    let init_m = 5.0_f64;
    let last_m = 2000.0_f64;

    let m_values: Vec<f64> = if n_pts == 1 {
        vec![init_m]
    } else {
        (0..n_pts)
            .map(|i| {
                let t = i as f64 / (n_pts - 1) as f64;
                (init_m + t * (last_m - init_m)).round()
            })
            .collect()
    };

    let mut optimal_m = Vec::new();
    let jump_type = 0_u8; // <-- you should set this appropriately
    let base_dir = "output"; // <-- update if needed

    for &lambda in vec_lambda {
        for &s in vec_s {
            let mut sum    = vec![0.0; m_values.len()];
            let mut sum_sq = vec![0.0; m_values.len()];
            let mut cnt    = vec![0;   m_values.len()];

            let betawc = 0.1;

            let s_int = s as i64;
            for i in 0..max_files {
                let dir_idx = i / 1_000;
                let subdir = format!("{}/l{:.2}_s{}_b{:.2}/{:05}", base_dir, lambda, betawc, s_int, dir_idx);
                let traj = format!("{}/traj_{:05}.zst", subdir, i);
                let path = Path::new(&traj);
                if !path.exists() { break; }

                let file = File::open(path)?;
                let mut dec = Decoder::new(BufReader::new(file))?;
                let mut times = Vec::new();
                loop {
                    let mut buf = [0u8; 1 + 8 + 8];
                    match dec.read_exact(&mut buf) {
                        Ok(()) => {
                            let id = buf[0];
                            let t = f64::from_le_bytes(buf[1..9].try_into().unwrap());
                            if id == jump_type {
                                times.push(t);
                            }
                        }
                        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                        Err(e) => return Err(Box::new(e)),
                    }
                }

                if times.len() <= 1 { continue; }
                let slice = &times[1..];

                for (j, &m) in m_values.iter().enumerate() {
                    let chunks = slice.len() as i32 / m as i32;
                    if chunks == 0 { continue; }

                    let mut last = 0.0;
                    for k in 1..=chunks {
                        let idx = (k * m as i32 - 1) as usize;
                        let delta = slice[idx] - last;
                        last = slice[idx];
                        sum[j] += delta;
                        sum_sq[j] += delta * delta;
                    }
                    cnt[j] += chunks;
                }
            }

            let mut acts = Vec::with_capacity(m_values.len());
            for j in 0..m_values.len() {
                if cnt[j] == 0 {
                    acts.push(0.0);
                } else {
                    let n = cnt[j] as f64;
                    let mu = sum[j] / n;
                    let var = sum_sq[j] / n - mu * mu;
                    acts.push(if var > 0.0 { mu * mu / var } else { 0.0 });
                }
            }

            // println!("{:?}", m_values);
            // println!("{:?}", acts);            

            match find_first_peak(&acts,5,0.1) {
                Some(index) => optimal_m.push(m_values[index]),
                None => return Err("No local maximum found.".into()),
            }
        }
    }

    Ok(optimal_m)
}

fn analyze_data(
    lambda: f64,
    s: f64,
    num_trajectories: usize,
    m: f64,
) -> Result<SimulationResults, Box<dyn std::error::Error>> {
    let base_dir = "output"; // <-- update if needed

    let mut ticks_n_set: Vec<f64> = Vec::new();
    let mut ticks_k_set: Vec<f64> = Vec::new();
    let mut ticks_q_set: Vec<f64> = Vec::new();

    let mut entropy_tick_n_sum = 0.0;
    let mut entropy_tick_k_sum = 0.0;
    let mut entropy_tick_q_sum = 0.0;

    let mut activity_tick_n_sum = 0.0;
    let mut activity_tick_k_sum = 0.0;
    let mut activity_tick_q_sum = 0.0;

    let mut exp_entropy_mar_n_sum = 0.0;
    let mut exp_entropy_mar_k_sum = 0.0;
    let mut exp_entropy_mar_q_sum = 0.0;

    let mut exp_entropy_tick_n_sum  = 0.0;
    let mut exp_entropy_tick_k_sum  = 0.0;
    let mut exp_entropy_tick_q_sum  = 0.0;

    let s_int = s as i64;
    for i in 0..num_trajectories {
        let dir_idx = i / 1_000;
        let subdir = format!("{}/l{:.2}_s{}_b0.10/{:05}", base_dir, lambda, s_int, dir_idx);
        let traj = format!("{}/traj_{:05}.zst", subdir, i);
        let path = Path::new(&traj);
        if !path.exists() { break; }

        let file = File::open(path)?;
        let mut dec = Decoder::new(BufReader::new(file))?;

        let mut buf = [0u8; 1 + 8 + 8];

        // --- 1) read & sample the very first record (line 0), but don’t push absolute times/entropies ---
        dec.read_exact(&mut buf)?;
        let _id0 = buf[0];
        let t0  = f64::from_le_bytes(buf[1..9].try_into().unwrap());
        let e0  = f64::from_le_bytes(buf[9..17].try_into().unwrap());

        // initialize “last seen” for each series at the line‑0 values
        let (mut last_t_k, mut last_e_k, mut last_a_k) = (t0, e0, 0usize);
        let (mut last_t_n, mut last_e_n, mut last_a_n) = (t0, e0, 0usize);
        let (mut last_t_q, mut last_e_q, mut last_a_q) = (t0, e0, 0usize);

        // println!("initial{}",t0);

        // output Δ‑arrays
        let mut ticks_k = Vec::new();
        let mut entropys_tick_k = Vec::new();
        let mut activity_tick_k = Vec::new();

        let mut ticks_n = Vec::new();
        let mut entropys_tick_n = Vec::new();
        let mut activity_tick_n = Vec::new();

        let mut ticks_q = Vec::new();
        let mut entropys_tick_q = Vec::new();
        let mut activity_tick_q = Vec::new();

        // --- 2) initialize counters and next‑thresholds (record 0 not counted) ---
        let mut count_total = 0;
        let mut count_zero  = 0;
        let mut _count_one   = 0;
        let mut diff        = 0isize;

        let mut next_k = m as usize;          // sample when total jumps == m, 2m, …
        let mut next_n = m as usize;          // sample when zero‑jumps == m, 2m, …
        let mut next_q = m as usize; // sample when diff == m, 2m, …

        // --- 3) now loop over the rest and push Δ’s when thresholds hit ---
        loop {
            match dec.read_exact(&mut buf) {
                Ok(()) => {
                    let id = buf[0];
                    let t  = f64::from_le_bytes(buf[1..9].try_into().unwrap());
                    let e  = f64::from_le_bytes(buf[9..17].try_into().unwrap());

                    // bump counters
                    count_total += 1;
                    if id == 0 { count_zero += 1; diff += 1; }
                    if id == 1 { _count_one  += 1; diff -= 1; }

                    // a) every m total → k
                    if count_total >= next_k {
                        // compute deltas relative to last_k
                        ticks_k.push(          t - last_t_k);
                        entropys_tick_k.push(  e - last_e_k);
                        activity_tick_k.push(count_total - last_a_k);
                        // update last_k
                        last_t_k = t; last_e_k = e; last_a_k = count_total;
                        next_k += m as usize;
                    }

                    // b) every m zeros → n
                    if count_zero >= next_n {
                        ticks_n.push(          t - last_t_n);
                        entropys_tick_n.push(  e - last_e_n);
                        activity_tick_n.push(count_total - last_a_n);
                        // update last_n
                        last_t_n = t; last_e_n = e; last_a_n = count_total;
                        next_n += m as usize;
                    }

                    // c) every m diff → q
                    if diff >= next_q as isize {
                        ticks_q.push(          t - last_t_q);
                        entropys_tick_q.push(  e - last_e_q);
                        activity_tick_q.push(count_total - last_a_q);
                        // update last_q
                        last_t_q = t; last_e_q = e; last_a_q = count_total;
                        next_q += m as usize;
                    }
                }
                Err(e) if e.kind() == ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(Box::new(e)),
            }
        }
        // println!("{}", ticks_n.len());

        // ——— Drop the first element in-place ———
        if ticks_n.len() > 1 {
            let rest = ticks_n.split_off(1);   // removes [t1, t2, …] into `rest`
            ticks_n_set.extend(rest.into_iter());
        }
        if ticks_k.len() > 1 {
            let rest = ticks_k.split_off(1);   // removes [t1, t2, …] into `rest`
            ticks_k_set.extend(rest.into_iter());
        }
        if ticks_q.len() > 1 {
            let rest = ticks_q.split_off(1);   // removes [t1, t2, …] into `rest`
            ticks_q_set.extend(rest.into_iter());
        }

        // If you need to keep the original `activity_tick_*` Vecs elsewhere, clone once:
        activity_tick_n.remove(0);
        activity_tick_k.remove(0);
        activity_tick_q.remove(0);

        // ——— Compute the exp of the “mar” entropy once each ———
        let exp_entropy_mar_n = (-entropys_tick_n[0]).exp();
        let exp_entropy_mar_k = (-entropys_tick_k[0]).exp();
        let exp_entropy_mar_q = (-entropys_tick_q[0]).exp();

        // ——— Now accumulate everything directly over the slices ———
        // Sum of entropies (skipping the 0‑th)
        entropy_tick_n_sum += entropys_tick_n[1..].iter().sum::<f64>();
        entropy_tick_k_sum += entropys_tick_k[1..].iter().sum::<f64>();
        entropy_tick_q_sum += entropys_tick_q[1..].iter().sum::<f64>();

        // Sum of activities (skipping the 0‑th)
        activity_tick_n_sum += activity_tick_n.iter().sum::<usize>() as f64;
        activity_tick_k_sum += activity_tick_k.iter().sum::<usize>() as f64;
        activity_tick_q_sum += activity_tick_q.iter().sum::<usize>() as f64;

        // Sum of the “mar” exps
        exp_entropy_mar_n_sum += exp_entropy_mar_n;
        exp_entropy_mar_k_sum += exp_entropy_mar_k;
        exp_entropy_mar_q_sum += exp_entropy_mar_q;

        // Sum of exp of each tick’s entropy (skipping the 0‑th)
        exp_entropy_tick_n_sum += entropys_tick_n[1..]
            .iter()
            .map(|&e| (-e).exp())
            .sum::<f64>();
        exp_entropy_tick_k_sum += entropys_tick_k[1..]
            .iter()
            .map(|&e| (-e).exp())
            .sum::<f64>();
        exp_entropy_tick_q_sum += entropys_tick_q[1..]
            .iter()
            .map(|&e| (-e).exp())
            .sum::<f64>();
    }

    // num ticks
    let num_ticks_n: f64 = ticks_n_set.len() as f64;
    let num_ticks_k: f64 = ticks_k_set.len() as f64;
    let num_ticks_q: f64 = ticks_q_set.len() as f64;

    // Computing accuracy
    let mean_t_n: f64 = ticks_n_set.iter().sum::<f64>() / num_ticks_n;
    let mean_sq_t_n: f64 = ticks_n_set.iter().map(|&e| e*e).sum::<f64>() / num_ticks_n;
    let var_t_n: f64 = mean_sq_t_n - mean_t_n.powi(2); 
    let accuracy_n: f64 = mean_t_n.powi(2) / var_t_n; 
    let resolution_n: f64 = 1.0 / mean_t_n; 

    let mean_t_k: f64 = ticks_k_set.iter().sum::<f64>() / num_ticks_k;
    let mean_sq_t_k: f64 = ticks_k_set.iter().map(|&e| e*e).sum::<f64>() / num_ticks_k;
    let var_t_k: f64 = mean_sq_t_k - mean_t_k.powi(2);
    let accuracy_k: f64 = mean_t_k.powi(2) / var_t_k;
    let resolution_k: f64 = 1.0 / mean_t_k; 

    let mean_t_q: f64 = ticks_q_set.iter().sum::<f64>() / num_ticks_q;
    let mean_sq_t_q: f64 = ticks_q_set.iter().map(|&e| e*e).sum::<f64>() / num_ticks_q;
    let var_t_q: f64 = mean_sq_t_q - mean_t_q.powi(2);
    let accuracy_q: f64 = mean_t_q.powi(2) / var_t_q;
    let resolution_q: f64 = 1.0 / mean_t_q; 

    let mean_exp_entropy_tick_n = exp_entropy_tick_n_sum / num_ticks_n;
    let mean_exp_entropy_tick_k = exp_entropy_tick_k_sum / num_ticks_k;
    let mean_exp_entropy_tick_q = exp_entropy_tick_q_sum / num_ticks_q;

    let mean_exp_entropy_mar_n = exp_entropy_mar_n_sum/num_trajectories as f64;
    let mean_exp_entropy_mar_k = exp_entropy_mar_k_sum/num_trajectories as f64;
    let mean_exp_entropy_mar_q = exp_entropy_mar_q_sum/num_trajectories as f64;

    let mean_act_n = activity_tick_n_sum / num_ticks_n;
    let mean_act_k = activity_tick_k_sum / num_ticks_k;
    let mean_act_q = activity_tick_q_sum / num_ticks_q;

    let mean_ent_n = entropy_tick_n_sum / num_ticks_n;
    let mean_ent_k = entropy_tick_k_sum / num_ticks_k;
    let mean_ent_q = entropy_tick_q_sum / num_ticks_q;

    // --- 2. Sort the waiting times ---
    let mut sorted_waits_n = ticks_n_set;
    let mut sorted_waits_k = ticks_k_set;
    let mut sorted_waits_q = ticks_q_set;

    sorted_waits_n.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sorted_waits_k.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sorted_waits_q.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // --- 3. Compute bin width using IQR rule ---
    let bw_n = bin_width(&sorted_waits_n);
    let bw_k = bin_width(&sorted_waits_k);
    let bw_q = bin_width(&sorted_waits_q);

    // --- 4. Determine range ---
    let min_n = *sorted_waits_n.first().unwrap_or(&0.0);
    let max_n = *sorted_waits_n.last().unwrap_or(&1.0);

    let min_k = *sorted_waits_k.first().unwrap_or(&0.0);
    let max_k = *sorted_waits_k.last().unwrap_or(&1.0);

    let min_q = *sorted_waits_q.first().unwrap_or(&0.0);
    let max_q = *sorted_waits_q.last().unwrap_or(&1.0);

    // println!("{}, {}", min_n, max_n);
    // println!("{}, {}", min_k, max_k);
    // println!("{}, {}", min_q, max_q);

    // --- 5. Count frequencies per bin ---
    let counts_n = counts_per_bin(&sorted_waits_n, bw_n, min_n, max_n);
    let counts_k = counts_per_bin(&sorted_waits_k, bw_n, min_k, max_k);
    let counts_q = counts_per_bin(&sorted_waits_q, bw_n, min_q, max_q);


    // println!("ticks{}",num_ticks_n);
    // println!("Exponentials {}", mean_exp_entropy_mar_n);
    // println!("Exponentials {}", mean_exp_entropy_tick_n);


    Ok(SimulationResults {
        counts_n: counts_n,
        counts_k: counts_k,
        counts_q: counts_q,
        bin_width_n: [bw_n, min_n, max_n],
        bin_width_k: [bw_k, min_k, max_k],
        bin_width_q: [bw_q, min_q, max_q],
        num_ticks_n: num_ticks_n as usize,
        num_ticks_k: num_ticks_k as usize,
        num_ticks_q: num_ticks_q as usize,
        entropy_tick_n: mean_ent_n,
        entropy_tick_k: mean_ent_k,
        entropy_tick_q: mean_ent_q,
        exp_entropy_tick_n: mean_exp_entropy_tick_n,
        exp_entropy_tick_k: mean_exp_entropy_tick_k,
        exp_entropy_tick_q: mean_exp_entropy_tick_q,
        exp_entropy_mar_n: mean_exp_entropy_mar_n, 
        exp_entropy_mar_k: mean_exp_entropy_mar_k, 
        exp_entropy_mar_q: mean_exp_entropy_mar_q, 
        accuracy_n: accuracy_n,
        accuracy_k: accuracy_k,
        accuracy_q: accuracy_q,
        resolution_n: resolution_n,
        resolution_k: resolution_k,
        resolution_q: resolution_q,
        activity_tick_n: mean_act_n,
        activity_tick_k: mean_act_k,
        activity_tick_q: mean_act_q,
    })

}

fn generate_parameter_vectors(n_pts: usize) -> (Vec<f64>, Vec<f64>) {
    let init_s = 50.0_f64;
    let last_s = 50.0_f64;
    let init_lambda = 2.0_f64;
    let last_lambda = 4.0_f64;

    let vec_s: Vec<f64>;
    let vec_lambda: Vec<f64>;

    if n_pts == 1 {
        vec_s = vec![init_s];
        vec_lambda = vec![init_lambda];
    } else {
        vec_s = (0..n_pts)
            .map(|i| {
                let t = i as f64 / (n_pts - 1) as f64;
                let val = init_s as f64 + t * (last_s - init_s) as f64;
                val.round()  // force to nearest integer as float
            })
            .collect();

        vec_lambda = (0..n_pts)
            .map(|i| {
                let t = i as f64 / (n_pts - 1) as f64;
                init_lambda + t * (last_lambda - init_lambda)
            })
            .collect();
    }

    (vec_s, vec_lambda)
}



fn main() -> Result<(), Box<dyn std::error::Error>>{
    
    let n_pts = 1_usize;

    
    // Generate parameter vectors
    let (vec_s, vec_lambda) = generate_parameter_vectors(n_pts);
    
    println!("Analyzing data");
    // In this part there is a file in output with fromat l{lambda}_s{s}, with files l{lambda}_s{s}/{:05}/traj_{:05}.zst
    let num_max_trajectories = 1000;

    // let optimal_m = optimal_threshold(&vec_lambda, &vec_s, num_trajectories)?;

    (10..num_max_trajectories)
        .step_by(10)
        .collect::<Vec<_>>() // Rayon needs a collection
        .into_par_iter()
        .for_each(|i| {
            vec_lambda.par_iter().for_each(|lambda| {
                vec_s.par_iter().for_each(|s| {
                    // If analyze_data returns Result, we need to handle errors properly
                    match analyze_data(*lambda, *s, i, 2.) {
                        Ok(results) => {
                            println!("{},{},{},{},{},{},{}", 
                                i, 
                                results.exp_entropy_mar_n, 
                                results.exp_entropy_mar_k, 
                                results.exp_entropy_mar_q, 
                                results.exp_entropy_tick_n, 
                                results.exp_entropy_tick_k, 
                                results.exp_entropy_tick_q
                            );
                        }
                        Err(e) => eprintln!("Error at i={}, lambda={}, s={}: {}", i, lambda, s, e),
                    }
                });
            });
        });
    // println!("{:?}", optimal_m);





    // let m = 5;
        // let results = 

        // let filename = format!(
        //     "results_s{:.2}_l{:.2}_n{}_m{}.txt",
        //     s, lambda, num_trajectories, m
        // );

        // // Write results to file
        // let mut file = OpenOptions::new()
        //     .create(true)
        //     .append(true)
        //     .open(&filename)?;

        // writeln!(file, "counts_n: {:?}", results.counts_n)?;
        // writeln!(file, "counts_k: {:?}", results.counts_k)?;
        // writeln!(file, "counts_q: {:?}", results.counts_q)?;
        // writeln!(file, "bin_width_n: {:?}", results.bin_width_n)?;
        // writeln!(file, "bin_width_k: {:?}", results.bin_width_k)?;
        // writeln!(file, "bin_width_q: {:?}", results.bin_width_q)?;
        // writeln!(file, "num_ticks_n: {:?}", results.num_ticks_n)?;
        // writeln!(file, "num_ticks_k: {:?}", results.num_ticks_k)?;
        // writeln!(file, "num_ticks_q: {:?}", results.num_ticks_q)?;
        // writeln!(file, "entropy_tick_n: {}", results.entropy_tick_n)?;
        // writeln!(file, "entropy_tick_k: {}", results.entropy_tick_k)?;
        // writeln!(file, "entropy_tick_q: {}", results.entropy_tick_q)?;
        // writeln!(file, "exp_entropy_tick_n: {}", results.exp_entropy_tick_n)?;
        // writeln!(file, "exp_entropy_tick_k: {}", results.exp_entropy_tick_k)?;
        // writeln!(file, "exp_entropy_tick_q: {}", results.exp_entropy_tick_q)?;
        // writeln!(file, "exp_entropy_mar_n: {}", results.exp_entropy_mar_n)?;
        // writeln!(file, "exp_entropy_mar_k: {}", results.exp_entropy_mar_k)?;
        // writeln!(file, "exp_entropy_mar_q: {}", results.exp_entropy_mar_q)?;
        // writeln!(file, "accuracy_n: {}", results.accuracy_n)?;
        // writeln!(file, "accuracy_k: {}", results.accuracy_k)?;
        // writeln!(file, "accuracy_q: {}", results.accuracy_q)?;
        // writeln!(file, "resolution_n: {}", results.resolution_n)?;
        // writeln!(file, "resolution_k: {}", results.resolution_k)?;
        // writeln!(file, "resolution_q: {}", results.resolution_q)?;
        // writeln!(file, "activity_tick_n: {}", results.activity_tick_n)?;
        // writeln!(file, "activity_tick_k: {}", results.activity_tick_k)?;
        // writeln!(file, "activity_tick_q: {}", results.activity_tick_q)?;
        // writeln!(file, "-----------------------------------------------\n")?;

        // // Store results as before
        // counts_n_set.push(results.counts_n);
        // counts_k_set.push(results.counts_k);
        // counts_q_set.push(results.counts_q);
        // bin_width_n_set.push(results.bin_width_n);
        // bin_width_k_set.push(results.bin_width_k);
        // bin_width_q_set.push(results.bin_width_q);
        // num_ticks_n_set.push(results.num_ticks_n);
        // num_ticks_k_set.push(results.num_ticks_k);
        // num_ticks_q_set.push(results.num_ticks_q);
        // entropys_tick_n_set.push(results.entropy_tick_n);
        // entropys_tick_k_set.push(results.entropy_tick_k);
        // entropys_tick_q_set.push(results.entropy_tick_q);
        // exp_entropy_tick_n_set.push(results.exp_entropy_tick_n);
        // exp_entropy_tick_k_set.push(results.exp_entropy_tick_k);
        // exp_entropy_tick_q_set.push(results.exp_entropy_tick_q);
        // exp_entropy_mar_n_set.push(results.exp_entropy_mar_n);
        // exp_entropy_mar_k_set.push(results.exp_entropy_mar_k);
        // exp_entropy_mar_q_set.push(results.exp_entropy_mar_q);
        // accuracy_n_set.push(results.accuracy_n);
        // accuracy_k_set.push(results.accuracy_k);
        // accuracy_q_set.push(results.accuracy_q);
        // resolution_n_set.push(results.resolution_n);
        // resolution_k_set.push(results.resolution_k);
        // resolution_q_set.push(results.resolution_q);
        // activity_tick_n_set.push(results.activity_tick_n);
        // activity_tick_k_set.push(results.activity_tick_k);
        // activity_tick_q_set.push(results.activity_tick_q);
 
    // plot_multiple_histogram(&counts_n_set, &bin_width_n_set, total_time, "Prueba.png")?;

    // println!("{:?}", counts_n_set);
    // println!("{:?}", bin_width_n_set);

    // println!("{:?}, {:?}", exp_entropy_tick_n_set, num_ticks_n_set);
    // println!("{:?}, {:?}", exp_entropy_tick_k_set, num_ticks_k_set);
    // println!("{:?}, {:?}", exp_entropy_tick_q_set, num_ticks_q_set);

    // println!("{:?}, {:?}", exp_entropy_mar_n_set, vec_num_trajectories);
    // println!("{:?}, {:?}", exp_entropy_mar_k_set, vec_num_trajectories);
    // println!("{:?}, {:?}", exp_entropy_mar_q_set, vec_num_trajectories);

    // plot_entropy_vs_n_traj(exp_entropy_tick_n_set, num_ticks_n_set, "entropy_vs_n_traj.png")?;

    // println!("{:?}", num_ticks_n_set);

    println!("Simulation completed successfully!");


    Ok(())
}
