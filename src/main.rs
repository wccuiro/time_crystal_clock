use ndarray::{array, Array1, Array2, linalg::kron};
use ndarray_linalg::{Eigh, UPLO, Solve};

use num_complex::Complex64;
use rand::Rng;

use rayon::prelude::*;

use byteorder::{WriteBytesExt, LittleEndian};

use std::io::{BufWriter, Write, BufReader, Read};

use std::{fs, fs::File, path::Path};
use zstd::stream::{write::Encoder, read::Decoder};



#[derive(Debug)]
struct JumpEvent {
    jump_type: u8, // 0 = m, 1 = p
    time_jump: f64,
    entropy: f64,
}

impl JumpEvent {
    fn write_to(&self, writer: &mut dyn Write) -> std::io::Result<()> {
        writer.write_u8(self.jump_type)?;
        writer.write_f64::<LittleEndian>(self.time_jump)?;
        writer.write_f64::<LittleEndian>(self.entropy)?;
        Ok(())
    }
}


/// Build `L₊ = J₊ − i λ S I` and `L₋ = J₋ + i λ S I`
/// in the spin-S basis of dimension d = 2S+1.
///
/// # Arguments
/// * `lambda` — displacement parameter λ  
/// * `S`      — total spin (e.g. n_s as f64 divided by 2)
fn create_jump_operators(
    lambda: f64,
    s: f64,
) -> (Array2<Complex64>, Array2<Complex64>) {
    // dimension of the symmetric subspace
    let d = (2.0 * s + 1.0) as usize;
    let i_complex = Complex64::new(0.0, 1.0);

    // 1. Build J₊
    let mut j_p = Array2::<Complex64>::zeros((d, d));
    for row in 0..d-1 {
        let m = -s + row as f64;               // row index → m
        let val = ((s - m) * (s + m + 1.0)).sqrt();
        j_p[[row+1, row]] = Complex64::new(val, 0.0);
    }

    // 2. Build J₋ = (J₊)ᵀ (real entries → Hermitian transpose = simple transpose)
    let j_m = j_p.t().mapv(|c| c);

    // 3. Identity
    let mut eye = Array2::<Complex64>::zeros((d, d));
    for n in 0..d {
        eye[[n, n]] = Complex64::new(1.0, 0.0);
    }

    // 4. Displaced jumps
    //    L₊ = J₊ − i λ S I
    //    L₋ = J₋ + i λ S I
    let disp = i_complex * lambda * s;
    let l_p = &j_p - &eye * disp;
    let l_m = &j_m + &eye * disp;

    (l_p, l_m)
}


fn steady_state(
    s: f64,
    lambda: f64,
    gamma_p: f64,
    gamma_m: f64,
) -> (Array2<Complex64>, Array1<f64>, Array2<Complex64>) {
    let (l_p, l_m) = create_jump_operators(lambda, s);
    let d = l_p.nrows();
    let eye_d = Array2::<Complex64>::eye(d);

    // 1) Build L (d²×d²)
    let mut l = Array2::<Complex64>::zeros((d*d, d*d));
    for (lk, gamma) in vec![(l_p.view(), gamma_p), (l_m.view(), gamma_m)] {
        // L_d = L_k† L_k
        let l_d = lk.t().mapv(|c| c.conj()).dot(&lk);

        // +γ (A⊗Cᵀ) for vec(L_k ρ L_k†)
        l = l + kron(&lk.to_owned(), &lk.mapv(|c| c.conj()).t())
             * Complex64::new(gamma/s, 0.0);

        // -γ/2 [L_d⊗I + I⊗L_dᵀ]
        l = l - kron(&l_d.to_owned(), &eye_d)
             * Complex64::new(0.5*gamma/s, 0.0);
        l = l - kron(&eye_d, &l_d.to_owned().t())
             * Complex64::new(0.5*gamma/s, 0.0);
    }

    // 2) Enforce Tr(ρ)=1 by replacing the last row
    let n = d*d;
    let mut l_mod = l;                           // consume `l`
    for j in 0..n {
        l_mod[(n-1, j)] = Complex64::new(0.0, 0.0);
    }
    for i in 0..d {
        l_mod[(n-1, i*d + i)] = Complex64::new(1.0, 0.0);
    }

    // 3) RHS = [0,…,0,1]^T
    let mut b = Array1::<Complex64>::zeros(n);
    b[n-1] = Complex64::new(1.0, 0.0);

    // 4) Solve for vec(ρ_ss)
    let rho_vec = l_mod
        .solve_into(b)
        .expect("Failed to solve steady state");

    // 5) Reshape into d×d ρ_ss
    let mut rho_ss = Array2::from_shape_vec((d, d), rho_vec.to_vec())
        .expect("Reshape error");

    let tr: Complex64 = rho_ss.indexed_iter()
                        .filter(|((i, j), _)| i == j)
                        .map(|(_, &val)| val)
                        .sum();
    
    rho_ss *= Complex64::new(1.0, 0.0) / tr;

    // tr = rho_ss.indexed_iter()
    //                     .filter(|((i, j), _)| i == j)
    //                     .map(|(_, &val)| val)
    //                     .sum();
    
    // println!("{},{}", tr.re, tr.im);

    // 6) Diagonalize ρ_ss (Hermitian) for sampling
    let (eigvals, eigvecs) = rho_ss
        .clone()
        .eigh(UPLO::Lower)
        .expect("rho_ss diagonalization failed");

    (rho_ss, eigvals, eigvecs)
}

fn inst_entropy(pi: &Array2<Complex64> , psi: &Array1<Complex64>, inst_n_m: usize, inst_n_p: usize, betawc:f64) -> f64 {
    let p = {
        let inner = pi.dot(psi);
        let amp = psi.mapv(|c| c.conj()).dot(&inner).re;
        amp.clamp(1e-12, 1.0)
    };

    let inst_q = betawc * (inst_n_m as f64 - inst_n_p as f64) ;
    let inst_s = -p.ln() + inst_q;

    inst_s
}

fn simulate_trajectory(
    gamma_p: f64,
    gamma_m: f64,
    s: f64,
    dt: f64,
    total_time: f64,
    betawc: f64,
    l_plus: &Array2<Complex64>,
    l_minus: &Array2<Complex64>,
    l_p_m: &Array2<Complex64>,
    l_m_p: &Array2<Complex64>,
    h_eff: &Array2<Complex64>,
    pi: &Array2<Complex64>,
    eigvecs: &Array2<Complex64>,
    eigvals: &Array1<f64>,
    writer: &mut dyn Write,
) -> Result<(), Box<dyn std::error::Error>> {
    // Calling a Buffer
    let mut buf = BufWriter::new(writer);

    // Normalize eigenvalues to use as probabilities (if needed)
    let eigvals_sum = eigvals.sum();
    let probabilities = eigvals.mapv(|x| x / eigvals_sum); // Optional: normalize to sum to 1

    // Generate a random number and select an eigenvector
    let mut rng = rand::thread_rng();
    let r: f64 = rng.gen(); // Random number in [0, 1)
    let mut cumulative_prob = 0.0;
    let mut i = 0;
    for (idx, &prob) in probabilities.iter().enumerate() {
        cumulative_prob += prob;
        if r < cumulative_prob {
            i = idx;
            break;
        }
    }

    // Extract and normalize the selected eigenvector
    let mut psi: Array1<Complex64> = eigvecs.column(i).to_owned();
    let norm = psi.mapv(|e| e.conj()).dot(&psi).sqrt(); // Hermitian norm: sqrt(psi^† * psi)
    psi /= norm;
    
    let steps: usize = (total_time / dt).ceil() as usize;
    
    
    let mut inst_n_m = 0; 
    let mut inst_n_p = 0;

    let mut ins_mar_entropy = inst_entropy(&pi , &psi, inst_n_m, inst_n_p, betawc);

    let mut r = rng.gen::<f64>();
    let mut q = rng.gen::<f64>();

    let mut p_p = 1.;
    let mut p_m = 1.;

    let ev0 = JumpEvent {jump_type: 0, time_jump: 0.0_f64, entropy: ins_mar_entropy};
    ev0.write_to(&mut buf)?;


    for i in 0..steps{
        
        
        let amp_m = psi.mapv(|e| e.conj()).dot(&l_p_m.dot(&psi));
        let amp_p = psi.mapv(|e| e.conj()).dot(&l_m_p.dot(&psi));
        
        let prob_p = (gamma_p / s) * amp_p.re * dt;
        let prob_m = (gamma_m / s) * amp_m.re * dt;
        
        let p_total = prob_p + prob_m;
        
        let dpsi_nh = {
            let h_psi = h_eff.dot(&psi);
            let p_term = psi.mapv(|x| x * (0.5 * p_total));
            (&h_psi * Complex64::new(0.0, -1.0))* dt + p_term 
        };
        
        if r >= p_p {
            let dpsi_j_p = l_plus.dot(&psi).mapv(|x| x / (amp_p.re).sqrt());
            psi = dpsi_j_p;
            psi = &psi + &dpsi_nh;
            psi /= psi.mapv(|e| e.conj()).dot(&psi).sqrt();
            
            r = rng.gen::<f64>();
            
            p_p = 1.;

            inst_n_p += 1;
            
            ins_mar_entropy = inst_entropy(&pi , &psi, inst_n_m, inst_n_p, betawc);
            // println!("{}", act_entropy);

            let ev = JumpEvent {jump_type: 1, time_jump: i as f64 * dt, entropy: ins_mar_entropy};
            ev.write_to(&mut buf)?;


        } else if q >= p_m {
            let dpsi_j_m = l_minus.dot(&psi).mapv(|x| x / (amp_m.re).sqrt());
            psi = dpsi_j_m;
            psi = &psi + &dpsi_nh;
            psi /= psi.mapv(|e| e.conj()).dot(&psi).sqrt();

            q = rng.gen::<f64>();
            
            p_m = 1.;

            inst_n_m += 1;

            // let act_entropy: f64 = psi.mapv(|e| e.conj()).dot(&pi.dot(&psi)).re;
            // println!("{}", act_entropy);

            ins_mar_entropy = inst_entropy(&pi , &psi, inst_n_m, inst_n_p, betawc);

            let ev = JumpEvent {jump_type: 0, time_jump: i as f64 * dt, entropy: ins_mar_entropy};
            ev.write_to(&mut buf)?; 


        } else {
            // No jump, just evolve
            psi = &psi + &dpsi_nh;
            psi /= psi.mapv(|e| e.conj()).dot(&psi).sqrt();
        }
        
        p_m *= 1.0 - prob_m;
        p_p *= 1.0 - prob_p;
        
    }

    buf.flush()?;

    Ok(())
}



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

// Configuration struct to organize parameters
#[derive(Debug, Clone)]
struct SimulationConfig {
    dt: f64,
    total_time: f64,
    steps: usize,
    omega_c: f64,
    beta: f64,
    gamma_p: f64,
    gamma_m: f64,
    lambda: f64,
    s: f64,
    num_trajectories: usize,
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

impl SimulationConfig {
    fn new(dt: f64, total_time: f64, omega_c: f64, beta: f64, gamma_p: f64, gamma_m: f64, lambda: f64, s: f64, num_trajectories: usize) -> Self {
        let steps = (total_time / dt).ceil() as usize;
        Self {
            dt,
            total_time,
            steps,
            omega_c,
            beta,
            gamma_p,
            gamma_m,
            lambda,
            s,
            num_trajectories,
        }
    }
}

fn find_first_peak(x: &[f64], y: &[f64]) -> Option<usize> {
    for i in 1..y.len() - 1 {
        if y[i] > y[i - 1] && y[i] > y[i + 1] {
            return Some(i);
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

            let s_int = s as i64;
            for i in 0..max_files {
                let dir_idx = i / 1_000;
                let subdir = format!("{}/l{:.2}_s{}/{:05}", base_dir, lambda, s_int, dir_idx);
                let traj = format!("{}/traj_{:05}.zst", subdir, i);
                let path = std::path::Path::new(&traj);
                if !path.exists() { break; }

                let file = std::fs::File::open(path)?;
                let mut dec = zstd::Decoder::new(std::io::BufReader::new(file))?;
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

            println!("{:?}", m_values);
            println!("{:?}", acts);            

            match find_first_peak(&m_values, &acts) {
                Some(index) => optimal_m.push(m_values[index]),
                None => return Err("No local maximum found.".into()),
            }
        }
    }

    Ok(optimal_m)
}
    // let mut last_tick_n = 0.0;
    // let mut last_tick_k = 0.0;
    // let mut last_tick_q = 0.0;
    
    // let mut ticks_n = Vec::new();
    // let mut ticks_k = Vec::new();
    // let mut ticks_q = Vec::new();
    // let mut activity_tick_n = Vec::new();
    // let mut activity_tick_k = Vec::new();
    // let mut activity_tick_q = Vec::new();

    // let mut last_activity_n = 0;
    // let mut last_activity_k = 0;
    // let mut last_activity_q = 0;

    // let mut entropys_tick_n = Vec::new();
    // let mut entropys_tick_k = Vec::new();
    // let mut entropys_tick_q = Vec::new();

    // if inst_n_m >= (ticks_n.len()+1) * m {
    //     // println!("{} >= {}, where {}", inst_n_m, (ticks_n.len()+1) * m, ticks_n.len());

    //     ticks_n.push(i as f64 * dt - last_tick_n);
    //     last_tick_n = i as f64 * dt;

    //     entropys_tick_n.push(inst_entropy(&pi, &psi, inst_n_m, inst_n_p, betawc) - inst_s_n);
    //     inst_s_n = inst_entropy(&pi, &psi, inst_n_m, inst_n_p, betawc);
        
    //     activity_tick_n.push((inst_n_m + inst_n_p) - last_activity_n);
    //     last_activity_n = inst_n_m + inst_n_p;

    // } 

    // if (inst_n_m + inst_n_p) >= (ticks_k.len()+1) * m {
    //     ticks_k.push(i as f64 * dt - last_tick_k);
    //     last_tick_k = i as f64 * dt;

    //     entropys_tick_k.push(inst_entropy(&pi, &psi, inst_n_m, inst_n_p, betawc) - inst_s_k);
    //     inst_s_k = inst_entropy(&pi, &psi, inst_n_m, inst_n_p, betawc);

    //     activity_tick_k.push((inst_n_m + inst_n_p) - last_activity_k);
    //     last_activity_k = inst_n_m + inst_n_p;            

    // }

    // if (inst_n_m as i32 - inst_n_p as i32) >= ((ticks_q.len()+1) * m) as i32 {
    //     ticks_q.push(i as f64 * dt - last_tick_q);
    //     last_tick_q = i as f64 * dt;
        
    //     entropys_tick_q.push(inst_entropy(&pi, &psi, inst_n_m, inst_n_p, betawc) - inst_s_q);
    //     inst_s_q = inst_entropy(&pi, &psi, inst_n_m, inst_n_p, betawc);
        
    //     activity_tick_q.push((inst_n_m + inst_n_p) - last_activity_q);
    //     last_activity_q = inst_n_m + inst_n_p;
        
    // }

    // println!("{}, {}, {}", ticks_n.len(), ticks_k.len(), ticks_q.len());

    // let ticks_n = ticks_n[1..].to_vec();
    // let ticks_k = ticks_k[1..].to_vec();
    // let ticks_q = ticks_q[1..].to_vec();

    // let activity_tick_n: Array1<usize> = Array1::from(activity_tick_n[1..].to_vec());
    // let activity_tick_k: Array1<usize> = Array1::from(activity_tick_k[1..].to_vec());
    // let activity_tick_q: Array1<usize> = Array1::from(activity_tick_q[1..].to_vec());
    
    // let exp_entropy_mar_n: f64 = (-entropys_tick_n[0]).exp();
    // let exp_entropy_mar_k: f64 = (-entropys_tick_k[0]).exp();
    // let exp_entropy_mar_q: f64 = (-entropys_tick_q[0]).exp();

    // let entropy_tick_n: Array1<f64> = Array1::from(entropys_tick_n[1..].to_vec());
    // let entropy_tick_k: Array1<f64> = Array1::from(entropys_tick_k[1..].to_vec());
    // let entropy_tick_q: Array1<f64> = Array1::from(entropys_tick_q[1..].to_vec());
    

    // // Computing cumulative results insead of vectors
    // let activity_tick_n_sum: f64 = activity_tick_n.iter().sum::<usize>() as f64;
    // let activity_tick_k_sum: f64 = activity_tick_k.iter().sum::<usize>() as f64;
    // let activity_tick_q_sum: f64 = activity_tick_q.iter().sum::<usize>() as f64;

    // let entropy_tick_n_sum: f64 = entropy_tick_n.iter().sum();
    // let entropy_tick_k_sum: f64 = entropy_tick_k.iter().sum();
    // let entropy_tick_q_sum: f64 = entropy_tick_q.iter().sum();

    // let exp_entropy_tick_n_sum = entropy_tick_n.mapv(|e| (-e).exp()).sum();
    // let exp_entropy_tick_k_sum = entropy_tick_k.mapv(|e| (-e).exp()).sum();
    // let exp_entropy_tick_q_sum = entropy_tick_q.mapv(|e| (-e).exp()).sum();

    //  activity_tick_n, activity_tick_k, activity_tick_q,
    //  entropy_tick_n, entropy_tick_k, entropy_tick_q,






    // let mean_act_n = activities_n_sum / waits_n.len() as f64; // Mean of entropies
    // let mean_ent_n = entropies_n_sum/waits_n.len() as f64; // Mean of entropies
    // let mean_exp_entropy_tick_n = exp_entropies_n_sum / waits_n.len() as f64;
    // let mean_exp_entropy_mar_n = entropies_mar_n / num_trajectories as f64;


    // let mean_act_k = activities_k_sum / waits_k.len() as f64; // Mean of entropies
    // let mean_ent_k = entropies_k_sum / waits_k.len() as f64; // Mean of entropies
    // let mean_exp_entropy_tick_k = exp_entropies_k_sum / waits_k.len() as f64;
    // let mean_exp_entropy_mar_k = entropies_mar_k / num_trajectories as f64;


    // let mean_act_q = activities_q_sum / waits_q.len() as f64; // Mean of entropies
    // let mean_ent_q = entropies_q_sum / waits_q.len() as f64; // Mean of entropies
    // let mean_exp_entropy_tick_q = exp_entropies_q_sum / waits_q.len() as f64;
    // let mean_exp_entropy_mar_q = entropies_mar_q / num_trajectories as f64;


    // // Compute accuracies and resolutions
    // let mean_waits_n = waits_n.iter().copied().sum::<f64>() / waits_n.len() as f64;
    // let var_waits_n = waits_n.iter().map(|x| (x - mean_waits_n).powi(2)).sum::<f64>() / (waits_n.len() as f64 - 1.0);
    // let accuracy_n = mean_waits_n.powi(2) / var_waits_n; 
    // let resolution_n = 1.0 / mean_waits_n;

    // let mean_waits_k = waits_k.iter().copied().sum::<f64>() / waits_k.len() as f64;
    // let var_waits_k = waits_k.iter().map(|x| (x - mean_waits_k).powi(2)).sum::<f64>() / (waits_k.len() as f64 - 1.0);
    // let accuracy_k = mean_waits_k.powi(2) / var_waits_k; 
    // let resolution_k = 1.0 / mean_waits_k;

    // let mean_waits_q = waits_q.iter().copied().sum::<f64>() / waits_q.len() as f64;
    // let var_waits_q = waits_q.iter().map(|x| (x - mean_waits_q).powi(2)).sum::<f64>() / (waits_q.len() as f64 - 1.0);
    // let accuracy_q = mean_waits_q.powi(2) / var_waits_q;  
    // let resolution_q = 1.0 / mean_waits_q;

    // // --- 2. Sort the waiting times ---
    // let mut sorted_waits_n = waits_n;
    // let mut sorted_waits_k = waits_k;
    // let mut sorted_waits_q = waits_q;

    // sorted_waits_n.sort_by(|a, b| a.partial_cmp(b).unwrap());
    // sorted_waits_k.sort_by(|a, b| a.partial_cmp(b).unwrap());
    // sorted_waits_q.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    // // --- 3. Compute bin width using IQR rule ---
    // let bw_n = bin_width(&sorted_waits_n);
    // let bw_k = bin_width(&sorted_waits_k);
    // let bw_q = bin_width(&sorted_waits_q);
    
    // // --- 4. Determine range ---
    // let min_n = *sorted_waits_n.first().unwrap_or(&0.0);
    // let max_n = *sorted_waits_n.last().unwrap_or(&1.0);
    
    // let min_k = *sorted_waits_k.first().unwrap_or(&0.0);
    // let max_k = *sorted_waits_k.last().unwrap_or(&1.0);

    // let min_q = *sorted_waits_q.first().unwrap_or(&0.0);
    // let max_q = *sorted_waits_q.last().unwrap_or(&1.0);

    // // println!("{}, {}", min_n, max_n);
    // // println!("{}, {}", min_k, max_k);
    // // println!("{}, {}", min_q, max_q);

    // // --- 5. Count frequencies per bin ---
    // let counts_n = counts_per_bin(&sorted_waits_n, bw_n, min_n, max_n);
    // let counts_k = counts_per_bin(&sorted_waits_k, bw_n, min_k, max_k);
    // let counts_q = counts_per_bin(&sorted_waits_q, bw_n, min_q, max_q);
    
    // // --- 6. Plot histogram ---
    // // let filename = format!("WTD-histogram__m-{}_omega_c-{}_dt-{}_tmax-{}_ntraj-{}.png", m, omega_c, dt, total_time, num_trajectories);
    // // plot_histogram(&counts_n, bw_n, min, max, &filename)?;

    // Ok(SimulationResults {
    //     counts_n: counts_n,
    //     counts_k: counts_k,
    //     counts_q: counts_q,
    //     bin_width_n: [bw_n, min_n, max_n],
    //     bin_width_k: [bw_k, min_k, max_k],
    //     bin_width_q: [bw_q, min_q, max_q],
    //     num_ticks_n: sorted_waits_n.len(),
    //     num_ticks_k: sorted_waits_k.len(),
    //     num_ticks_q: sorted_waits_q.len(),
    //     entropy_tick_n: mean_ent_n,
    //     entropy_tick_k: mean_ent_k,
    //     entropy_tick_q: mean_ent_q,
    //     exp_entropy_tick_n: mean_exp_entropy_tick_n,
    //     exp_entropy_tick_k: mean_exp_entropy_tick_k,
    //     exp_entropy_tick_q: mean_exp_entropy_tick_q,
    //     exp_entropy_mar_n: mean_exp_entropy_mar_n, 
    //     exp_entropy_mar_k: mean_exp_entropy_mar_k, 
    //     exp_entropy_mar_q: mean_exp_entropy_mar_q, 
    //     accuracy_n: accuracy_n,
    //     accuracy_k: accuracy_k,
    //     accuracy_q: accuracy_q,
    //     resolution_n: resolution_n,
    //     resolution_k: resolution_k,
    //     resolution_q: resolution_q,
    //     activity_tick_n: mean_act_n,
    //     activity_tick_k: mean_act_k,
    //     activity_tick_q: mean_act_q,
    // })

// }

/// Run a complete quantum jump simulation for given parameters
fn run_quantum_simulation(config: &SimulationConfig) -> Result<(), Box<dyn std::error::Error>> {
    
    let dt = config.dt;
    let total_time = config.total_time;
    let _steps = config.steps;
    let omega_c = config.omega_c;
    let beta = config.beta;
    let gamma_p = config.gamma_p;
    let gamma_m = config.gamma_m;
    let lambda = config.lambda;
    let s = config.s;
    let num_trajectories = config.num_trajectories;
    
    let betawc = beta * omega_c;

    let (l_plus, l_minus) = create_jump_operators(lambda, s);

    let l_p_m = &l_plus.dot(&l_minus);  
    let l_m_p = &l_minus.dot(&l_plus);  

    let h_eff = l_plus.dot(&l_minus).mapv(|x| x * Complex64::new(0.0, -0.5 * gamma_m / s)) 
    + l_minus.dot(&l_plus).mapv(|x| x * Complex64::new(0.0, -0.5 * gamma_p / s));
    
    println!("Initialazing data");
    let (pi, eigvals, eigvecs) = steady_state(s, lambda, gamma_p, gamma_m);
    
    println!("Generating data");
    // 2) Phase 1: simulate in parallel, updating the bar
    (0..num_trajectories)
        .into_par_iter()
        .for_each(|i| {
            // 1) make subfolder
            let subdir = format!("output/l{:.2}_s{}/{:05}", lambda, s as i64, i / 1_000);
            if fs::create_dir_all(&subdir).is_err() {
                return;
            }

            // 2) open per-trajectory .zst
            let path = format!("{}/traj_{:05}.zst", subdir, i);
            let file = match File::create(&path) {
                Ok(f) => f,
                Err(_) => return,
            };
            let mut encoder = match Encoder::new(file, 3) {
                Ok(e) => e,
                Err(_) => return,
            };

            // Coerce encoder to a dyn Write trait object
            let writer: &mut dyn Write = &mut encoder;

            // 3) simulate & write jumps
            if simulate_trajectory(
                gamma_p, gamma_m, s, dt, total_time, betawc,
                &l_plus, &l_minus, &l_p_m, &l_m_p,
                &h_eff, &pi, &eigvecs, &eigvals,
                writer,
            ).is_err() {
                return;
            }

            let _ = encoder.finish();
        });
    Ok(())
}

fn generate_parameter_vectors(n_pts: usize) -> (Vec<f64>, Vec<f64>) {
    let init_s = 20.0_f64;
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
    // Fixed simulation parameters
    let dt: f64 = 0.001;            // dt = 10-3 ~20 n_ticks and after that does not increase for a threshold of 1100 and beta 2.0
    let total_time: f64 = 5000.0;        // Total time 5000 set it to have an average of 20 n_ticks for a threshold of 1100 and beta 2.0
    let omega_c: f64 = 0.01; // Frequency scale
    let beta: f64 = 2.0 / omega_c; // Inverse temperature
    let betawc = beta * omega_c;
    let gamma_z = 1. ;// 1./1000.*omega_c;
    let nb = 1./(betawc.exp() - 1.);
    
    let n_pts = 1_usize;

    let num_trajectories = 100;

    // Generate parameter vectors
    let (vec_s, vec_lambda) = generate_parameter_vectors(n_pts);

    println!("Running simulations with S: {:?}, lambda: {:?}", vec_s, vec_lambda);

    // Generate Data
    for (&s, &lambda) in vec_s.iter().zip(vec_lambda.iter()) {
        let gamma_p: f64 = gamma_z / s * nb;
        let gamma_m: f64 = gamma_z / s * (nb + 1.0);
        
        let config = SimulationConfig::new(
            dt, total_time, omega_c, beta, gamma_p, gamma_m, lambda, s, num_trajectories,
        );
        
        run_quantum_simulation(&config)?;
        
    }

    println!("Analyzing data");
    // In this part there is a file in output with fromat l{lambda}_s{s}, with files l{lambda}_s{s}/{:05}/traj_{:05}.zst
    let max_files = 2;

    let optimal_m = optimal_threshold(&vec_lambda, &vec_s, max_files)?;

    println!("{:?}", optimal_m);





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
