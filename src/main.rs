use ndarray::{array, Array1, Array2};

use num_complex::Complex64;
use rand::Rng;

use rayon::prelude::*;

fn create_jump_operators(lambda: f64, s: f64) -> (Array2<Complex64>, Array2<Complex64>) {

    let sigma_plus = array![
        [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        [Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)]
    ];
    let sigma_minus = array![
        [Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)],
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]
    ];

    let identity = array![
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)]
    ];

    let l_plus = sigma_plus - identity.mapv(|e| e * Complex64::new(0.0, 1.0) * lambda * s);
    let l_minus = sigma_minus + identity.mapv(|e| e * Complex64::new(0.0, 1.0) * lambda * s);
    
    (l_plus, l_minus)
} 

fn steady_state(s: f64, lambda: f64, gamma_p: f64, gamma_m: f64) -> (Array2<Complex64>, Array1<Complex64>, Array1<Complex64>, Array1<f64>) {
    let delta: f64 =  ((gamma_m + gamma_p).powi(2) + 4. * (gamma_m - gamma_p).powi(2) * s.powi(2) * lambda.powi(2)).sqrt();

    // Eigenvalues
    let eig_val_1 = (gamma_m.powi(2) * (1. + 2. * s.powi(2) * lambda.powi(2)) + gamma_p * (gamma_p + 2. * gamma_p * s.powi(2) * lambda.powi(2) - delta) + gamma_m * (gamma_p * (2. - 4. * s.powi(2) * lambda.powi(2)) + delta)) / (2. * (gamma_m + gamma_p).powi(2) + 4. * (gamma_m - gamma_p).powi(2) * s.powi(2) * lambda.powi(2));

    let eig_val_2 = (gamma_m.powi(2) * (1. + 2. * s.powi(2) * lambda.powi(2)) + gamma_p * (gamma_p + 2. * gamma_p * s.powi(2) * lambda.powi(2) + delta) - gamma_m * (gamma_p * (-2. + 4. * s.powi(2) * lambda.powi(2)) + delta)) / (2. * (gamma_m + gamma_p).powi(2) + 4. * (gamma_m - gamma_p).powi(2) * s.powi(2) * lambda.powi(2));

    let mut eig_vals: Array1<f64> = array![eig_val_1, eig_val_2];
    eig_vals = eig_vals.mapv(|e| e/ (eig_val_1 + eig_val_2));

    // Eigenvectors
    let i = Complex64::new(0.0, 1.0);

    // psi1
    let top11 = i * (gamma_m + gamma_p - delta);
    let norm1 = (2.0 * ((gamma_m + gamma_p).powi(2)
        + 4.0 * (gamma_m - gamma_p).powi(2) * s.powi(2) * lambda.powi(2)
        - gamma_m * delta
        - gamma_p * delta))
        .sqrt();
    let top12 = (4. * (gamma_m - gamma_p).powi(2) * s.powi(2) * lambda.powi(2)).sqrt();
    let mut psi1 = Array1::from(vec![top11 / norm1, Complex64::new( top12 / norm1, 0.0)]);
    psi1 /= psi1.mapv(|e| e.conj()).dot(&psi1).sqrt();

    // psi2
    let top21 = i * (gamma_m + gamma_p + delta);
    let norm2 = (2.0 * ((gamma_m + gamma_p).powi(2)
        + 4.0 * (gamma_m - gamma_p).powi(2) * s.powi(2) * lambda.powi(2)
        + gamma_m * delta
        + gamma_p * delta))
        .sqrt();
    let top22 = (4. * (gamma_m - gamma_p).powi(2) * s.powi(2) * lambda.powi(2)).sqrt();
    let mut psi2 = Array1::from(vec![top21 / norm2, Complex64::new( top22 / norm2, 0.0)]);
    psi2 /= psi2.mapv(|e| e.conj()).dot(&psi2).sqrt();


    // Steady state density matrix
    let a = psi1[0];
    let b = psi1[1];
    let op1 = array![
        [a * a.conj(), a * b.conj()],
        [b * a.conj(), b * b.conj()]
    ];

    // Compute |psi2><psi2|
    let c = psi2[0];
    let d = psi2[1];
    let op2 = array![
        [c * c.conj(), c * d.conj()],
        [d * c.conj(), d * d.conj()]
    ];

    // Weighted sum
    let mut pi = op1.mapv(|v| v * eig_vals[0]) + op2.mapv(|v| v * eig_vals[1]);

    // Normalize by trace
    let trace: f64 = (pi[(0, 0)] + pi[(1, 1)]).re;
    pi /= Complex64::new(trace, 0.0);

    (pi, psi1, psi2, eig_vals)
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
    m: usize,
    l_plus: &Array2<Complex64>,
    l_minus: &Array2<Complex64>,
    h_eff: &Array2<Complex64>,
    pi: &Array2<Complex64>,
    psi1: &Array1<Complex64>,
    psi2: &Array1<Complex64>,
    eigvals: &Array1<f64>,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64) {
    let mut rng = rand::thread_rng();
    let i = if rng.gen::<f64>() < eigvals[0] { 0 } else { 1 };
    let mut psi;
    
    let steps: usize = (total_time / dt).ceil() as usize;
    
    if i == 0 {
        psi = psi1.clone();
        psi /= psi.mapv(|e| e.conj()).dot(&psi).sqrt();
    } else {
        psi = psi2.clone();
        psi /= psi.mapv(|e| e.conj()).dot(&psi).sqrt();
    }

    let mut ticks_n = Vec::new();
    let mut ticks_k = Vec::new();
    let mut ticks_q = Vec::new();
    
    let mut last_tick_n = 0.0;
    let mut last_tick_k = 0.0;
    let mut last_tick_q = 0.0;

    let mut activity_tick_n = Vec::new();
    let mut activity_tick_k = Vec::new();
    let mut activity_tick_q = Vec::new();

    let mut last_activity_n = 0;
    let mut last_activity_k = 0;
    let mut last_activity_q = 0;

    let mut inst_n_m = 0; 
    let mut inst_n_p = 0;

    let mut entropys_tick_n = Vec::new();
    let mut entropys_tick_k = Vec::new();
    let mut entropys_tick_q = Vec::new();

    let mut inst_s_n = inst_entropy(&pi , &psi, inst_n_m, inst_n_p, betawc);
    let mut inst_s_k = inst_s_n;
    let mut inst_s_q = inst_s_n;
    
    let mut r = rng.gen::<f64>();
    let mut q = rng.gen::<f64>();

    let mut p_p = 1.;
    let mut p_m = 1.;

    
    for i in 0..steps{
        
        
        let amp_m = psi.mapv(|e| e.conj()).dot(&l_plus.dot(l_minus).dot(&psi));
        let amp_p = psi.mapv(|e| e.conj()).dot(&l_minus.dot(l_plus).dot(&psi));
        
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
            
        } else if q >= p_m {
            let dpsi_j_m = l_minus.dot(&psi).mapv(|x| x / (amp_m.re).sqrt());
            psi = dpsi_j_m;
            psi = &psi + &dpsi_nh;
            psi /= psi.mapv(|e| e.conj()).dot(&psi).sqrt();
                        
            q = rng.gen::<f64>();
            
            p_m = 1.;

            inst_n_m += 1;

        } else {
            // No jump, just evolve
            psi = &psi + &dpsi_nh;
            psi /= psi.mapv(|e| e.conj()).dot(&psi).sqrt();
        }

        if inst_n_m >= (ticks_n.len()+1) * m {
            // println!("{} >= {}, where {}", inst_n_m, (ticks_n.len()+1) * m, ticks_n.len());

            ticks_n.push(i as f64 * dt - last_tick_n);
            last_tick_n = i as f64 * dt;

            entropys_tick_n.push(inst_entropy(&pi, &psi, inst_n_m, inst_n_p, betawc) - inst_s_n);
            inst_s_n = inst_entropy(&pi, &psi, inst_n_m, inst_n_p, betawc);
            
            activity_tick_n.push((inst_n_m + inst_n_p) - last_activity_n);
            last_activity_n = inst_n_m + inst_n_p;

        } 

        if (inst_n_m + inst_n_p) >= (ticks_k.len()+1) * m {
            ticks_k.push(i as f64 * dt - last_tick_k);
            last_tick_k = i as f64 * dt;

            entropys_tick_k.push(inst_entropy(&pi, &psi, inst_n_m, inst_n_p, betawc) - inst_s_k);
            inst_s_k = inst_entropy(&pi, &psi, inst_n_m, inst_n_p, betawc);

            activity_tick_k.push((inst_n_m + inst_n_p) - last_activity_k);
            last_activity_k = inst_n_m + inst_n_p;            

        }

        if (inst_n_m as i32 - inst_n_p as i32) >= ((ticks_q.len()+1) * m) as i32 {
            ticks_q.push(i as f64 * dt - last_tick_q);
            last_tick_q = i as f64 * dt;
            
            entropys_tick_q.push(inst_entropy(&pi, &psi, inst_n_m, inst_n_p, betawc) - inst_s_q);
            inst_s_q = inst_entropy(&pi, &psi, inst_n_m, inst_n_p, betawc);
            
            activity_tick_q.push((inst_n_m + inst_n_p) - last_activity_q);
            last_activity_q = inst_n_m + inst_n_p;
            
        }
        
        p_m *= 1.0 - prob_m;
        p_p *= 1.0 - prob_p;
        
    }

    println!("{}, {}, {}", ticks_n.len(), ticks_k.len(), ticks_q.len());

    let ticks_n = ticks_n[1..].to_vec();
    let ticks_k = ticks_k[1..].to_vec();
    let ticks_q = ticks_q[1..].to_vec();

    let activity_tick_n: Array1<usize> = Array1::from(activity_tick_n[1..].to_vec());
    let activity_tick_k: Array1<usize> = Array1::from(activity_tick_k[1..].to_vec());
    let activity_tick_q: Array1<usize> = Array1::from(activity_tick_q[1..].to_vec());
    
    let exp_entropy_mar_n: f64 = (-entropys_tick_n[0]).exp();
    let exp_entropy_mar_k: f64 = (-entropys_tick_k[0]).exp();
    let exp_entropy_mar_q: f64 = (-entropys_tick_q[0]).exp();

    let entropy_tick_n: Array1<f64> = Array1::from(entropys_tick_n[1..].to_vec());
    let entropy_tick_k: Array1<f64> = Array1::from(entropys_tick_k[1..].to_vec());
    let entropy_tick_q: Array1<f64> = Array1::from(entropys_tick_q[1..].to_vec());
    

    // Computing cumulative results insead of vectors
    let activity_tick_n_sum: f64 = activity_tick_n.iter().sum::<usize>() as f64;
    let activity_tick_k_sum: f64 = activity_tick_k.iter().sum::<usize>() as f64;
    let activity_tick_q_sum: f64 = activity_tick_q.iter().sum::<usize>() as f64;

    let entropy_tick_n_sum: f64 = entropy_tick_n.iter().sum();
    let entropy_tick_k_sum: f64 = entropy_tick_k.iter().sum();
    let entropy_tick_q_sum: f64 = entropy_tick_q.iter().sum();

    let exp_entropy_tick_n_sum = entropy_tick_n.mapv(|e| (-e).exp()).sum();
    let exp_entropy_tick_k_sum = entropy_tick_k.mapv(|e| (-e).exp()).sum();
    let exp_entropy_tick_q_sum = entropy_tick_q.mapv(|e| (-e).exp()).sum();

    (ticks_n, ticks_k, ticks_q, 
        activity_tick_n_sum, activity_tick_k_sum, activity_tick_q_sum,
        entropy_tick_n_sum, entropy_tick_k_sum, entropy_tick_q_sum,
        exp_entropy_tick_n_sum, exp_entropy_tick_k_sum, exp_entropy_tick_q_sum,
        exp_entropy_mar_n, exp_entropy_mar_k, exp_entropy_mar_q)
    //  activity_tick_n, activity_tick_k, activity_tick_q,
    //  entropy_tick_n, entropy_tick_k, entropy_tick_q,
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
    m: usize,
}

// Results struct to organize outputs
#[derive(Debug)]
struct SimulationResults {
    counts_n: Vec<f64>,
    counts_k: Vec<f64>,
    counts_q: Vec<f64>,
    bin_width_n: f64,
    bin_width_k: f64,
    bin_width_q: f64,
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
    fn new(dt: f64, total_time: f64, omega_c: f64, beta: f64, gamma_p: f64, gamma_m: f64, lambda: f64, s: f64, num_trajectories: usize, m: usize) -> Self {
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
            m,
        }
    }
}

/// Run a complete quantum jump simulation for given parameters
fn run_quantum_simulation(config: &SimulationConfig) -> Result<SimulationResults, Box<dyn std::error::Error>> {
    
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
    let m = config.m;
    
    let betawc = beta * omega_c;

    let (l_plus, l_minus) = create_jump_operators(lambda, s);

    let h_eff = l_plus.dot(&l_minus).mapv(|x| x * Complex64::new(0.0, -0.5 * gamma_m / s)) 
    + l_minus.dot(&l_plus).mapv(|x| x * Complex64::new(0.0, -0.5 * gamma_p / s));
    
    let (pi, psi1, psi2, eigvals) = steady_state(s, lambda, gamma_p, gamma_m);

        
    // 2) Phase 1: simulate in parallel, updating the bar
    let (waits_n, waits_k, waits_q, 
        activities_n_sum, activities_k_sum, activities_q_sum,
        entropies_n_sum, entropies_k_sum, entropies_q_sum,
        exp_entropies_n_sum, exp_entropies_k_sum, exp_entropies_q_sum,
        entropies_mar_n, entropies_mar_k, entropies_mar_q): 
        (Vec<f64>, Vec<f64>, Vec<f64>, 
        f64, f64, f64,
        f64, f64, f64,
        f64, f64, f64,
        f64, f64, f64) = 
        (0..num_trajectories)
            .into_par_iter()
            .map(|_| simulate_trajectory(gamma_p, gamma_m, s, dt, total_time, betawc, m, &l_plus, &l_minus, &h_eff, &pi, &psi1, &psi2, &eigvals))
            .filter(|(ticks_n, _, _, _, _, _, _, _, _, _, _, _, _, _, _)| ticks_n.len() >= 2)
            .reduce(
                || (
                    Vec::new(), Vec::new(), Vec::new(), // waits_n, waits_k, waits_q
                    0.0, 0.0, 0.0,         // activities_n_sum, activities_k_sum, activities_q_sum
                    0.0, 0.0, 0.0,         // entropies_n_sum, entropies_k_sum, entropies_q_sum
                    0.0, 0.0, 0.0,         // exp_entropies_n_sum, exp_entropies_k_sum, exp_entropies_q_sum
                    0.0, 0.0, 0.0          // entropies_mar_n, entropies_mar_k, entropies_mar_q
                ),
                |mut acc, x| {
                    acc.0.extend(x.0);  // waits_n - flatten Array1 to Vec
                    acc.1.extend(x.1);  // waits_k - flatten Array1 to Vec
                    acc.2.extend(x.2);  // waits_q - flatten Array1 to Vec
                    acc.3 += x.3;  // activities_n_sum
                    acc.4 += x.4;  // activities_k_sum
                    acc.5 += x.5;  // activities_q_sum
                    acc.6 += x.6;  // entropies_n_sum
                    acc.7 += x.7;  // entropies_k_sum
                    acc.8 += x.8;  // entropies_q_sum
                    acc.9 += x.9;  // exp_entropies_n_sum
                    acc.10 += x.10; // exp_entropies_k_sum
                    acc.11 += x.11; // exp_entropies_q_sum
                    acc.12 += x.12; // entropies_mar_n
                    acc.13 += x.13; // entropies_mar_k
                    acc.14 += x.14; // entropies_mar_q
                    acc
                }
            );

    let mean_act_n = activities_n_sum / waits_n.len() as f64; // Mean of entropies
    let mean_ent_n = entropies_n_sum/waits_n.len() as f64; // Mean of entropies
    let mean_exp_entropy_tick_n = exp_entropies_n_sum / waits_n.len() as f64;
    let mean_exp_entropy_mar_n = entropies_mar_n / num_trajectories as f64;


    let mean_act_k = activities_k_sum / waits_k.len() as f64; // Mean of entropies
    let mean_ent_k = entropies_k_sum / waits_k.len() as f64; // Mean of entropies
    let mean_exp_entropy_tick_k = exp_entropies_k_sum / waits_k.len() as f64;
    let mean_exp_entropy_mar_k = entropies_mar_k / num_trajectories as f64;


    let mean_act_q = activities_q_sum / waits_q.len() as f64; // Mean of entropies
    let mean_ent_q = entropies_q_sum / waits_q.len() as f64; // Mean of entropies
    let mean_exp_entropy_tick_q = exp_entropies_q_sum / waits_q.len() as f64;
    let mean_exp_entropy_mar_q = entropies_mar_q / num_trajectories as f64;


    // Compute accuracies and resolutions
    let mean_waits_n = waits_n.iter().copied().sum::<f64>() / waits_n.len() as f64;
    let var_waits_n = waits_n.iter().map(|x| (x - mean_waits_n).powi(2)).sum::<f64>() / (waits_n.len() as f64 - 1.0);
    let accuracy_n = mean_waits_n.powi(2) / var_waits_n; 
    let resolution_n = 1.0 / mean_waits_n;

    let mean_waits_k = waits_k.iter().copied().sum::<f64>() / waits_k.len() as f64;
    let var_waits_k = waits_k.iter().map(|x| (x - mean_waits_k).powi(2)).sum::<f64>() / (waits_k.len() as f64 - 1.0);
    let accuracy_k = mean_waits_k.powi(2) / var_waits_k; 
    let resolution_k = 1.0 / mean_waits_k;

    let mean_waits_q = waits_q.iter().copied().sum::<f64>() / waits_q.len() as f64;
    let var_waits_q = waits_q.iter().map(|x| (x - mean_waits_q).powi(2)).sum::<f64>() / (waits_q.len() as f64 - 1.0);
    let accuracy_q = mean_waits_q.powi(2) / var_waits_q;  
    let resolution_q = 1.0 / mean_waits_q;

    // --- 2. Sort the waiting times ---
    let mut sorted_waits_n = waits_n;
    let mut sorted_waits_k = waits_k;
    let mut sorted_waits_q = waits_q;

    sorted_waits_n.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sorted_waits_k.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sorted_waits_q.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    // --- 3. Compute bin width using IQR rule ---
    let bw_n = bin_width(&sorted_waits_n);
    let bw_k = bin_width(&sorted_waits_k);
    let bw_q = bin_width(&sorted_waits_q);
    
    // --- 4. Determine range ---
    let min = *sorted_waits_n.first().unwrap_or(&0.0);
    let max = *sorted_waits_n.last().unwrap_or(&1.0);
    
    // --- 5. Count frequencies per bin ---
    let counts_n = counts_per_bin(&sorted_waits_n, bw_n, min, max);
    let counts_k = counts_per_bin(&sorted_waits_k, bw_n, min, max);
    let counts_q = counts_per_bin(&sorted_waits_q, bw_n, min, max);
    
    // --- 6. Plot histogram ---
    // let filename = format!("WTD-histogram__m-{}_omega_c-{}_dt-{}_tmax-{}_ntraj-{}.png", m, omega_c, dt, total_time, num_trajectories);
    // plot_histogram(&counts_n, bw_n, min, max, &filename)?;

    Ok(SimulationResults {
        counts_n: counts_n,
        counts_k: counts_k,
        counts_q: counts_q,
        bin_width_n: bw_n,
        bin_width_k: bw_k,
        bin_width_q: bw_q,
        num_ticks_n: sorted_waits_n.len(),
        num_ticks_k: sorted_waits_k.len(),
        num_ticks_q: sorted_waits_q.len(),
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

fn generate_parameter_vectors(n_pts: usize) -> (Vec<f64>, Vec<f64>, Vec<usize>, Vec<usize>) {
    let init_s = 50.0_f64;
    let last_s = 50.0_f64;
    let vec_omega: Vec<f64> = (0..n_pts)
        .map(|i| {
            let t = i as f64 / (n_pts - 1) as f64;
            init_s + t * (last_s - init_s)
        })
        .collect();

    let init_lambda = 2.0_f64;
    let last_lambda = 2.0_f64;
    let vec_gamma: Vec<f64> = (0..n_pts)
        .map(|i| {
            let t = i as f64 / (n_pts - 1) as f64;
            init_lambda + t * (last_lambda - init_lambda)
        })
        .collect();

    let init_num_trajectories = 100_usize;
    let last_num_trajectories = 1000_usize;
    let vec_num_trajectories: Vec<usize> = (0..n_pts)
        .map(|i| {
            let t = i as f64 / (n_pts - 1) as f64;
            (init_num_trajectories as f64 + t * (last_num_trajectories - init_num_trajectories) as f64) as usize
        })
        .collect();

    let init_m = 1100_usize;
    let last_m = 1100_usize;
    let vec_m: Vec<usize> = (0..n_pts)
        .map(|i| {
            let t = i as f64 / (n_pts - 1) as f64;
            (init_m as f64 + t * (last_m - init_m) as f64) as usize
        })
        .collect();
    
    // let vec_m: Vec<usize> = vec![355, 650, 705, 1100];      // Values for figure A2

    (vec_omega, vec_gamma, vec_num_trajectories, vec_m)
}


fn main() -> Result<(), Box<dyn std::error::Error>>{
    // Fixed simulation parameters
    let dt: f64 = 0.001;            // if dt = 10-2 then total_time 5000 generates ~20 n ticks but for dt = 10-3 ~190 n ticks and after  that does not increase
    let total_time: f64 = 5000.0;        // Total time 5000 set it to have an average of 20 ticks for a threshold of 1100 and beta 2.0
    let omega_c: f64 = 0.01; // Frequency scale
    let beta: f64 = 0.1 / omega_c; // Inverse temperature
    let betawc = beta * omega_c;
    let gamma_z = 1. ;// 1./1000.*omega_c;
    let nb = 1./(betawc.exp() - 1.);
    
    let n_pts = 2_usize;

    // Generate parameter vectors
    let (vec_s, vec_lambda, vec_num_trajectories, vec_m) = generate_parameter_vectors(n_pts);

    println!("Running simulations with S: {:?}, lambda: {:?}, n_traj: {:?} and M: {:?}", vec_s, vec_lambda, vec_num_trajectories, vec_m);

    // Pre-allocate result vectors
    let mut counts_n_set: Vec<Vec<f64>> = Vec::with_capacity(n_pts);
    let mut counts_k_set: Vec<Vec<f64>> = Vec::with_capacity(n_pts);
    let mut counts_q_set: Vec<Vec<f64>> = Vec::with_capacity(n_pts);
    let mut bin_width_n_set: Vec<f64> = Vec::with_capacity(n_pts);
    let mut bin_width_k_set: Vec<f64> = Vec::with_capacity(n_pts);
    let mut bin_width_q_set: Vec<f64> = Vec::with_capacity(n_pts);
    let mut num_ticks_n_set: Vec<usize> = Vec::with_capacity(n_pts);
    let mut num_ticks_k_set: Vec<usize> = Vec::with_capacity(n_pts);
    let mut num_ticks_q_set: Vec<usize> = Vec::with_capacity(n_pts);
    let mut entropys_tick_n_set: Vec<f64> = Vec::with_capacity(n_pts);
    let mut entropys_tick_k_set: Vec<f64> = Vec::with_capacity(n_pts);
    let mut entropys_tick_q_set: Vec<f64> = Vec::with_capacity(n_pts);
    let mut exp_entropy_tick_n_set: Vec<f64> = Vec::with_capacity(n_pts);
    let mut exp_entropy_tick_k_set: Vec<f64> = Vec::with_capacity(n_pts);
    let mut exp_entropy_tick_q_set: Vec<f64> = Vec::with_capacity(n_pts);
    let mut exp_entropy_mar_n_set: Vec<f64> = Vec::with_capacity(n_pts);
    let mut exp_entropy_mar_k_set: Vec<f64> = Vec::with_capacity(n_pts);
    let mut exp_entropy_mar_q_set: Vec<f64> = Vec::with_capacity(n_pts);
    let mut accuracy_n_set: Vec<f64> = Vec::with_capacity(n_pts);
    let mut accuracy_k_set: Vec<f64> = Vec::with_capacity(n_pts);
    let mut accuracy_q_set: Vec<f64> = Vec::with_capacity(n_pts);
    let mut resolution_n_set: Vec<f64> = Vec::with_capacity(n_pts);
    let mut resolution_k_set: Vec<f64> = Vec::with_capacity(n_pts);
    let mut resolution_q_set: Vec<f64> = Vec::with_capacity(n_pts);
    let mut activity_tick_n_set: Vec<f64> = Vec::with_capacity(n_pts);
    let mut activity_tick_k_set: Vec<f64> = Vec::with_capacity(n_pts);
    let mut activity_tick_q_set: Vec<f64> = Vec::with_capacity(n_pts);

    // Run simulations
    for (((&s, &lambda), &num_trajectories), &m) in vec_s.iter().zip(vec_lambda.iter()).zip(vec_num_trajectories.iter()).zip(vec_m.iter()) 
    {
        let gamma_p: f64 = gamma_z/s * nb;
        let gamma_m: f64 = gamma_z/s * ( nb + 1.);

        let config = SimulationConfig::new(dt, total_time, omega_c, beta, gamma_p, gamma_m, lambda, s, num_trajectories, m);

        let results = run_quantum_simulation(&config)?;

        // Store results
        counts_n_set.push(results.counts_n);
        counts_k_set.push(results.counts_k);
        counts_q_set.push(results.counts_q);
        bin_width_n_set.push(results.bin_width_n);
        bin_width_k_set.push(results.bin_width_k);
        bin_width_q_set.push(results.bin_width_q);
        num_ticks_n_set .push(results.num_ticks_n);
        num_ticks_k_set .push(results.num_ticks_k);
        num_ticks_q_set .push(results.num_ticks_q);
        entropys_tick_n_set.push(results.entropy_tick_n);
        entropys_tick_k_set.push(results.entropy_tick_k);
        entropys_tick_q_set.push(results.entropy_tick_q);
        exp_entropy_tick_n_set.push(results.exp_entropy_tick_n);
        exp_entropy_tick_k_set.push(results.exp_entropy_tick_k);
        exp_entropy_tick_q_set.push(results.exp_entropy_tick_q);
        exp_entropy_mar_n_set.push(results.exp_entropy_mar_n);
        exp_entropy_mar_k_set.push(results.exp_entropy_mar_k);
        exp_entropy_mar_q_set.push(results.exp_entropy_mar_q);
        accuracy_n_set.push(results.accuracy_n);
        accuracy_k_set.push(results.accuracy_k);
        accuracy_q_set.push(results.accuracy_q);
        resolution_n_set.push(results.resolution_n);
        resolution_k_set.push(results.resolution_k);
        resolution_q_set.push(results.resolution_q);
        activity_tick_n_set.push(results.activity_tick_n);
        activity_tick_k_set.push(results.activity_tick_k);
        activity_tick_q_set.push(results.activity_tick_q);
    }

    // plot_multiple_histogram(&counts_n_set, &bin_width_n_set, total_time, "Prueba.png")?;

    println!("{:?}, {:?}, {:?}", exp_entropy_tick_n_set, exp_entropy_mar_n_set, num_ticks_n_set);
    // plot_entropy_vs_n_traj(exp_entropy_tick_n_set, num_ticks_n_set, "entropy_vs_n_traj.png")?;

    // println!("{:?}", num_ticks_n_set);

    println!("Simulation completed successfully!");


    Ok(())
}
