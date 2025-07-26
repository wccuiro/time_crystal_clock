use ndarray::{Array1, Array2, linalg::kron};
use ndarray_linalg::{Eigh, UPLO, Solve};

use num_complex::Complex64;
use rand::Rng;

use rayon::prelude::*;

use byteorder::{WriteBytesExt, LittleEndian};

use std::io::{BufWriter, Write};

use std::{fs, fs::File};
use zstd::stream::{write::Encoder};



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
    s: f64,        // time‐scale (>0)
    lambda: f64,   // Hamiltonian coupling
    gamma_p: f64,  // jump rate for L₊
    gamma_m: f64,  // jump rate for L₋
) -> (Array2<Complex64>, Array1<f64>, Array2<Complex64>) {
    // 1) get jump ops
    let (l_plus, l_minus) = create_jump_operators(lambda, s);
    let d = l_plus.nrows();
    let n = d * d;
    let identity = Array2::<Complex64>::eye(d);

    let term2_m = kron(&l_minus, &l_plus.t());
    let term2_p = kron(&l_plus, &l_minus.t());

    let left_p = l_minus.dot(&l_plus);
    let right_p = l_plus.t().dot(&l_minus.t());

    let left_m = l_plus.dot(&l_minus);
    let right_m = l_minus.t().dot(&l_plus.t());

    let term3_m = (kron(&left_m, &identity) + kron(&identity, &right_m)).mapv(|e| e * 0.5); 
    let term3_p = (kron(&left_p, &identity) + kron(&identity, &right_p)).mapv(|e| e * 0.5);

    let s_l = (&term2_m - &term3_m).mapv(|e| e * gamma_m / s) + (&term2_p - &term3_p).mapv(|e| e * gamma_p / s);

    // 3) enforce Tr(ρ)=1 by replacing last row of L
    let mut l_mod = s_l.clone();
    for j in 0..n {
        l_mod[(n-1, j)] = Complex64::new(0.0, 0.0);
    }
    for i in 0..d {
        l_mod[(n-1, i*d + i)] = Complex64::new(1.0, 0.0);
    }

    // 4) RHS = [0,...,1]^T
    let mut b = Array1::<Complex64>::zeros(n);
    b[n-1] = Complex64::new(1.0, 0.0);

    // 5) solve for vec(ρ_ss)
    let rho_vec = l_mod
        .solve_into(b)
        .expect("Failed to solve steady state");

    // 6) reshape → d×d
    let mut rho_ss = Array2::from_shape_vec((d, d), rho_vec.to_vec())
        .expect("Reshape error");

    // 7) symmetrize to enforce Hermiticity
    let rho_hc = rho_ss.t().mapv(|c| c.conj());
    rho_ss = (&rho_ss + &rho_hc) * Complex64::new(0.5, 0.0);

    // 8) renormalize trace
    let tr: Complex64 = rho_ss.diag().iter().copied().sum();
    rho_ss *= Complex64::new(1.0, 0.0) / tr;

    // 9) diagonalize (Hermitian)
    let (eigvals, eigvecs) = rho_ss.eigh(UPLO::Lower)
        .expect("ρ_ss diagonalization failed");

    // 10) clamp tiny negatives
    let eigvals = eigvals.mapv(|x| if x < 0.0 { 0.0 } else { x });

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
    // println!("{:?}",probabilities);
    // println!("{}",eigvals_sum);
    // println!("{}",i);


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
            let subdir = format!("output/l{:.2}_s{}_b{:.2}/{:05}", lambda, s as i64, betawc, i / 1_000);
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
    // Fixed simulation parameters
    let dt: f64 = 0.001;            // dt = 10-3 ~20 n_ticks and after that does not increase for a threshold of 1100 and beta 2.0
    let total_time: f64 = 5000.0;        // Total time 5000 set it to have an average of 20 n_ticks for a threshold of 1100 and beta 2.0
    let omega_c: f64 = 0.01; // Frequency scale
    let beta: f64 = 0.1 / omega_c; // Inverse temperature
    let betawc = beta * omega_c;
    let gamma_z = 1. ;// 1./1000.*omega_c;
    let nb = 1./(betawc.exp() - 1.);
    
    let n_pts = 1_usize;

    let num_trajectories = 10;

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

    println!("Generation data completed successfully!");


    Ok(())
}
