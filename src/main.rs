use ndarray::{array, Array1, Array2, s};
use ndarray::linalg::kron;

use num_complex::Complex64;
use rand::Rng;

use rayon::prelude::*;
use plotters::prelude::*;

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

fn simulate_trajectory(
    gamma_p: f64,
    gamma_m: f64,
    lambda: f64,
    s: f64,
    dt: f64,
    t_max: f64,
) -> (Array1<f64>, Array1<usize>, Vec<Array1<Complex64>>) {
    let (l_plus, l_minus) = create_jump_operators(lambda, s);

    let h_eff = l_plus.dot(&l_minus).mapv(|x| x * Complex64::new(0.0, -0.5 * gamma_m / s)) 
    + l_minus.dot(&l_plus).mapv(|x| x * Complex64::new(0.0, -0.5 * gamma_p / s));
    
    let (_, psi1, psi2, eigvals) = steady_state(s, lambda, gamma_p, gamma_m);
    let mut rng = rand::thread_rng();
    let i = if rng.gen::<f64>() < eigvals[0] { 0 } else { 1 };
    let mut psi;
    
    let steps: usize = (t_max / dt).ceil() as usize;
    
    if i == 0 {
        psi = psi1.clone();
        psi /= psi.mapv(|e| e.conj()).dot(&psi).sqrt();
    } else {
        psi = psi2.clone();
        psi /= psi.mapv(|e| e.conj()).dot(&psi).sqrt();
    }

    let mut times = Vec::new();
    let mut types = Vec::new();
    let mut wfs = Vec::new();
    
    
    let mut r = rng.gen::<f64>();
    let mut q = rng.gen::<f64>();

    let mut p_p = 1.;
    let mut p_m = 1.;
    
    for i in 0..steps{
        
        
        let amp_m = psi.mapv(|e| e.conj()).dot(&l_plus.dot(&l_minus).dot(&psi));
        let amp_p = psi.mapv(|e| e.conj()).dot(&l_minus.dot(&l_plus).dot(&psi));
        
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
            
            times.push(i as f64 * dt);
            types.push(1);
            wfs.push(psi.clone());
        } else if q >= p_m {
            let dpsi_j_m = l_minus.dot(&psi).mapv(|x| x / (amp_m.re).sqrt());
            psi = dpsi_j_m;
            psi = &psi + &dpsi_nh;
            psi /= psi.mapv(|e| e.conj()).dot(&psi).sqrt();
                        
            q = rng.gen::<f64>();
            
            p_m = 1.;

            times.push(i as f64 * dt);
            types.push(0);
            wfs.push(psi.clone());
        } else {
            // No jump, just evolve
            psi = &psi + &dpsi_nh;
            psi /= psi.mapv(|e| e.conj()).dot(&psi).sqrt();
        }


        p_m *= 1.0 - prob_m;
        p_p *= 1.0 - prob_p;



    }

    // Convert times to Array1
    let times: Array1<f64> = Array1::from(times);
    let types: Array1<usize> = Array1::from(types);

    (times, types, wfs)
}


fn lindblad_simulation(s: f64, lambda: f64, gamma_p: f64, gamma_m: f64, total_time: f64, dt: f64) -> Vec<f64> {
    let max_steps = (total_time / dt).ceil() as usize;
    let mut sz_exp = Vec::with_capacity(max_steps);

    let (l_plus, l_minus) = create_jump_operators(lambda, s);


    // let sigma_pm = sigma_plus.dot(&sigma_minus);
    let identity = array![
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)]
    ];

    let v_identity = array![
        Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)
    ];

    let v_sigma_z = array![
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)],
        [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)],
        [Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0), Complex64::new(0.0, 0.0)],
        [Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0)]
    ];


    let term2_m = kron(&l_minus, &l_plus.t());
    let term2_p = kron(&l_plus, &l_minus.t());

    let left_p = l_minus.dot(&l_plus);
    let right_p = l_plus.t().dot(&l_minus.t());

    let left_m = l_plus.dot(&l_minus);
    let right_m = l_minus.t().dot(&l_plus.t());

    let term3_m = (kron(&left_m, &identity) + kron(&identity, &right_m)).mapv(|e| e * 0.5); 
    let term3_p = (kron(&left_p, &identity) + kron(&identity, &right_p)).mapv(|e| e * 0.5);

    let s_l = (&term2_m - &term3_m).mapv(|e| e * gamma_m / s) + (&term2_p - &term3_p).mapv(|e| e * gamma_p / s);

    let (rho,_,_,_) = steady_state(s, lambda, gamma_p, gamma_m);
    let mut v_rho: Array1<Complex64> = rho.iter().cloned().collect();

    for _ in 0..max_steps {
        v_rho = &v_rho + (&s_l.dot(&v_rho)).mapv(|e| e * dt);
        v_rho = v_rho.mapv(|e| e / v_identity.dot(&v_rho).re);

        let szz = v_identity.dot(&v_sigma_z.dot(&v_rho)).re;
        sz_exp.push(szz);
    }

    sz_exp
}

fn compute_tick_times(
    types: &Array1<usize>,
    a_minus: usize,
    a_plus: usize,
    m: usize,
) -> Array1<usize> {
    let mut n_acc: usize = 0;
    let mut aux_ticks = vec![];
    let mut next_threshold = m;

    for (i, &k) in types.iter().enumerate() {
        if k == 1 {
            n_acc += a_plus;
        } else {
            n_acc += a_minus;
        }

        if n_acc >= next_threshold {
            aux_ticks.push(i);
            next_threshold += m;
        }
    }

    let aux_ticks: Array1<usize> = Array1::from(aux_ticks);
    aux_ticks
}

fn analyze_ticks(
    times: &Array1<f64>,
    types: &Array1<usize>,
    wfs: &Vec<Array1<Complex64>>,
    aux_ticks: &Array1<usize>,
    beta: f64,
    omega_c: f64,
    pi: &Array2<Complex64>,
) -> (Vec<f64>, Vec<usize>, Vec<f64>) {
    let mut waiting_times = Vec::new();
    let mut activity_ticks = Vec::new();
    let mut entropy_ticks = Vec::new();

    let ticks = aux_ticks.as_slice().expect("aux_ticks must be contiguous");

    for pair in ticks.windows(2) {
        if pair.len() != 2 {
            continue; // Skip incomplete pair
        }
        let i1 = pair[0];
        let i2 = pair[1];

        let t1 = times[i1];
        let t2 = times[i2];

        let slice_types = &types.slice(s![i1..i2]);
        let n_plus = slice_types.iter().filter(|&&k| k == 1).count();
        let n_minus = slice_types.iter().filter(|&&k| k == 0).count();

        let q = omega_c * (n_minus as f64 - n_plus as f64);
        let beta_q = beta * q;

        let psi1 = &wfs[i1];
        let psi2 = &wfs[i2];

        let p1 = {
            let inner = pi.dot(psi1);
            let amp = psi1.mapv(|c| c.conj()).dot(&inner).re;
            amp.clamp(1e-12, 1.0)
        };
        let p2 = {
            let inner = pi.dot(psi2);
            let amp = psi2.mapv(|c| c.conj()).dot(&inner).re;
            amp.clamp(1e-12, 1.0)
        };

        let delta_spsi = p1.ln() - p2.ln();
        let s_tick = delta_spsi + beta_q;

        waiting_times.push(t2 - t1);
        activity_ticks.push(i2 - i1);
        entropy_ticks.push(s_tick);
    }
    

    (waiting_times, activity_ticks, entropy_ticks)
}

fn build_wtd(
    gamma_p: f64,
    gamma_m: f64,
    lambda: f64,
    s: f64,
    a_minus: usize,
    a_plus: usize,
    m: usize,
    beta: f64,
    omega_c: f64,
    n_traj: usize,
    dt: f64,
    t_max: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let (pi, _, _, _) = steady_state(s, lambda, gamma_p, gamma_m);

    // Phase 1: simulate in parallel and collect valid trajectories
    let trajectories: Vec<_> = (0..n_traj)
        .into_par_iter()
        .map(|_| simulate_trajectory(gamma_p, gamma_m, lambda, s, dt, t_max))
        .filter(|(times, _, _)| times.len() >= 2)
        .collect();

    // Phase 2: analyze each trajectory in parallel
    let results: Vec<(Vec<f64>, Vec<usize>, Vec<f64>)> = trajectories
        .into_par_iter()
        .filter_map(|(times, types, wfs)| {
            let aux_ticks = compute_tick_times(&types, a_minus, a_plus, m);
            if aux_ticks.len() > 1 {
                Some(analyze_ticks(&times, &types, &wfs, &aux_ticks, beta, omega_c, &pi))
            } else {
                None
            }
        })
        .collect();

    // Combine all results
    let mut all_waits = Vec::new();        // flattened for histogram
    let mut avg_acts_per_traj = Vec::new(); // one avg per trajectory
    let mut avg_ents_per_traj = Vec::new(); // one avg per trajectory

    for (waits, acts, ents) in results {
        all_waits.extend(waits); // flatten all waits

        // Average number of actions per trajectory
        if !acts.is_empty() {
            let avg_act = acts.iter().map(|x| *x as f64).sum::<f64>();
            avg_acts_per_traj.push(avg_act);
        }

        // Average entropy per trajectory
        if !ents.is_empty() {
            let avg_ent = ents[0];
            avg_ents_per_traj.push(avg_ent);
        }
    }

    (all_waits, avg_acts_per_traj, avg_ents_per_traj)
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
    let num_bins = ((max - min) / bin_width).ceil() as usize;
    let mut counts = vec![0; num_bins];

    for &value in data {
        if value >= min && value < max {
            let bin_index = ((value - min) / bin_width).floor() as usize;
            if bin_index < num_bins {
                counts[bin_index] += 1;
            }
        }
    }

    let total: f64 = counts.iter().sum::<usize>() as f64;
    let norm_counts: Vec<f64> = counts.iter().map(|&e| e as f64 / total).collect();

    norm_counts
}


fn plot_histogram(
    counts: &Vec<f64>,
    bin_width: f64,
    min: f64,
    max: f64,
    filename: &str,
) -> Result<(), Box<dyn std::error::Error>> {

    // Set up drawing area
    let root = BitMapBackend::new(filename, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let max_count = counts
        .iter()
        .cloned()
        .fold(0.0, f64::max);


    let mut chart = ChartBuilder::on(&root)
        .caption("Histogram", ("FiraCode Nerd Font", 40))
        .margin(30)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(min..max, 0.0..max_count)?;

    chart.configure_mesh()
        .x_desc("Value")
        .y_desc("Count")
        .draw()?;

    // Draw bars
    for (i, &count) in counts.iter().enumerate() {
        let x0 = min + i as f64 * bin_width;
        let x1 = x0 + bin_width;
        chart.draw_series(
            std::iter::once(Rectangle::new(
                [(x0, 0.), (x1, count)],
                BLUE.filled(),
            )),
        )?;
    }

    Ok(())
}

fn plot_trajectory_avg(
    // avg_cm: Array1<f64>,
    // avg_rj: Array1<f64>,
    lindblad_avg: Vec<f64>,
    steps: usize,
    filename: &str,
) -> Result<(), Box<dyn std::error::Error>> {

    let root = BitMapBackend::new(&filename, (1600, 1200)).into_drawing_area();
    root.fill(&WHITE)?;

    let min = lindblad_avg
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);

    let max = lindblad_avg
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    let mut chart = ChartBuilder::on(&root)
        .caption("Average <σ_z> trajectory", ("FiraCode Nerd Font", 30))
        .margin(100)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0..steps, (1.1 * min)..(1.1* max))?;

    chart.configure_mesh()
        .x_desc("Time steps")
        .y_desc("<σ_z>")
        .label_style(("FiraCode Nerd Font", 30).into_font())
        .draw()?;


    chart.draw_series(LineSeries::new(
        lindblad_avg.iter().enumerate().map(|(x, y)| (x, *y)),
        &MAGENTA,
    ))?
    .label("Avg")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &MAGENTA));

    chart.configure_series_labels()
    .position(SeriesLabelPosition::UpperRight)
    .label_font(("FiraCode Nerd Font", 40).into_font())
    .draw()?;

    Ok(())
}


fn main() -> Result<(), Box<dyn std::error::Error>>{
    let s: f64 = 50.;
    let lambda: f64 = 2.0;
    let omega_c: f64 = 0.01; // Frequency scale
    let beta: f64 = 2.0 / omega_c; // Inverse temperature
    let betawc = beta * omega_c;
    let gamma_z = 1. ;// 1./1000.*omega_c;
    let nb = 1./(betawc.exp() - 1.);
    let gamma_p: f64 = gamma_z/s * nb;
    let gamma_m: f64 = gamma_z/s * ( nb + 1.);

    let num_trajectories: usize = 1000; // Number of trajectories for waiting time
    let dt: f64 = 0.001;
    let t_max: f64 = 5000.0;

    let a_minus: usize = 1; // Weight for emission
    let a_plus: usize = 0; // Weight for absorption
    let m: usize = 1100; // Threshold for waiting time

    // Calculate number of steps as usize
    let steps: usize = (t_max / dt).ceil() as usize;

    
    let rho_sz = lindblad_simulation(s, lambda, gamma_p, gamma_m, t_max, dt);
    // println!("{:?}", rho_norm);

    let filename_traj = format!("Avg_trajectory__m-{}_omega_c-{}_dt-{}_tmax-{}_ntraj-{}.png", m, omega_c, dt, t_max, num_trajectories);
    plot_trajectory_avg(rho_sz, steps, &filename_traj)?;

    
    // --- 1. Generate data ---
    let (waits, _activities, entropies) = build_wtd(
        gamma_p,             // emission rate
        gamma_m,             // absorption rate
        lambda,            // lambda parameter
        s,              // spin size
        a_minus,        // weight for emission
        a_plus,         // weight for absorption
        m,              // threshold
        beta,           // inverse temperature
        omega_c,        // frequency scale
        num_trajectories,
        dt,         // number of trajectories
        t_max           // max time per trajectory
    );
    
    let arr_ent = Array1::from(entropies); // assuming entropies: Vec<f64>

    let mean_ent = (arr_ent.mapv(|e| (-e).exp()).sum().ln() - 
        (arr_ent.len() as f64).ln()).exp() ;// Mean of entropies, using log mean
    let std_dev_ent = arr_ent.std(0.0); // 0.0 = population std dev, use 1.0 for sample std dev


    // let arr_act = Array1::from(activities); // assuming entropies: Vec<f64>

    // let mean_act = (arr_act.mapv(|e| (-1.* e).exp()).sum().ln() - 
    //     (arr_act.len() as f64).ln()).exp() ;// Mean of entropies, using log mean
    // let std_dev_act = arr_act.std(0.0); // 0.0 = population std dev, use 1.0 for sample std dev

    println!("Mean of entropies: {}", mean_ent);
    println!("Standard deviation of entropies: {}", std_dev_ent);

    // println!("Mean of entropies: {}", mean_act);
    // println!("Standard deviation of entropies: {}", std_dev_act);

    // --- 2. Sort the waiting times ---
    let mut sorted_waits = waits.clone();
    sorted_waits.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    // --- 3. Compute bin width using IQR rule ---
    let bw = bin_width(&sorted_waits);
    
    // --- 4. Determine range ---
    let min = *sorted_waits.first().unwrap_or(&0.0);
    let max = *sorted_waits.last().unwrap_or(&1.0);
    
    // --- 5. Count frequencies per bin ---
    let counts = counts_per_bin(&sorted_waits, bw, min, max);
    
    // --- 6. Plot histogram ---
    let filename = format!("WTD-histogram__m-{}_omega_c-{}_dt-{}_tmax-{}_ntraj-{}.png", m, omega_c, dt, t_max, num_trajectories);
    plot_histogram(&counts, bw, min, max, &filename)?;
    
    println!("Simulation completed successfully!");
    

    Ok(())
}
