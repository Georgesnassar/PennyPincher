
# PennyPincher

## What is this?
QFA is a "smart filter" that sits *before* your standard analysis pipeline (BLS, MCMC, etc.). 
traditional binning destroys high-frequency transit details (ingress/egress). 
QFA uses a **Quantum-Inspired Density Matrix** algorithm to identify and preserve these high-fidelity points while binning the rest of the lightcurve.

**Result:** "Augmented Binning" — A lightcurve with the stability of binning but the precision of raw data at critical moments.

## The Algorithm
QFA treats the lightcurve as a stream of rotation operators acting on a qubit density matrix $\rho$.
*   **Flux $\to$ Rotation:** $R_y(\theta)$ where $\theta \propto \tanh(f)$.
*   **Memory $\to$ Decay:** Continuous amplitude damping channel.
*   **Detection:** High Fidelity ($\langle 0|\rho|0 \rangle \approx 1$) means "Steady State". Low Fidelity means "Anomaly/Transit".

## Classification
*   **Type:** Quantum-Inspired Classical Algorithm (Density Matrix Emulation).
*   **Complexity:** $O(N)$ (Linear Time).

## Running
1.  **Prepare Data:** Ensure your lightcurves are `.csv` files with `time` and `flux` columns. Esure they are detrended.
2.  **Run QFA:**
    ```bash
    python run_qfa.py --input_dir ./my_data --output_dir ./clean_data --qfa_pct 5.0
    ```
3.  **Output:** You will get `clean_data/augmented_star.csv`.
    *   `time`: Reduced timestamps.
    *   `flux`: The processed flux.
    *   `source`: `0` (Binned Baseline) or `1` (QFA Detail Point).
