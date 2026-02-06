from dataclasses import dataclass

@dataclass
class QFAConfig:
    """
    Configuration for the Quantum Finite Automaton pipeline.
    
    Attributes:
        sensitivity (float): Gain factor. Higher = detects fainter signals, 
                             but increases noise sensitivity. Recommended: 0.1 - 0.2.
        decay (float): Memory Horizon (Lambda). 
                       0.1 = Short memory (~10 steps). 
                       0.01 = Long memory (~100 steps).
        trigger_threshold (float): Fidelity score below which an anomaly is flagged.
        window_size (int): Rolling window size for background trend subtraction.
    """
    # sensitivity (gain): higher = detects smaller planets, but more noise.
    # optimized "robust universal" value: 0.03 (filters active star noise, fast decay catches earths)
    sensitivity: float = 0.03
    decay: float = 0.05
    trigger_threshold: float = 0.70
    window_size: int = 201
    
    # gain autoscaling: automatically adjusts sensitivity based on the noise floor of the data.
    # this enables "zero-shot" detection for both quiet and noisy stars.
    gain_autoscaling: bool = True
    
    # multi-scale decays: the memory horizons for the parallel qfas.
    # [fast (earth), medium (neptune), long (jupiter)]
    decays: tuple = (0.2, 0.1, 0.05, 0.025, 0.01)

    # bidirectional scanning: runs the qfa twice (left->right, right->left)
    # and combines them to eliminate phase lag and sharpen detections.
    # cost: ~2x engine runtime (negligible compared to i/o).
    bidirectional_scan: bool = True
    
    # --- Advanced Thresholding & Heuristics Parameters ---
    
    # initial sigma for thresholding (fidelity < median - sigma * mad)
    initial_sigma: float = 3.0
    
    # density checks (clustering detection)
    min_points_for_clustering: int = 20
    short_period_gap_threshold: int = 100  # points (~3 hours at 2-min cadence)
    
    # target densities for relaxation loop
    target_density_short_period: float = 0.12  # 12%
    target_density_long_period: float = 0.08   # 8%
    
    # relaxation loop limits
    min_sigma_limit: float = 1.5
    max_relaxation_attempts: int = 5
    sigma_step: float = 0.3
    
    # temporal continuity filter
    continuity_min_duration: int = 2
    continuity_max_gap: int = 15
    
    # baseline mixing for bls context
    baseline_pct_short_period: float = 0.15
    baseline_pct_long_period: float = 0.10
