import numpy as np
from scipy.stats import median_abs_deviation

def calculate_adaptive_sensitivity(flux_array: np.ndarray, base_sensitivity: float = 0.03) -> float:
    """
    Calculates adaptive sensitivity based on the noise level of the data.
    
    Noisier data needs lower sensitivity to avoid triggering on noise.
    Quieter data can use higher sensitivity to catch subtle signals.
    
    Args:
        flux_array: Raw or normalized flux array.
        base_sensitivity: Baseline sensitivity for typical TESS noise.
        
    Returns:
        float: Adjusted sensitivity value.
    """
    # calculate robust noise estimate
    mad = median_abs_deviation(flux_array, scale='normal')
    
    # typical TESS noise is ~200 ppm, normalized MAD ~1.0
    # if MAD is higher, reduce sensitivity proportionally
    noise_factor = np.clip(1.0 / max(mad, 0.1), 0.5, 2.0)
    
    adaptive_sensitivity = base_sensitivity * noise_factor
    
    return adaptive_sensitivity
