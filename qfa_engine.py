import numpy as np
from typing import Tuple
from numba import jit
import logging

logging.getLogger('numba').setLevel(logging.WARNING)

@jit(nopython=True, fastmath=True)
def _multi_scale_scan_optimized(data_stream, sensitivity, decays, gain_autoscaling=True):
    """
    Static QFA Kernel
    """
    # forward scan, preparing the empty arrays
    n_points = len(data_stream)
    n_decays = len(decays)
    fidelity_trace = np.zeros(n_points, dtype=np.float64)
    coherence_trace = np.zeros(n_points, dtype=np.float64)
    # initialize the variables
    r00 = np.ones(n_decays, dtype=np.complex128)
    r01 = np.zeros(n_decays, dtype=np.complex128)
    r10 = np.zeros(n_decays, dtype=np.complex128)
    r11 = np.zeros(n_decays, dtype=np.complex128)
    # max angle
    max_angle = np.pi / 2.0
    # loop through the data stream
    for t in range(n_points):
        # get the value
        val = data_stream[t]
        # calculate the angle
        if gain_autoscaling:
            theta = max_angle * np.tanh((val * sensitivity) / max_angle)
        else:
            theta = val * sensitivity
        # calculate the cos and sin
        # basis of the gate, and the coherence we want to measure    
        c = np.cos(theta)
        s = np.sin(theta)
        c2, s2, cs = c*c, s*s, c*s
        # initialize the values we want to track
        min_fidelity = 1.0
        sum_fid = 0.0
        max_coh = 0.0
        # loop through the decays
        for k in range(n_decays):
            # get the decay
            d = decays[k]
            one_minus_d = 1.0 - d
            # get the previous values
            old_r00, old_r01, old_r10, old_r11 = r00[k], r01[k], r10[k], r11[k]
            # calculate the terms
            term_off = old_r01 + old_r10
            term_diag = old_r00 - old_r11
            # calculate the new values
            n00 = c2 * old_r00 - cs * term_off + s2 * old_r11
            n11 = s2 * old_r00 + cs * term_off + c2 * old_r11
            n01 = c2 * old_r01 + cs * term_diag - s2 * old_r10
            n10 = c2 * old_r10 + cs * term_diag - s2 * old_r01
            # update the values
            r00[k] = one_minus_d * n00 + d
            r01[k] = one_minus_d * n01
            r10[k] = one_minus_d * n10
            r11[k] = one_minus_d * n11
            # calculate the fidelity and coherence
            # fidelity is the real part of r00
            # coherence is the magnitude of r01
            current_fid = r00[k].real
            coh = np.sqrt(r01[k].real**2 + r01[k].imag**2)
            # update the values we want to track
            min_fidelity = min(min_fidelity, current_fid)
            sum_fid += current_fid
            max_coh = max(max_coh, coh)
        # calculate the final values    
        fidelity_trace[t] = 0.5 * (sum_fid/n_decays) + 0.5 * min_fidelity
        coherence_trace[t] = max_coh
        
    return fidelity_trace, coherence_trace


class MultiScaleQFA:
    """
    OPTIMIZED STATIC QFA ENGINE
    
    1. Static Decay: Robust noise suppression
    2. Single-Pass / Two-Pass: Flexible scanning
    3. Combination: MAX (Conservative Voting) - Rejects single-pass noise
       -> Too noisy for phase aligned work
    """
    def __init__(self, sensitivity: float, decays: list, gain_autoscaling: bool = True):
        self.sensitivity = float(sensitivity)
        self.decays = np.array(decays, dtype=np.float64)
        self.gain_autoscaling = gain_autoscaling
        
    def scan(self, data_stream: np.ndarray, bidirectional: bool = True) -> np.ndarray:
        # forward scan
        fwd_fid, _ = _multi_scale_scan_optimized(
            data_stream, self.sensitivity, self.decays, self.gain_autoscaling
        )
        if not bidirectional:
            return fwd_fid 
        # backward scan
        bwd_fid_raw, _ = _multi_scale_scan_optimized(
            data_stream[::-1], self.sensitivity, self.decays, self.gain_autoscaling
        )
        bwd_fid = bwd_fid_raw[::-1]
        # conservative combination (MAX) without shift
        # was a bit of trial by fire here
        return np.maximum(fwd_fid, bwd_fid)
    
    def scan_with_coherence(self, data_stream: np.ndarray, bidirectional: bool = True):
        # forward scan
        fwd_fid, fwd_coh = _multi_scale_scan_optimized(
            data_stream, self.sensitivity, self.decays, self.gain_autoscaling
        )
        if not bidirectional:
            return fwd_fid, fwd_coh
        # backward scan
        bwd_fid_raw, bwd_coh_raw = _multi_scale_scan_optimized(
            data_stream[::-1], self.sensitivity, self.decays, self.gain_autoscaling
        )
        bwd_fid = bwd_fid_raw[::-1]
        
        # return unshifted MAX
        return np.maximum(fwd_fid, bwd_fid), fwd_coh
