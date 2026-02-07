import argparse
import numpy as np
import pandas as pd
import logging
import time
import os
import glob
from pathlib import Path
from scipy.stats import median_abs_deviation
import concurrent.futures

from config import QFAConfig
from preprocessing import calculate_adaptive_sensitivity
from qfa_engine import MultiScaleQFA

# setup logging
logging.basicConfig(
    format='%(asctime)s - [QFA Tool] - %(message)s', 
    level=logging.INFO,
    datefmt='%H:%M:%S'
)

def binning_downsample(t, f, target_pct=15.0):
    """
    Adaptive Binning: Bins data to reduce point count to target_pct%.
    """
    n = len(t)
    target_n = int(n * (target_pct / 100.0))
    if target_n == 0: target_n = 1
    bin_size = max(1, n // target_n)
    n_bins = n // bin_size
    
    # reshape and mean
    # discards remainder points at end of array
    t_trim = t[:n_bins*bin_size]
    f_trim = f[:n_bins*bin_size]
    
    t_bin = t_trim.reshape(n_bins, bin_size).mean(axis=1)
    f_bin = f_trim.reshape(n_bins, bin_size).mean(axis=1)
    
    return t_bin, f_bin

def process_file(file_path, output_dir, config):
    """
    Runs QFA Augmented Binning Strategy on a single CSV file
    Strategy: Standard Binning (Baseline) + QFA Points (Detail).
    """
    filename = os.path.basename(file_path)
    
    try:
        # load data
        df = pd.read_csv(file_path)
        
        if 'time' not in df.columns or 'flux' not in df.columns:
            logging.warning(f"Skipping {filename}: Missing 'time' or 'flux'.")
            return

        t = df['time'].values
        f = df['flux'].values
        
        # NaNs
        if np.isnan(f).any():
            f = np.nan_to_num(f, nan=np.nanmedian(f))

        # normalization
        f_norm = f - np.median(f)
        mad = median_abs_deviation(f_norm, scale='normal')
        if mad < 1e-12: mad = 1e-12
        f_norm = f_norm / mad
        
        # qfa scan (calculate fidelity)
        adaptive_sens = calculate_adaptive_sensitivity(f_norm, config.sensitivity)
        engine = MultiScaleQFA(adaptive_sens, config.decays, config.gain_autoscaling)
        fidelity = engine.scan(f_norm, bidirectional=config.bidirectional_scan)
        
        # strategy: augmented binning
        
        # binning (baseline)
        # using 15%
        t_bin, f_bin = binning_downsample(t, f, target_pct=15.0)
        source_bin = np.zeros(len(t_bin), dtype=int) # 0 = Binning
        
        # qfa selection (detail)
        # 5%
        n_select = int(len(f) * (config.qfa_pct / 100.0))
        qfa_idx = np.argsort(fidelity)[:n_select]
        
        t_qfa = t[qfa_idx]
        f_qfa = f[qfa_idx]
        source_qfa = np.ones(len(t_qfa), dtype=int) # 1 = QFA
        
        # combine
        t_final = np.concatenate([t_bin, t_qfa])
        f_final = np.concatenate([f_bin, f_qfa])
        source_final = np.concatenate([source_bin, source_qfa])
        
        # sort (time order)
        sort_mask = np.argsort(t_final)
        t_final = t_final[sort_mask]
        f_final = f_final[sort_mask]
        source_final = source_final[sort_mask]
        
        # save results
        output_df = pd.DataFrame({
            'time': t_final,
            'flux': f_final,
            'source': source_final # 0=Bin, 1=QFA
        })
        
        output_path = os.path.join(output_dir, f"augmented_{filename}")
        output_df.to_csv(output_path, index=False)
        
        # logging.info(f"[{filename}] Processed. Points: {len(t)} -> {len(t_final)} (Comb)")
        
    except Exception as e:
        logging.error(f"Failed to process {filename}: {e}")

# main
def main():
    # arguments
    parser = argparse.ArgumentParser(description="QFA: Augmented Binning Pipeline")
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing flattened .csv files')
    parser.add_argument('--output_dir', type=str, default='./qfa_augmented_results', help='Directory to save results')
    parser.add_argument('--qfa_pct', type=float, default=5.0, help='Percentage of QFA points to keep (Default: 5.0)')
    parser.add_argument('--workers', type=int, default=os.cpu_count(), help='Number of parallel worker processes')
    
    # parse arguments
    args = parser.parse_args()
    config = QFAConfig()
    config.qfa_pct = args.qfa_pct # inject runtime override
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    files = glob.glob(os.path.join(args.input_dir, "*.csv"))
    if not files:
        logging.error(f"No .csv files found in {args.input_dir}")
        return
    
    logging.info(f"Starting Augmented Binning on {len(files)} files with {args.workers} workers.")
    
    start_time = time.time()
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_file, f, args.output_dir, config) for f in files]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logging.error(f"Worker Error: {e}")
                
    total_time = time.time() - start_time
    logging.info(f"Complete. Processed {len(files)} files in {total_time:.2f}s.")

if __name__ == "__main__":
    main()
