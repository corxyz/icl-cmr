# Utility functions

import numpy as np

def load_scores_in_range(all_head_scores, score_key, select_range=5):
  # load scores from the recorded attention heads (e.g., attention scores, sem of attention scores)
  # narrow down to the given range
  attn_scores = all_head_scores[score_key]
  max_lag = attn_scores.shape[-1] // 2
  ind = np.arange(-max_lag, max_lag)
  attn_range = (-select_range <= ind) & (ind <= select_range)
  return attn_scores[:,:,attn_range]

def truncate_crp_to_range(crp, select_range=5):
  # truncate pre-computed CRP to the given range
  max_lag = crp.size // 2
  ind = np.arange(-max_lag, max_lag+1)
  select_idx = (-select_range <= ind) & (ind <= select_range)
  return crp[select_idx]

def load_head_labels(all_head_scores):
  # load all head labels
  return all_head_scores['labels']
