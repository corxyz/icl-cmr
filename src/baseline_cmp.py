import numpy as np
import joblib
from scipy.stats import ttest_ind
from fit_attn_score import fit_head_cmr
from fit_gaus_kernel import fit_top_induction_head as fit_top_induction_head_guas
from fit_gaus_kernel import fit_gaus_kernel
from util import *

def fit_top_induction_head(n, head_scores, attn_scores):
  # fit the top n induction heads by minimizing CRP distance
  # returns the CRP distance of individual heads
  all_attn_scores = attn_scores.reshape((-1, attn_scores.shape[-1]))
  head_scores = head_scores.flatten()
  top_heads = np.argpartition(head_scores.flatten(), -n)[-n:]
  mses_top_heads = np.array([fit_head_cmr(all_attn_scores[i])['MSE'] for i in top_heads])
  return mses_top_heads

def model_cmp(model_name='llama3-8b', select_range=5):
  # compare CMR and Gaussian function w.r.t. the top 1%, 5%, 10%, and 20% induction heads of a specific model
  all_head_scores = joblib.load('./saved_scores/llama3-8b/induction_head_all_scores_{}.pkl'.format(model_name))
  attn_scores = load_scores_in_range(all_head_scores, 'sorted_CRP_scores', select_range=select_range)
  n_heads = attn_scores.shape[0] * attn_scores.shape[1]
  for n in [int(n_heads * 0.01), int(n_heads * 0.05), int(n_heads * 0.1), int(n_heads * 0.2)]:
    mse = fit_top_induction_head(n, all_head_scores['head_scores'], attn_scores)
    mse_gaus = fit_top_induction_head_guas(n, all_head_scores['head_scores'], attn_scores)
    print(n, ttest_ind(mse, mse_gaus))

def pythia_top_cmp(n, select_range=5):
  # compare CMR and Gaussian function w.r.t. the top n induction heads across Pythia models
  # (n can be 20, 50, 100, or 200)
  top_induction_heads = joblib.load('saved_top_heads/cmr_top_head_score.pkl')
  cmr_dist, gaus_mse = [], []
  for model_name, labels in top_induction_heads['top{}'.format(n)]['selection'].items():
    all_head_scores = joblib.load('./saved_scores/{}/induction_head_all_scores_{}_cp142.pkl'.format(model_name, '_'.join(model_name.split('-')[:2])))
    for label in labels:
      ind = np.where(all_head_scores['labels'] == label)
      attn_scores = all_head_scores['CRP_scores'][ind].squeeze()
      max_lag = attn_scores.shape[-1] // 2
      og_range = np.arange(-max_lag, max_lag)
      load_range = (-select_range <= og_range) & (og_range <= select_range)
      attn_scores = attn_scores[load_range]
      cmr_dist.append(fit_head_cmr(attn_scores)['MSE'])
      gaus_mse.append(fit_gaus_kernel(attn_scores, max_lag=select_range))
  print(n, np.mean(cmr_dist), np.mean(gaus_mse), ttest_ind(cmr_dist, gaus_mse))

pythia_top_cmp(20)
pythia_top_cmp(50)
pythia_top_cmp(100)
pythia_top_cmp(200)