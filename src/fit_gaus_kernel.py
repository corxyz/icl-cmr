# "Baseline": modeling attention scores by minimizing MSE to a gaussian function
import numpy as np
from scipy.optimize import curve_fit
from util import *

def gaus_kernel(x, a, b, mu, sigma):
    return a + b * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

def fit_gaus_kernel(attn_scores, max_lag=5):
  # fit the attention scores of a single head to a gaussian function by minimizing MSE
  # returns the MSE
  lags = np.arange(-max_lag, max_lag+1)
  mu0 = attn_scores.max()
  try:
    popt, _ = curve_fit(gaus_kernel, lags, attn_scores, p0=[0, 1, mu0, 1])
    fitted = gaus_kernel(lags, *popt)
  except:
    fitted = gaus_kernel(lags, attn_scores[max_lag], 0, 0, 1)   # flat line
  mse = np.mean((fitted - attn_scores) ** 2 / np.var(attn_scores))
  return mse

def fit_top_induction_head(n, head_scores, attn_scores):
  # fit the top n induction heads using a gaussian function
  # returns the MSE of individual heads
  all_attn_scores = attn_scores.reshape((-1, attn_scores.shape[-1]))
  head_scores = head_scores.flatten()
  top_heads = np.argpartition(head_scores.flatten(), -n)[-n:]
  mses_top_heads = np.array([fit_gaus_kernel(all_attn_scores[i]) for i in top_heads])
  return mses_top_heads
