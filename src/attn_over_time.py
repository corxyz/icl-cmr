# fit all attention heads of a given model by minimizing CMR distance over training
# reproduces Figure 7b, Supplementary Figure S2

import argparse
import numpy as np
import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
from fit_attn_score import fit
from util import *

########################################
# Define constants & hyperparameters
########################################

EPSILON = 1e-15
cm = 1/2.54

max_dist = 8                            # magnitude of maximum CRP relative lag computed
select_range = 5                        # magnitude of maximum CRP relative lag considered for fitting

beta_gridsize = 1/20                    # size of each grid for grid search of beta
gamma_gridsize = 1/10                   # size of each grid for grid search of gamma

mse_thres = 0.5                         # threshold of CMR distance (only plot results for heads under the threshold)

cp_interv = 10                          # checkpoint interval
n_cp = 143                              # number of checkpoints

########################################
# Define CLI parser
########################################

parser = argparse.ArgumentParser(description='Fit attention scores with CMR (over training).')
parser.add_argument('-m', '--model_name', type=str, default='pythia-70m-deduped-v0',
                    help='Model name')

########################################
# Load pre-computed CRP
########################################

crp = np.load('./saved_crps/cmr_crp_avg_20_11.npy')
crp_sem = np.load('./saved_crps/cmr_crp_sem_20_11.npy')
crp_params = np.mgrid[beta_gridsize:1+beta_gridsize:beta_gridsize, 0:1+beta_gridsize:beta_gridsize, 
                      0:1+gamma_gridsize:gamma_gridsize]
crp_range = np.arange(-max_dist, max_dist+1)

########################################
# Model fitting
########################################

def fit_over_time(model_name, file_prefix):
  cps = list(range(0, n_cp, cp_interv)) + [n_cp - 1]
  mses, params, scale_factors = None, None, None
  copy_scores, head_scores = None, None
  labels = None

  for i, cp in enumerate(cps):
    all_head_scores = joblib.load('./saved_scores/{}/{}{}.pkl'.format(model_name, file_prefix, cp))
    n_layers, n_heads = all_head_scores['sorted_labels'].shape
    attn_scores = load_scores_in_range(all_head_scores, 'sorted_CRP_scores', select_range=select_range)
    fit_res = fit(attn_scores)

    if i == 0:
      mses = np.zeros((len(cps), n_layers, n_heads))
      params = np.zeros((len(cps), n_layers, n_heads, len(crp.shape)-1))
      scale_factors = np.zeros((len(cps), n_layers, n_heads))
      copy_scores = np.zeros((len(cps), n_layers, n_heads))
      head_scores = np.zeros((len(cps), n_layers, n_heads))
      labels = all_head_scores['sorted_labels']
    mses[i] = fit_res['fitted_MSE']
    params[i] = fit_res['fitted_params']
    scale_factors[i] = fit_res['fitted_scale_factors']
    copy_scores[i] = all_head_scores['sorted_copying_score']
    head_scores[i] = all_head_scores['sorted_head_scores']

  fit_res = {
    'fitted_MSE': mses,
    'fitted_params': params,
    'fitted_scale_factors': scale_factors,
    'copy_scores': copy_scores,
    'head_scores': head_scores,
    'labels': labels, 
    'checkpoints': cps,
  }
  return fit_res

########################################
# Plotting functions
########################################

def plot_fit_over_time(time, mses):
  n_layers = mses.shape[1]
  # plot overall trend
  plt.figure(figsize=[9*cm, 6*cm], dpi=300)
  ax = plt.gca()
  ax.tick_params(axis='both', labelsize=5)
  for l in range(n_layers):
    mask = [~np.isnan(mses[i,l]) for i in range(mses.shape[0])]
    low_mse_pct = [(mses[i,l,mask[i]] < mse_thres).mean()*100 for i in range(mses.shape[0])]
    ls = plt.plot(time, low_mse_pct, lw=1, color=mpl.cm.gnuplot(l / n_layers), label='L{}'.format(l))
  plt.xlabel('Checkpoint', fontsize=5)
  plt.ylabel('% CMR distance < {}'.format(mse_thres), fontsize=5)
  plt.legend(bbox_to_anchor=(1.01, 0), loc='lower left', fontsize=5, ncol=2)
  plt.tight_layout(pad=0.3)
  plt.savefig('./figs/{}/fit_over_time_all_thres{}.pdf'.format(model_name, mse_thres))

def plot_params_over_time(time, params, mses):
  n_layers = params.shape[1]
  # plot overall trend
  assert(type(mses) in [list, np.ndarray])
  _, axes = plt.subplots(params.shape[1] // 2, 6, figsize=[4*3*cm, params.shape[1]//2*2*cm], dpi=300, sharex=True, sharey=True)
  for l in range(n_layers):
    # filter based on model fit at the last checkpoint
    mask = mses[-1,l,:] < mse_thres
    layer_enc_avg = np.array([np.nanmean(params[i,l,mask,0]) for i in range(mses.shape[0])])
    layer_enc_min = np.array([np.min(params[i,l,mask,0]) if mask.sum() > 0 else np.nan for i in range(mses.shape[0])])
    layer_enc_max = np.array([np.max(params[i,l,mask,0]) if mask.sum() > 0 else np.nan for i in range(mses.shape[0])])
    layer_rec_avg = np.array([np.nanmean(params[i,l,mask,1]) for i in range(mses.shape[0])])
    layer_rec_min = np.array([np.nanmin(params[i,l,mask,1]) if mask.sum() > 0 else np.nan for i in range(mses.shape[0])])
    layer_rec_max = np.array([np.nanmax(params[i,l,mask,1]) if mask.sum() > 0 else np.nan for i in range(mses.shape[0])])
    layer_gft_avg = np.array([np.nanmean(params[i,l,mask,2]) for i in range(mses.shape[0])])
    layer_gft_min = np.array([np.nanmin(params[i,l,mask,2]) if mask.sum() > 0 else np.nan for i in range(mses.shape[0])])
    layer_gft_max = np.array([np.nanmax(params[i,l,mask,2]) if mask.sum() > 0 else np.nan for i in range(mses.shape[0])])
    ls = axes[l//2, l%2].plot(time, layer_enc_avg, color=mpl.cm.gnuplot(l / n_layers), lw=0.4)
    axes[l//2, l%2].fill_between(time, layer_enc_min, layer_enc_max, color=mpl.cm.gnuplot(l / n_layers), alpha=0.2)
    axes[l//2, l%2].tick_params(axis='both', labelsize=5)
    axes[l//2, l%2].set_title('L{}'.format(l), fontsize=5)
    if l % 2 == 0:
      axes[l//2, l%2].set_ylabel('Avg ' r'$\beta_{\rm enc}$', fontsize=5)
    ls = axes[l//2, l%2+2].plot(time, layer_rec_avg, color=mpl.cm.gnuplot(l / n_layers), lw=0.4)
    axes[l//2, l%2+2].fill_between(time, layer_rec_min, layer_rec_max, color=mpl.cm.gnuplot(l / n_layers), alpha=0.2)
    axes[l//2, l%2+2].tick_params(axis='both', labelsize=5)
    axes[l//2, l%2+2].set_title('L{}'.format(l), fontsize=5)
    if l % 2 == 0:
      axes[l//2, l%2+2].set_ylabel('Avg ' r'$\beta_{\rm rec}$', fontsize=5)
    ls = axes[l//2, l%2+4].plot(time, layer_gft_avg, color=mpl.cm.gnuplot(l / n_layers), lw=0.4)
    axes[l//2, l%2+4].fill_between(time, layer_gft_min, layer_gft_max, color=mpl.cm.gnuplot(l / n_layers), alpha=0.2)
    axes[l//2, l%2+4].tick_params(axis='both', labelsize=5)
    axes[l//2, l%2+4].set_title('L{}'.format(l), fontsize=5)
    if l % 2 == 0:
      axes[l//2, l%2+4].set_ylabel('Avg ' r'$\gamma_{\rm FT}$', fontsize=5)
  for i in range(axes.shape[-1]):
    axes[-1,i].set_xlabel('Checkpoint', fontsize=5)
  plt.tight_layout(pad=0.3)
  plt.savefig('./figs/{}/params_over_time_all_thres{}.pdf'.format(model_name, mse_thres))

def plot_scale_factors_over_time(time, scale_factors, mses):
  n_layers = mses.shape[1]
  # plot overall trend
  plt.figure(figsize=[6*cm, 3*cm], dpi=300)
  ax = plt.gca()
  ax.tick_params(axis='both', labelsize=5)
  for l in range(n_layers):
    # filter based on model fit at the last checkpoint
    mask = mses[-1,l,:] < mse_thres
    layer_avg = np.array([np.nanmean(scale_factors[i,l,mask]) for i in range(mses.shape[0])])
    layer_min = np.array([np.min(scale_factors[i,l,mask]) if mask.sum() > 0 else np.nan for i in range(mses.shape[0])])
    layer_max = np.array([np.max(scale_factors[i,l,mask]) if mask.sum() > 0 else np.nan for i in range(mses.shape[0])])
    plt.plot(time, layer_avg, lw=0.4, color=mpl.cm.gnuplot(l / n_layers), label='L{}'.format(l))
    # plt.fill_between(time, layer_min, layer_max, color=mpl.cm.gnuplot(l / n_layers), alpha=0.2)
  plt.xlabel('Checkpoint', fontsize=5)
  plt.ylabel('Average ' r'$\tau^{-1}$', fontsize=5)
  plt.legend(bbox_to_anchor=(1.01, 0), loc='lower left', fontsize=5, ncol=2)
  plt.tight_layout(pad=0.3)
  plt.savefig('./figs/{}/scale_factors_over_time_thres{}.pdf'.format(model_name, mse_thres))

def plot_scores_over_time(time, head_scores, copy_scores, mses):
  n_layers = head_scores.shape[1]
  # plot overall trend
  assert(type(mses) in [list, np.ndarray])
  _, axes = plt.subplots(head_scores.shape[1] // 2, 4, figsize=[4*3*cm, head_scores.shape[1]//2*3*cm], dpi=300, sharex=True, sharey=True)
  for l in range(n_layers):
    # filter based on model fit at the last checkpoint
    mask = mses[-1,l,:] < mse_thres
    layer_head_avg = np.array([head_scores[i,l,mask].mean() for i in range(mses.shape[0])])
    layer_head_min = np.array([head_scores[i,l,mask].min() if mask.sum() > 0 else np.nan for i in range(mses.shape[0])])
    layer_head_max = np.array([head_scores[i,l,mask].max() if mask.sum() > 0 else np.nan for i in range(mses.shape[0])])
    layer_copy_avg = np.array([copy_scores[i,l,mask].mean() for i in range(mses.shape[0])])
    layer_copy_min = np.array([copy_scores[i,l,mask].min() if mask.sum() > 0 else np.nan for i in range(mses.shape[0])])
    layer_copy_max = np.array([copy_scores[i,l,mask].max() if mask.sum() > 0 else np.nan for i in range(mses.shape[0])])
    axes[l//2, l%2].plot(time, layer_head_avg, lw=0.4, color=mpl.cm.gnuplot(l / n_layers))
    axes[l//2, l%2].fill_between(time, layer_head_min, layer_head_max, color=mpl.cm.gnuplot(l / n_layers), alpha=0.2)
    axes[l//2, l%2].tick_params(axis='both', labelsize=5)
    axes[l//2, l%2].set_title('L{}'.format(l), fontsize=5)
    if l % 2 == 0:
      axes[l//2, l%2].set_ylabel('Average head score', fontsize=5)
    axes[l//2, l%2+2].plot(time, layer_copy_avg, lw=0.4, color=mpl.cm.gnuplot(l / n_layers))
    axes[l//2, l%2+2].fill_between(time, layer_copy_min, layer_copy_max, color=mpl.cm.gnuplot(l / n_layers), alpha=0.2)
    axes[l//2, l%2+2].tick_params(axis='both', labelsize=5)
    axes[l//2, l%2+2].set_title('L{}'.format(l), fontsize=5)
    if l % 2 == 0:
      axes[l//2, l%2+2].set_ylabel('Average copying score', fontsize=5)
  for i in range(axes.shape[-1]):
    axes[-1,i].set_xlabel('Checkpoint', fontsize=5)
  plt.tight_layout(pad=0.3)
  plt.savefig('./figs/{}/scores_over_time_thres{}.pdf'.format(model_name, mse_thres))

if __name__ == '__main__':
  # parse command line arguments
  args = parser.parse_args()
  model_name = args.model_name
  file_prefix = 'all_scores_' + '_'.join(model_name.split('-')[:2]) + '_cp'

  # fit all average attention scores for each head over training
  fit_res = fit_over_time(model_name, file_prefix)
  mses = fit_res['fitted_MSE']
  params = fit_res['fitted_params']
  scale_factors = fit_res['fitted_scale_factors']
  copy_scores = fit_res['copy_scores']
  head_scores = fit_res['head_scores']
  labels = fit_res['labels']
  cps = fit_res['checkpoints']
  
  # plot results
  plot_fit_over_time(cps, mses)
  plot_params_over_time(cps, params, mses)
  plot_scale_factors_over_time(cps, scale_factors, mses)
  plot_scores_over_time(cps, head_scores, copy_scores, mses)
