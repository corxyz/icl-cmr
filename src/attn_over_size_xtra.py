import math, argparse
import numpy as np
import joblib
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

size_dict = {                           # size (number of parameters) of each model in 10m
  'mistral-7b': 780,
  'qwen-7b': 650,
  'llama3-8b': 780,
}

shape_dict = {                          # (n_layers, n_heads) of each model
  'mistral-7b': (32,32),
  'qwen-7b': (32,32),
  'llama3-8b': (32,32),
}

# plotting colors
colors = ['blue', 'orange', 'green']

########################################
# Define CLI parser
########################################

parser = argparse.ArgumentParser(description='Fit attention scores with CMR (compare across models).')
parser.add_argument('-m', '--models', type=str, nargs='+', 
                    default=['qwen-7b','mistral-7b','llama3-8b'],
                    help='Model name(s)')

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

def fit_over_size(models=['qwen-7b','mistral-7b','llama3-8b']):
  n_layers_max, n_heads_max = max(shape_dict.values(), key=lambda sub: sub[0])
  mses = np.zeros((len(models), n_layers_max, n_heads_max))
  params = np.zeros((len(models), n_layers_max, n_heads_max, len(crp.shape)-1))
  scale_factors = np.zeros((len(models), n_layers_max, n_heads_max))
  mses[:] = np.nan
  params[:] = np.nan
  scale_factors[:] = np.nan
  copy_scores, head_scores = np.zeros((len(models), n_layers_max, n_heads_max)), np.zeros((len(models), n_layers_max, n_heads_max))
  copy_scores[:] = np.nan
  head_scores[:] = np.nan
  labels = np.zeros((n_layers_max, n_heads_max))

  for i, s in enumerate(models):
    all_head_scores = joblib.load('./saved_scores/{}/induction_head_all_scores_{}.pkl'.format(s, s))
    n_layers, n_heads = all_head_scores['sorted_labels'].shape
    attn_scores = load_scores_in_range(all_head_scores, 'sorted_CRP_scores', select_range=select_range)
    fit_res = fit(attn_scores)

    if n_layers == n_layers_max and n_heads == n_heads_max:
      labels = all_head_scores['sorted_labels']
    mses[i, :n_layers, :n_heads] = fit_res['fitted_MSE']
    params[i, :n_layers, :n_heads] = fit_res['fitted_params']
    scale_factors[i, :n_layers, :n_heads] = fit_res['fitted_scale_factors']
    copy_scores[i, :n_layers, :n_heads] = all_head_scores['sorted_copying_score']
    head_scores[i, :n_layers, :n_heads] = all_head_scores['sorted_head_scores']
  
  fit_res = {
    'fitted_MSE': mses,
    'fitted_params': params,
    'fitted_scale_factors': scale_factors,
    'copy_scores': copy_scores,
    'head_scores': head_scores,
    'labels': labels, 
    'sizes': models,
  }
  return fit_res

########################################
# Plotting functions
########################################

def plot_fit_over_size(models, mses, labels, rel_layer=False, 
                       use_thres=False, plot_individual_head=False):
  n_layers = mses.shape[1]

  # plot overall trend
  plt.figure(figsize=[6*cm, 4*cm], dpi=300)
  ax = plt.gca()
  ax.tick_params(axis='both', labelsize=5)
  ax.set_prop_cycle(color=colors)
  x = [np.arange(n_layers) / shape_dict[s][0] * 100 for s in sizes]
  xlabel = 'Relative layer position (%)'

  for i, s in enumerate(sizes):
    mask = [~np.isnan(mses[i,l]) for l in range(mses.shape[1])]
    low_mse_pct = [(mses[i,l,mask[l]] < mse_thres).mean()*100 for l in range(mses.shape[1])]
    plt.plot(x[i], low_mse_pct, lw=1, label=s)
  plt.xlim([0,100])
  plt.xlabel(xlabel, fontsize=5)
  plt.ylabel('% CMR distance < {}'.format(mse_thres), fontsize=5)
  plt.legend(bbox_to_anchor=(1.01, 0), loc='lower left', fontsize=5)
  plt.tight_layout()
  plt.savefig('./figs/cmp_xtra/fit_over_size_all_thres{}.pdf'.format(mse_thres))

def plot_params_over_size(sizes, params, mses):
  n_layers = params.shape[1]
  
  # plot overall trend
  x = [np.arange(n_layers) / shape_dict[s][0] * 100 for s in sizes]
  xlabel = 'Relative layer position (%)'

  assert(type(mses) in [list, np.ndarray])
  _, axes = plt.subplots(len(sizes), 3, figsize=[6*cm, len(sizes)*3*cm], dpi=300, sharey='row', sharex='col')
  for i, s in enumerate(sizes):
    mask = [mses[i,l,:] < mse_thres for l in range(mses.shape[1])]
    model_enc_avg = np.array([np.nanmean(params[i,l,mask[l],0]) for l in range(mses.shape[1])])
    model_enc_min = np.array([np.min(params[i,l,mask[l],0]) if mask[l].sum() > 0 else np.nan for l in range(mses.shape[1])])
    model_enc_max = np.array([np.max(params[i,l,mask[l],0]) if mask[l].sum() > 0 else np.nan for l in range(mses.shape[1])])
    model_rec_avg = np.array([np.nanmean(params[i,l,mask[l],1]) for l in range(mses.shape[1])])
    model_rec_min = np.array([np.nanmin(params[i,l,mask[l],1]) if mask[l].sum() > 0 else np.nan for l in range(mses.shape[1])])
    model_rec_max = np.array([np.nanmax(params[i,l,mask[l],1]) if mask[l].sum() > 0 else np.nan for l in range(mses.shape[1])])
    model_gft_avg = np.array([np.nanmean(params[i,l,mask[l],2]) for l in range(mses.shape[1])])
    model_gft_min = np.array([np.nanmin(params[i,l,mask[l],2]) if mask[l].sum() > 0 else np.nan for l in range(mses.shape[1])])
    model_gft_max = np.array([np.nanmax(params[i,l,mask[l],2]) if mask[l].sum() > 0 else np.nan for l in range(mses.shape[1])])
    ls = axes[i,0].plot(x[i], model_enc_avg, 'k-', lw=0.4, label=s)
    axes[i,0].fill_between(x[i], model_enc_min, model_enc_max, color=ls[0].get_color(), alpha=0.2)
    axes[i,0].tick_params(axis='both', labelsize=5)
    axes[i,0].set_ylabel('Avg ' r'$\beta_{\rm enc}$', fontsize=5)
    axes[i,0].set_title('Pythia-{}-deduped-v0'.format(s), fontsize=5)
    ls = axes[i,1].plot(x[i], model_rec_avg, 'k-', lw=0.4, label=s)
    axes[i,1].fill_between(x[i], model_rec_min, model_rec_max, color=ls[0].get_color(), alpha=0.2)
    axes[i,1].tick_params(axis='both', labelsize=5)
    axes[i,1].set_ylabel('Avg ' r'$\beta_{\rm rec}$', fontsize=5)
    axes[i,1].set_title('Pythia-{}-deduped-v0'.format(s), fontsize=5)
    ls = axes[i,2].plot(x[i], model_gft_avg, 'k-', lw=0.4, label=s)
    axes[i,2].fill_between(x[i], model_gft_min, model_gft_max, color=ls[0].get_color(), alpha=0.2)
    axes[i,2].tick_params(axis='both', labelsize=5)
    axes[i,2].set_ylabel('Avg ' r'$\gamma_{\rm FT}$', fontsize=5)
    axes[i,2].set_title('Pythia-{}-deduped-v0'.format(s), fontsize=5)
  axes[-1,0].set_xlabel(xlabel, fontsize=5)
  axes[-1,1].set_xlabel(xlabel, fontsize=5)
  for ax in axes.flatten(): 
    ax.set_xlim([0,100])
  plt.tight_layout(pad=0.3)
  plt.savefig('./figs/cmp_xtra/params_over_size_all_thres{}.pdf'.format(mse_thres))

def plot_scale_factors_over_size(sizes, scale_factors, mses):
  # plot individual heads
  n_layers = scale_factors.shape[1]

  # plot overall trend
  plt.figure(figsize=[6*cm, 6*cm], dpi=300)
  ax = plt.gca()
  ax.tick_params(axis='both', labelsize=5)
  ax.set_prop_cycle(color=colors)
  x = [np.arange(n_layers) / shape_dict[s][0] * 100 for s in sizes]
  xlabel = 'Relative layer position (%)'

  assert(type(mses) in [list, np.ndarray])
  for i, s in enumerate(sizes):
    mask = [mses[i,l,:] < mse_thres for l in range(mses.shape[1])]
    model_avg = np.array([np.nanmean(scale_factors[i,l,mask[l]]) if mask[l].sum() > 0 else np.nan for l in range(mses.shape[1])])
    model_min = np.array([np.min(scale_factors[i,l,mask[l]]) if mask[l].sum() > 0 else np.nan for l in range(mses.shape[1])])
    model_max = np.array([np.max(scale_factors[i,l,mask[l]]) if mask[l].sum() > 0 else np.nan for l in range(mses.shape[1])])
    ls = plt.plot(x[i], model_avg, lw=0.4, label=s)
    # plt.fill_between(x[i], model_min, model_max, color=ls[0].get_color(), alpha=0.2)
  plt.xlim([0,100])
  plt.xlabel(xlabel, fontsize=5)
  plt.ylabel('Average ' r'$\tau^{-1}$', fontsize=5)
  plt.legend(fontsize=5)
  plt.tight_layout()
  plt.savefig('./figs/cmp_xtra/scale_factors_over_size_thres{}.pdf'.format(mse_thres))

def plot_scores_over_size(sizes, head_scores, copy_scores, mses):
  n_layers = head_scores.shape[1]
  
  # plot overall trend
  x = [np.arange(n_layers) / shape_dict[s][0] * 100 for s in sizes]
  xlabel = 'Relative layer position (%)'

  assert(type(mses) in [list, np.ndarray])
  _, axes = plt.subplots(len(sizes), 2, figsize=[6*cm, len(sizes)*3*cm], dpi=300, sharey='row', sharex='col')
  for i, s in enumerate(sizes):
    mask = [mses[i,l,:] < mse_thres for l in range(mses.shape[1])]
    model_hs_avg = np.array([np.nanmean(head_scores[i,l,mask[l]]) for l in range(mses.shape[1])])
    model_hs_min = np.array([np.min(head_scores[i,l,mask[l]]) if mask[l].sum() > 0 else np.nan for l in range(mses.shape[1])])
    model_hs_max = np.array([np.max(head_scores[i,l,mask[l]]) if mask[l].sum() > 0 else np.nan for l in range(mses.shape[1])])
    model_cs_avg = np.array([np.nanmean(copy_scores[i,l,mask[l]]) for l in range(mses.shape[1])])
    model_cs_min = np.array([np.nanmin(copy_scores[i,l,mask[l]]) if mask[l].sum() > 0 else np.nan for l in range(mses.shape[1])])
    model_cs_max = np.array([np.nanmax(copy_scores[i,l,mask[l]]) if mask[l].sum() > 0 else np.nan for l in range(mses.shape[1])])
    ls = axes[i,0].plot(x[i], model_hs_avg, 'k-', lw=0.4, label=s)
    axes[i,0].fill_between(x[i], model_hs_min, model_hs_max, color=ls[0].get_color(), alpha=0.2)
    axes[i,0].tick_params(axis='both', labelsize=5)
    axes[i,0].set_ylabel('Average head score', fontsize=5)
    axes[i,0].set_title('Pythia-{}-deduped-v0'.format(s), fontsize=5)
    ls = axes[i,1].plot(x[i], model_cs_avg, 'k-', lw=0.4, label=s)
    axes[i,1].fill_between(x[i], model_cs_min, model_cs_max, color=ls[0].get_color(), alpha=0.2)
    axes[i,1].tick_params(axis='both', labelsize=5)
    axes[i,1].set_ylabel('Average copying score', fontsize=5)
    axes[i,1].set_title('Pythia-{}-deduped-v0'.format(s), fontsize=5)
  axes[-1,0].set_xlabel(xlabel, fontsize=5)
  axes[-1,1].set_xlabel(xlabel, fontsize=5)
  for ax in axes.flatten(): 
    ax.set_xlim([0,100])
  plt.tight_layout(pad=0.3)
  plt.savefig('./figs/cmp_xtra/scores_over_size_all_thres{}.pdf'.format(mse_thres))

if __name__ == '__main__':
  # parse command line arguments
  args = parser.parse_args()
  models = args.models

  # fit all average attention scores for each head as a function of relative layers
  # compare additional models
  fit_res = fit_over_size(models=models)
  mses = fit_res['fitted_MSE']
  params = fit_res['fitted_params']
  scale_factors = fit_res['fitted_scale_factors']
  copy_scores = fit_res['copy_scores']
  head_scores = fit_res['head_scores']
  labels = fit_res['labels']
  sizes = fit_res['sizes']

  # plot results
  plot_fit_over_size(sizes, mses)
  plot_params_over_size(sizes, params, mses)
  plot_scale_factors_over_size(sizes, scale_factors, mses)
  plot_scores_over_size(sizes, head_scores, copy_scores, mses)
  