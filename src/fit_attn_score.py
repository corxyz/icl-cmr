# fit all attention heads of a given model by minimizing CMR distance
# reproduces Figure 5 and Figure 6a

import math, argparse
import numpy as np
import joblib
import matplotlib.pyplot as plt
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

########################################
# Define CLI parser
########################################

parser = argparse.ArgumentParser(description='Fit attention scores with CMR.')
parser.add_argument('-s', '--save_dir', type=str, default='gpt2-small',
                    help='Save directory')
parser.add_argument('-f', '--all_head_scores', type=str, default='gpt2-small/all_scores_gpt2_small.pkl',
                    help='Path to file with the model\'s head scores')

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

def get_crp(idx_beta_enc, idx_beta_rec, idx_gamma_ft, select_range=5, sem=False):
  # get normalized CRP in the specified range and standard errors if specified
  select_idx = (crp_range >= -select_range) & (crp_range <= select_range)
  d = crp[idx_beta_enc, idx_beta_rec, idx_gamma_ft, select_idx]
  if sem:
    err = crp_sem[idx_beta_enc, idx_beta_rec, idx_gamma_ft, select_idx]
    return d / d.sum(), err / d.sum()
  else:
    return d / d.sum()

def scale_crp_to_attn(cmr_crp, attn_scores):
  # scale CRP to the same range as attn_scores (attention scores)
  # assuming the minimum value of attn_scores is zero
  assert(attn_scores.min() == 0)
  cmr_crp_scaled = cmr_crp - cmr_crp.min()
  scale_factor = attn_scores.max() / cmr_crp_scaled.max()
  cmr_crp_scaled = cmr_crp_scaled * scale_factor
  return cmr_crp_scaled, scale_factor

def get_head_mse(attn_scores, idx_beta_enc, idx_beta_rec, idx_gamma_ft):
  # compute the MSE between an attention score pattern and a CRP from specified CMR params
  # the two patterns are scaled to the same range for fitting
  cmr_crp = get_crp(idx_beta_enc, idx_beta_rec, idx_gamma_ft)
  assert(attn_scores.size == cmr_crp.size)

  attn_scores_scaled = attn_scores - attn_scores.min()
  cmr_crp_scaled, scale_factor = scale_crp_to_attn(cmr_crp, attn_scores_scaled)
  weight = np.ones_like(attn_scores_scaled)
  result = {
    'pattern': cmr_crp,
    'pattern_scaled': cmr_crp_scaled, 
    'attn_scaled': attn_scores_scaled,
    'scale_factor': scale_factor,
    'MSE': np.mean((cmr_crp_scaled - attn_scores_scaled) ** 2 * weight / np.var(attn_scores_scaled)),
  }
  return result

def fit_head_cmr(attn_scores):
  # compute the CMR distance between the average attention scores and all CRPs
  param_dim = crp.shape[:-1]
  mse = np.zeros(param_dim)
  patterns = np.zeros(tuple(list(param_dim) + [attn_scores.size]))
  pattern_scaled = np.zeros(tuple(list(param_dim) + [attn_scores.size]))
  attn_scaled = np.zeros(tuple(list(param_dim) + [attn_scores.size]))
  scale_factors = np.zeros(param_dim)
  for i in range(crp.shape[0]):
    for j in range(crp.shape[1]):
      for k in range(crp.shape[2]):
        res = get_head_mse(attn_scores, i, j, k)
        mse[i,j,k], patterns[i,j,k] = res['MSE'], res['pattern']
        pattern_scaled[i,j,k] = res['pattern_scaled']
        attn_scaled[i,j,k] = res['attn_scaled']
        scale_factors[i,j,k] = res['scale_factor']
  
  # find the best fit
  fitted_beta_enc, fitted_beta_rec, fitted_gamma_ft = np.unravel_index(np.argmin(mse), mse.shape)
  fitted_params = crp_params[:, fitted_beta_enc, fitted_beta_rec, fitted_gamma_ft]
  fitted_scale_factor = scale_factors[fitted_beta_enc, fitted_beta_rec, fitted_gamma_ft]
  head_fit = {
    'MSE': mse[fitted_beta_enc, fitted_beta_rec, fitted_gamma_ft],
    'params': fitted_params,
    'scale_factors': fitted_scale_factor,
    'patterns': patterns[fitted_beta_enc, fitted_beta_rec, fitted_gamma_ft],
    'pattern_scaled': pattern_scaled[fitted_beta_enc, fitted_beta_rec, fitted_gamma_ft],
    'attn_scaled': attn_scaled[fitted_beta_enc, fitted_beta_rec, fitted_gamma_ft],
  }
  return head_fit

def fit(attn_scores):
  # fit all attention heads
  n_layers, n_heads, n_score = attn_scores.shape
  fitted_mse = np.zeros((n_layers, n_heads))
  fitted_params = np.zeros((n_layers, n_heads, len(crp.shape)-1))
  fitted_patterns = np.zeros((n_layers, n_heads, n_score))
  fitted_pattern_scaled = np.zeros((n_layers, n_heads, n_score))
  attn_scaled = np.zeros((n_layers, n_heads, n_score))
  fitted_scale_factors = np.zeros((n_layers, n_heads))

  for l in range(n_layers):
    for h in range(n_heads):
      attn_pattern = attn_scores[l,h]
      head_fit = fit_head_cmr(attn_pattern)
      fitted_mse[l,h] = head_fit['MSE']
      fitted_params[l,h] = head_fit['params']
      fitted_patterns[l,h] = head_fit['patterns']
      fitted_pattern_scaled[l,h] = head_fit['pattern_scaled']
      attn_scaled[l,h] = head_fit['attn_scaled']
      fitted_scale_factors[l,h] = head_fit['scale_factors']
  
  fit_res = {
    'fitted_MSE': fitted_mse,
    'fitted_params': fitted_params,
    'fitted_patterns': fitted_patterns,
    'fitted_pattern_scaled': fitted_pattern_scaled,
    'attn_scaled': attn_scaled,
    'fitted_scale_factors': fitted_scale_factors,
  }
  return fit_res

########################################
# Plotting functions
########################################

def plot_fit(fit_res, head_scores, copy_scores, attn_scores, attn_scores_sem, labels, n_row=3, save_dir='.'):
  # plot induction-head matching scores and copying scores of each head (one figure per layer)
  n_layers, n_heads = fit_res['fitted_MSE'].shape
  for l in range(n_layers):
    plt.figure(figsize=[12*cm, 12*cm], dpi=300)
    plt.plot(head_scores[l], label='original', marker='o')
    plt.xticks(np.arange(n_heads), labels[l], rotation=90)
    plt.ylabel('Induction head matching score', fontsize=7)
    plt.legend()
    ax = plt.gca()
    ax.set_ylim([0,1])
    ax2 = ax.twinx()
    ax2.plot(copy_scores[l], color='red', marker='x')
    ax2.plot([0] * n_heads, linestyle='--', color="red")
    ax2.set_ylabel('Copying score', color='red', fontsize=7)
    ax2.set_ylim([-1, 1])
    plt.tight_layout()
    plt.savefig('./figs/{}/head_copy_score_L{}.pdf'.format(save_dir, l), dpi=300)

    # plot the fitted CRP curve over each attention pattern
    plt.figure(figsize=[16*cm, 12*cm], dpi=300)
    x = np.arange(-select_range, select_range+1)
    n_col = math.ceil(n_heads/n_row)
    for h in range(n_heads):
      plt.subplot(n_row, n_col, h+1)
      attn_patt = attn_scores[l,h]
      plt.plot(x, attn_patt, marker='o', markersize=1, 
               color='k', label='Head')
      plt.errorbar(x, attn_patt, yerr=attn_scores_sem[l,h],
                   color='k', alpha=0.6)
      plt.vlines(0, attn_patt.min(), attn_patt.max(),
                 color='k', linestyles='--')
      ax = plt.gca()
      ax.tick_params(axis='both', labelsize=5)
      ax2 = ax.twinx()
      ax2.plot(x, fit_res['fitted_patterns'][l,h], marker='x', markersize=1,
               color='r', alpha=0.3, label='CMR fit')
      ax2.tick_params(axis='both', labelsize=5)
      fitted_beta_enc, fitted_beta_rec, fitted_gamma_ft = fit_res['fitted_params'][l,h]
      plt.title('CMR distance = {:.2}\n' r'$\beta_{{\rm enc}}={:.2},\beta_{{\rm rec}}={:.2},\gamma_{{\rm FT}}={:.2}$'.format(fit_res['fitted_MSE'][l,h], 
                                                                                                              fitted_beta_enc, fitted_beta_rec, fitted_gamma_ft), 
                fontsize=5)
      if h >= n_col * (n_row - 1):
        ax.set_xlabel('Lag', fontsize=5)
      if h % n_col == 0:
        ax.set_ylabel('Attention scores', fontsize=5)
      elif (h + 1) % n_col == 0:
        ax2.set_ylabel('CRP', color='red', fontsize=5)
    plt.tight_layout(pad=0.3)
    plt.legend(fontsize=5)
    plt.savefig('./figs/{}/head_fit_L{}.pdf'.format(save_dir, l), dpi=300)

def plot_score_against_fit(copy_scores, head_scores, fit_res, n_row=2, n_col=1, save_dir='.'):
  # plot (1) CRP scores against fitted MSEs, and (2) histogram of fitted MSEs
  mse = fit_res['fitted_MSE'].flatten()
  head_scores = head_scores.flatten()
  colors = copy_scores.flatten() > 0
  colors = np.where(colors, 'b', 'r')
  first_blue_idx = np.where(colors == 'b')[0][0]
  first_red_idx = np.where(colors == 'r')[0][0]
  _, axes = plt.subplots(n_row, n_col, figsize=[6*cm, 8*cm], 
                           sharex=True, tight_layout=True, dpi=300)
  axes[0].scatter(mse, head_scores, c=colors, alpha=0.3, s=3)
  # label one blue and one red point
  axes[0].scatter([mse[first_blue_idx]], [head_scores[first_blue_idx]], c='b', alpha=0.3, s=5, label='Copying score > 0')
  axes[0].scatter([mse[first_red_idx]], [head_scores[first_red_idx]], c='r', alpha=0.3, s=5, label='Copying score < 0')
  axes[0].legend(fontsize=5)
  axes[0].set_ylabel('Induction head\nmatching score', fontsize=5)
  axes[0].tick_params(axis='both', labelsize=5)
  axes[1].hist(mse, bins=25)
  axes[1].set_xlabel('CMR distance (MSE)', fontsize=5)
  axes[1].set_ylabel('Count', fontsize=5)
  axes[1].tick_params(axis='both', labelsize=5)
  plt.savefig('./figs/{}/score_against_fit.pdf'.format(save_dir), dpi=300)

def plot_score_against_scale_factors(copy_scores, head_scores, fit_res, n_row=2, n_col=1, save_dir='.'):
  scale_factors = fit_res['fitted_scale_factors'].flatten()
  head_scores = head_scores.flatten()
  colors = copy_scores.flatten() > 0
  colors = np.where(colors, 'b', 'r')
  first_blue_idx = np.where(colors == 'b')[0][0]
  first_red_idx = np.where(colors == 'r')[0][0]
  _, axes = plt.subplots(n_row, n_col, figsize=[12*cm, 12*cm], 
                           sharex=True, tight_layout=True, dpi=300)
  axes[0].scatter(scale_factors, head_scores, c=colors, alpha=0.3, s=5)
  # label one blue and one red point
  axes[0].scatter([scale_factors[first_blue_idx]], [head_scores[first_blue_idx]], c='b', alpha=0.3, s=5, label='Copying score > 0')
  axes[0].scatter([scale_factors[first_red_idx]], [head_scores[first_red_idx]], c='r', alpha=0.3, s=5, label='Copying score < 0')
  axes[0].legend(fontsize=7)
  axes[0].set_ylabel('Induction head matching score', fontsize=7)
  axes[1].hist(scale_factors, bins=25)
  axes[1].set_xlabel(r'$1/\tau$', fontsize=7)
  axes[1].set_ylabel('Count', fontsize=7)
  plt.savefig('./figs/{}/score_against_scale_factor.pdf'.format(save_dir), dpi=300)

def plot_fit_against_layer(mse, save_dir='.'):
  n_layers, _ = mse.shape

  # plot overall trend
  plt.figure(figsize=[4*cm, 4*cm], dpi=300)
  ax = plt.gca()
  ax.tick_params(axis='both', labelsize=5)
  x = np.arange(n_layers) / n_layers * 100
  xlabel = 'Relative layer position (%)'

  mask = [~np.isnan(mse[l]) for l in range(n_layers)]
  low_mse_pct = [(mse[l,mask[l]] < mse_thres).mean()*100 for l in range(n_layers)]
  plt.plot(x, low_mse_pct, 'k-', lw=1)
  plt.xlim([0,100])
  plt.xlabel(xlabel, fontsize=5)
  plt.ylabel('% CMR distance < {}'.format(mse_thres), fontsize=5)
  plt.tight_layout()
  plt.savefig('./figs/{}/fit_against_layer_thres{}.pdf'.format(save_dir, mse_thres))


if __name__ == '__main__':
  # parse command line arguments
  args = parser.parse_args()
  save_dir = args.save_dir
  all_head_scores = joblib.load('./saved_scores/' + args.all_head_scores)

  # load and fit all average attention scores for each head
  head_scores = all_head_scores['sorted_head_scores']
  copy_scores = all_head_scores['sorted_copying_score']
  labels = all_head_scores['sorted_labels']
  attn_scores = load_scores_in_range(all_head_scores, 'sorted_CRP_scores', select_range=select_range)
  attn_scores_sem = load_scores_in_range(all_head_scores, 'sorted_CRP_scores_sem', select_range=select_range)
  fit_res = fit(attn_scores)

  # plot results
  plot_fit_against_layer(fit_res['fitted_MSE'], save_dir=save_dir)
  plot_fit(fit_res, head_scores, copy_scores, attn_scores, attn_scores_sem, labels, select_range=select_range, save_dir=save_dir)
  plot_score_against_fit(copy_scores, head_scores, fit_res, save_dir=save_dir)
  plot_score_against_scale_factors(copy_scores, head_scores, fit_res, save_dir=save_dir)
