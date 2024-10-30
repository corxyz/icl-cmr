# find and plot top attention heads
# reproduces Figure 7c-f, Supplementary Figure S3

import argparse
import numpy as np
import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import sem
from fit_attn_score import fit, fit_head_cmr
from select_fit_xtra import select_subset_top, plot_selected_fit, plot_selected_params, plot_selected_scale_factors
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

file_prefix = {
  'pythia-70m-deduped-v0': 'all_scores_pythia_70m_cp',
  'pythia-160m-deduped-v0': 'all_scores_pythia_160m_cp',
  'pythia-410m-deduped-v0': 'all_scores_pythia_410m_cp',
  'pythia-1b-deduped-v0': 'all_scores_pythia_1b_cp',
  'pythia-1.4b-deduped-v0': 'all_scores_pythia_1.4b_cp',
  'pythia-2.8b-deduped-v0': 'all_scores_pythia_2.8b_cp',
  'pythia-6.9b-deduped-v0': 'all_scores_pythia_6.9b_cp',
  'pythia-12b-deduped-v0': 'all_scores_pythia_12b_cp',
}

########################################
# Define CLI parser
########################################

parser = argparse.ArgumentParser(description='Fit attention scores with CMR (top attention heads).')
parser.add_argument('-m', '--models', type=str, nargs='+', 
                    default=['pythia-70m-deduped-v0', 'pythia-160m-deduped-v0', 'pythia-410m-deduped-v0',
                             'pythia-1b-deduped-v0', 'pythia-1.4b-deduped-v0', 'pythia-2.8b-deduped-v0',
                             'pythia-6.9b-deduped-v0', 'pythia-12b-deduped-v0'],
                    help='Model(s)')
parser.add_argument('-t', '--n_top', type=int, nargs='+', 
                    default=[20, 50, 100, 200],
                    help='Number of top heads to analyze')
parser.add_argument('-u', '--use_saved', type=bool, default=False, 
                    help='Use saved selection of top heads')

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

def fit_selected_head(selection):
  # fit a selection of attention heads
  # the selection should be a dictionary and have the format {model_name: [label(s)]}
  cps = list(range(0, n_cp, cp_interv)) + [n_cp - 1]
  n_cps = len(cps)    # number of checkpoints to fit
  n_heads = sum([len(selection[model_name]) for model_name in selection]) # number of selected heads
  mses = np.zeros((n_heads, n_cps))
  params = np.zeros((n_heads, n_cps, len(crp.shape)-1))
  scale_factors = np.zeros((n_heads, n_cps))
  copy_scores, head_scores = np.zeros((n_heads, n_cps)), np.zeros((n_heads, n_cps))
  labels = []

  for i, cp in enumerate(cps):
    selected_attn_scores = np.zeros((0, select_range * 2 + 1))
    selected_copy_scores = np.empty(0)
    selected_head_scores = np.empty(0)
    for model_name in selection:
      all_head_scores = joblib.load('./saved_scores/{}/{}{}.pkl'.format(model_name, file_prefix[model_name], cp))
      attn_scores = load_scores_in_range(all_head_scores, 'CRP_scores', select_range=select_range)
      for head_label in selection[model_name]:
        head_index = np.where(all_head_scores['labels'] == head_label)
        selected_attn_scores = np.append(selected_attn_scores, attn_scores[head_index], axis=0)
        selected_copy_scores = np.append(selected_copy_scores, all_head_scores['copying_score'][head_index])
        selected_head_scores = np.append(selected_head_scores, all_head_scores['head_scores'][head_index])
        if i == 0:  # record head label once
          labels.append(model_name.split('-')[1] + ':' + head_label)
    fit_res = fit(np.expand_dims(selected_attn_scores, axis=0))
    # record MSE, fitted parameters (beta_enc, bet_rec), inverse temperature (scale factor)
    mses[:,i] = fit_res['fitted_MSE']
    params[:,i] = fit_res['fitted_params']
    scale_factors[:,i] = fit_res['fitted_scale_factors']
    # record head and copying scores
    copy_scores[:,i] = selected_copy_scores
    head_scores[:,i] = selected_head_scores

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
# Top head selection
########################################

def select_top_heads(models, n, cp=142):
  # select n attention heads with the lowest CMR distance across specified models
  all_mses, all_inv_temps = [], []
  all_beta_enc, all_beta_rec, all_gamma_ft = [], [], []
  all_labels = []
  selection = dict()
  for model_name in models:
    all_head_scores = joblib.load('./saved_scores/{}/{}{}.pkl'.format(model_name, file_prefix[model_name], cp))
    attn_scores = load_scores_in_range(all_head_scores, 'CRP_scores', select_range=select_range)
    labels = all_head_scores['labels'].flatten()
    prefix = np.array([model_name + '-'] * labels.size)
    fit_res = fit(attn_scores)
    all_mses.append(fit_res['fitted_MSE'].flatten())
    all_beta_enc.append(fit_res['fitted_params'][:,:,0].flatten())
    all_beta_rec.append(fit_res['fitted_params'][:,:,1].flatten())
    all_gamma_ft.append(fit_res['fitted_params'][:,:,2].flatten())
    all_inv_temps.append(fit_res['fitted_scale_factors'].flatten())
    all_labels.append(np.char.add(prefix, labels))
  all_mses = np.concatenate(all_mses)
  all_params = np.column_stack([np.concatenate(all_beta_enc), np.concatenate(all_beta_rec), np.concatenate(all_gamma_ft)])
  all_inv_temps = np.concatenate(all_inv_temps)
  all_labels = np.concatenate(all_labels)

  top_heads = np.argpartition(all_mses, n)[:n]        # top n heads (!NOT GUARANTEED TO BE IN ANY PARTICULAR ORDER)
  for label in all_labels[top_heads]:
    model_name = '-'.join(label.split('-')[:-1])
    if model_name not in selection:
      selection[model_name] = []
    selection[model_name].append(label.split('-')[-1])
  res = {
    'MSE': all_mses[top_heads],
    'params': all_params[top_heads],
    'scale_factors': all_inv_temps[top_heads],
    'labels': all_labels[top_heads],
    'selection': selection,
    'selection_name': 'top {}'.format(n),
  }
  return res

def select_top_induction_heads(models, n, cp=142):
  # select n attention heads with the highest induction-head matching score across specified models
  head_scores = []
  copy_scores = []
  all_labels = []
  selection = dict()
  for model_name in models:
    all_head_scores = joblib.load('./saved_scores/{}/{}{}.pkl'.format(model_name, file_prefix[model_name], cp))
    head_scores.append(all_head_scores['head_scores'].flatten())
    copy_scores.append(all_head_scores['copying_score'].flatten()) 
    labels = all_head_scores['labels'].flatten()
    prefix = np.array([model_name + '-'] * labels.size)
    all_labels.append(np.char.add(prefix, labels))
  head_scores = np.concatenate(head_scores)
  copy_scores = np.concatenate(copy_scores)
  all_labels = np.concatenate(all_labels)

  top_heads = np.argpartition(head_scores, -n)[-n:]
  for label in all_labels[top_heads]:
    model_name = '-'.join(label.split('-')[:-1])
    if model_name not in selection:
      selection[model_name] = []
    selection[model_name].append(label.split('-')[-1])
  
  attn_scores = load_scores_in_range(all_head_scores, 'CRP_scores', select_range=select_range)
  attn_scores = attn_scores.reshape((-1,attn_scores.shape[-1]))
  fitted_mse, fitted_params, fitted_scale_factors = [], [], []
  for i in top_heads:
    head_fit = fit_head_cmr(attn_scores[i])
    fitted_mse.append(head_fit['MSE'])
    fitted_params.append(head_fit['params'])
    fitted_scale_factors.append(head_fit['scale_factors'])

  res = {
    'MSE': np.array(fitted_mse),
    'params': np.array(fitted_params),
    'scale_factors': np.array(fitted_scale_factors),
    'head_scores': head_scores[top_heads],
    'copy_scores': copy_scores[top_heads],
    'labels': all_labels[top_heads],
    'selection': selection,
    'selection_name': 'top {}'.format(n),
  }
  return res

########################################
# Plotting functions
########################################

def plot_selected_fit_over_time(time, mses, labels, save_name_suffix=''):
  # plot the average CMR distance of the top heads in a model 
  # as a function of training checkpoints
  plt.figure(figsize=[6*cm, 3*cm], dpi=300)
  ax = plt.gca()
  ax.tick_params(axis='both', labelsize=5)
  for i, mse in enumerate(mses):
    plt.plot(time, mse, lw=0.4, label=labels[i], color=mpl.cm.gnuplot(i / mses.shape[0]))
  plt.xlabel('Checkpoint', fontsize=5)
  plt.ylabel('CMR distance', fontsize=5)
  plt.legend(bbox_to_anchor=(1.01, 0), loc='lower left', fontsize=5)
  plt.tight_layout()
  plt.savefig('./figs/selected/fit_over_time_{}.pdf'.format(save_name_suffix), dpi=300)

def plot_selected_params_over_time(time, params, labels, save_name_suffix=''):
  # plot the average beta_enc, beta_dec, and gamma_FT of the top heads in a model
  # as a function of training checkpoints
  fig, axes = plt.subplots(1, 3, figsize=[14*cm, 3*cm], dpi=300)
  for ax in axes: 
    ax.tick_params(axis='both', labelsize=5)
  for i, p in enumerate(params):
    beta_enc, beta_dec, gamma_ft = p[:,0], p[:,1], p[:,2]
    axes[0].plot(time, beta_enc, lw=0.4, label=labels[i], color=mpl.cm.gnuplot(i / params.shape[0]))
    axes[0].set_xlabel('Checkpoint', fontsize=5)
    axes[0].set_ylabel(r'$\beta_{\rm enc}$', fontsize=5)
    axes[0].set_ylim([0.5,1])
    axes[1].plot(time, beta_dec, lw=0.4, label=labels[i], color=mpl.cm.gnuplot(i / params.shape[0]))
    axes[1].set_xlabel('Checkpoint', fontsize=5)
    axes[1].set_ylabel(r'$\beta_{\rm rec}$', fontsize=5)
    axes[1].set_ylim([0.5,1])
    axes[2].plot(time, gamma_ft, lw=0.4, label=labels[i], color=mpl.cm.gnuplot(i / params.shape[0]))
    axes[2].set_xlabel('Checkpoint', fontsize=5)
    axes[2].set_ylabel(r'$\gamma_{\rm FT}$', fontsize=5)
    axes[2].set_ylim([0,0.5])
    axes[2].legend(bbox_to_anchor=(1.01, 0), loc='lower left', fontsize=5)
  fig.tight_layout()
  plt.savefig('./figs/selected/params_over_time_{}.pdf'.format(save_name_suffix), dpi=300)

def plot_selected_scale_factors_over_time(time, scale_factors, labels, save_name_suffix=''):
  # plot the scale factor (tau^-1) of the top heads in a model
  # as a functiion of training checkpoints
  plt.figure(figsize=[5*cm, 3*cm], dpi=300)
  ax = plt.gca()
  ax.tick_params(axis='both', labelsize=5)
  for i, inv_tau in enumerate(scale_factors):
    plt.plot(time, inv_tau, lw=0.4, label=labels[i], color=mpl.cm.gnuplot(i / scale_factors.shape[0]))
  plt.xlabel('Checkpoint', fontsize=5)
  plt.ylabel(r'$\tau^{-1}$', fontsize=5)
  plt.legend(bbox_to_anchor=(1.01, 0), loc='lower left', fontsize=5)
  plt.tight_layout()
  plt.savefig('./figs/selected/scale_factor_over_time_{}.pdf'.format(save_name_suffix), dpi=300)

def plot_selected_scores_over_time(time, head_scores, copy_scores, labels, save_name_suffix=''):
  # plot the head and copy scores of the top heads in a model
  # as a function of training checkpoints
  _, axes = plt.subplots(1, 2, figsize=[10*cm, 3*cm], dpi=300)
  for ax in axes: 
    ax.tick_params(axis='both', labelsize=5)
  for i, hscore in enumerate(head_scores):
    cscore = copy_scores[i]
    axes[0].plot(time, hscore, lw=0.4, label=labels[i], color=mpl.cm.gnuplot(i / head_scores.shape[0]))
    axes[0].set_xlabel('Checkpoint', fontsize=5)
    axes[0].set_ylabel('Head score', fontsize=5)
    axes[1].plot(time, cscore, lw=0.4, label=labels[i], color=mpl.cm.gnuplot(i / copy_scores.shape[0]))
    axes[1].set_xlabel('Checkpoint', fontsize=5)
    axes[1].set_ylabel('Copying score', fontsize=5)
    axes[1].legend(bbox_to_anchor=(1.01, 0), loc='lower left', fontsize=5)
  plt.tight_layout()
  plt.savefig('./figs/selected/scores_over_time_{}.pdf'.format(save_name_suffix), dpi=300)

def plot_selected_heads(selection, save_name_suffix=''):
  fit_res = fit_selected_head(selection)
  plot_selected_fit_over_time(fit_res['checkpoints'], fit_res['fitted_MSE'], fit_res['labels'], 
                              save_name_suffix=save_name_suffix)
  plot_selected_params_over_time(fit_res['checkpoints'], fit_res['fitted_params'], fit_res['labels'], 
                                 save_name_suffix=save_name_suffix)
  plot_selected_scale_factors_over_time(fit_res['checkpoints'], fit_res['fitted_scale_factors'], fit_res['labels'], 
                                        save_name_suffix=save_name_suffix)
  plot_selected_scores_over_time(fit_res['checkpoints'], fit_res['head_scores'], fit_res['copy_scores'], fit_res['labels'], 
                                 save_name_suffix=save_name_suffix)

def plot_selection_cmp(selections, save_name_suffix=''):
  # compare the top n heads of a model given different values of n
  checkpoints = None
  mses, params, inv_temps = [], [], []
  head_scores, copy_scores = [], []
  labels = []
  for selection in selections:
    fit_res = fit_selected_head(selection['selection'])
    if checkpoints is None:
      checkpoints = fit_res['checkpoints']
    mses.append(fit_res['fitted_MSE'].mean(axis=0))
    params.append(fit_res['fitted_params'].mean(axis=0))
    inv_temps.append(fit_res['fitted_scale_factors'].mean(axis=0))
    head_scores.append(np.nanmean(fit_res['head_scores'], axis=0))
    copy_scores.append(np.nanmean(fit_res['copy_scores'],axis=0))
    labels.append(selection['selection_name'])
  plot_selected_fit_over_time(checkpoints, np.row_stack(mses), labels, save_name_suffix=save_name_suffix)
  plot_selected_params_over_time(checkpoints, np.stack(params, axis=0), labels, save_name_suffix=save_name_suffix)
  plot_selected_scale_factors_over_time(checkpoints, np.row_stack(inv_temps), labels, save_name_suffix=save_name_suffix)
  plot_selected_scores_over_time(checkpoints, np.row_stack(head_scores), np.row_stack(copy_scores), labels, save_name_suffix=save_name_suffix)

def plot_saved_selection_cmp(selections, save_name_suffix=''):
  # compare the top n heads across different Pythia models given different values of n
  # NOTE: current implementation only draws an odd number of models nicely
  mses, params, inv_temps = [], [], []
  mses_sem, params_sem, inv_temps_sem = [], [], []
  labels = []  
  for selection in selections:
    mses.append(selection['MSE'].mean())
    mses_sem.append(sem(selection['MSE']))
    params.append(selection['params'].mean(axis=0))
    params_sem.append(sem(selection['params'], axis=0))
    inv_temps.append(selection['scale_factors'].mean())
    inv_temps_sem.append(sem(selection['scale_factors']))
    labels.append(selection['selection_name'])
  plot_selected_fit(np.row_stack(mses), np.row_stack(mses_sem), labels, save_name_suffix=save_name_suffix)
  plot_selected_params(np.stack(params, axis=0), np.stack(params_sem, axis=0), labels, save_name_suffix=save_name_suffix)
  plot_selected_scale_factors(np.row_stack(inv_temps), np.row_stack(inv_temps_sem), labels, save_name_suffix=save_name_suffix)

def plot_saved_selections():
  # plot top CMR-like heads from saved records
  top_crp_fit = joblib.load('saved_top_heads/cmr_top_crp_fit_pythia.pkl')
  top20 = top_crp_fit['top20']
  top50 = top_crp_fit['top50']
  top100 = top_crp_fit['top100']
  top200 = top_crp_fit['top200']
  plot_saved_selection_cmp([top20, top50, top100, top200], save_name_suffix='fit')

  # plot top induction heads from saved records
  top_head_score = joblib.load('saved_top_heads/cmr_top_head_score_pythia.pkl')
  top20 = top_head_score['top20']
  top50 = top_head_score['top50']
  top100 = top_head_score['top100']
  top200 = top_head_score['top200']
  plot_saved_selection_cmp([top20, top50, top100, top200], save_name_suffix='hs')

if __name__ == '__main__':
  # parse command line arguments
  args = parser.parse_args()
  models = args.models
  n_top = sorted(args.n_top, reverse=True)  # sort top selection sizes from the largest to the smallest
  use_saved = args.use_saved

  if use_saved:
    plot_saved_selections()
  else:
    # compute top heads from scratch
    # first find and plot top CMR-like heads over training
    all_selection_cmr = []
    for i, n in enumerate(n_top):
      print('Selecting top {} CMR-like heads...'.format(n))
      if i == 0:  # only fit the largest subset
        selected = select_top_heads(models, n)
        all_selection_cmr.append(selected)
      else:       # all other selections are subsets of the largest subset
        assert(len(all_selection_cmr) > 0)
        selected = select_subset_top(all_selection_cmr[0], n)
        all_selection_cmr.append(selected)
    print('Done.')
    plot_selection_cmp(all_selection_cmr, save_name_suffix='cmr')
    
    # save top CMR-like heads
    d = dict()
    for i, n in enumerate(n_top):
      d['top' + str(n)] = all_selection_cmr[i]
    joblib.dump(d, 'saved_top_heads/cmr_top_crp_fit_pythia.pkl')
    print('Selections saved.')

    # then find and plot top induction head over training
    all_selection_hs = []
    for i, n in enumerate(n_top):
      print('Selecting top {} induction heads...'.format(n))
      if i == 0:  # only fit the largest subset
        selected = select_top_induction_heads(models, n)
        all_selection_hs.append(selected)
      else:       # all other selections are subsets of the largest subset
        assert(len(all_selection_hs) > 0)
        selected = select_subset_top(all_selection_hs[0], n)
        all_selection_hs.append(selected)
    print('Done.')
    plot_selection_cmp(all_selection_hs, save_name_suffix='hs')
    
    # save top induction heads
    d = dict()
    for i, n in enumerate(n_top):
      d['top' + str(n)] = all_selection_hs[i]
    joblib.dump(d, 'saved_top_heads/cmr_top_head_score_pythia.pkl')
    print('Selections saved.')
