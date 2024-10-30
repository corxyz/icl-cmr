# find and plot top attention heads (extra models)

import argparse
import numpy as np
from scipy.stats import sem
import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
from fit_attn_score import fit, fit_head_cmr
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

########################################
# Define CLI parser
########################################

parser = argparse.ArgumentParser(description='Fit attention scores with CMR (top attention heads).')
parser.add_argument('-m', '--models', type=str, nargs='+', 
                    default=['qwen-7b', 'mistral-7b', 'llama3-8b'],
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
  n_heads = sum([len(selection[model_name]) for model_name in selection]) # number of selected heads
  mses = np.zeros(n_heads)
  params = np.zeros((n_heads, len(crp.shape)-1))
  scale_factors = np.zeros(n_heads)
  copy_scores, head_scores = np.zeros(n_heads), np.zeros(n_heads)
  labels = []

  selected_attn_scores = np.zeros((0, select_range * 2 + 1))
  selected_copy_scores = np.empty(0)
  selected_head_scores = np.empty(0)
  for model_name in selection:
    all_head_scores = joblib.load('./saved_scores/{}/induction_head_all_scores_{}.pkl'.format(model_name, model_name))
    attn_scores = load_scores_in_range(all_head_scores, 'CRP_scores', select_range=select_range)
    for head_label in selection[model_name]:
      head_index = np.where(all_head_scores['labels'] == head_label)
      selected_attn_scores = np.append(selected_attn_scores, attn_scores[head_index], axis=0)
      selected_copy_scores = np.append(selected_copy_scores, all_head_scores['copying_score'][head_index])
      selected_head_scores = np.append(selected_head_scores, all_head_scores['head_scores'][head_index])
      labels.append(model_name.split('-')[1] + ':' + head_label)
  fit_res = fit(np.expand_dims(selected_attn_scores, axis=0))
  # record MSE, fitted parameters (beta_enc, bet_rec), inverse temperature (scale factor)
  mses = fit_res['fitted_MSE']
  params = fit_res['fitted_params']
  scale_factors = fit_res['fitted_scale_factors']
  # record head and copying scores
  copy_scores = selected_copy_scores
  head_scores = selected_head_scores

  fit_res = {
    'fitted_MSE': mses,
    'fitted_params': params,
    'fitted_scale_factors': scale_factors,
    'copy_scores': copy_scores,
    'head_scores': head_scores,
    'labels': labels,
  }
  return fit_res

def select_top_heads(models, n):
  # select n attention heads with the lowest CMR distance across specified models
  all_mses, all_inv_temps = [], []
  all_beta_enc, all_beta_rec, all_gamma_ft = [], [], []
  head_scores, copy_scores = [], []
  all_labels = []
  selection = dict()
  for model_name in models:
    if model_name == 'gpt2-small':
      all_head_scores = joblib.load('./saved_scores/{}/all_scores_gpt2_small.pkl')
    else:
      all_head_scores = joblib.load('./saved_scores/{}/induction_head_all_scores_{}.pkl'.format(model_name, model_name))
    attn_scores = load_scores_in_range(all_head_scores, 'CRP_scores', select_range=select_range)
    labels = all_head_scores['labels'].flatten()
    prefix = np.array([model_name + '-'] * labels.size)
    fit_res = fit(attn_scores)
    all_mses.append(fit_res['fitted_MSE'].flatten())
    all_beta_enc.append(fit_res['fitted_params'][:,:,0].flatten())
    all_beta_rec.append(fit_res['fitted_params'][:,:,1].flatten())
    all_gamma_ft.append(fit_res['fitted_params'][:,:,2].flatten())
    all_inv_temps.append(fit_res['fitted_scale_factors'].flatten())
    head_scores.append(all_head_scores['head_scores'].flatten())
    copy_scores.append(all_head_scores['copying_score'].flatten()) 
    all_labels.append(np.char.add(prefix, labels))
  all_mses = np.concatenate(all_mses)
  all_params = np.column_stack([np.concatenate(all_beta_enc), np.concatenate(all_beta_rec), np.concatenate(all_gamma_ft)])
  all_inv_temps = np.concatenate(all_inv_temps)
  head_scores = np.concatenate(head_scores)
  copy_scores = np.concatenate(copy_scores)
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
    'head_score': head_scores[top_heads],
    'copy_score': copy_scores[top_heads],
    'labels': all_labels[top_heads],
    'selection': selection,
    'selection_name': 'top {}'.format(n),
  }
  return res

def select_top_induction_heads(models, n):
  # select n attention heads with the highest induction-head matching score across specified models
  head_scores = []
  copy_scores = []
  all_labels = []
  selection = dict()
  for model_name in models:
    if model_name == 'gpt2-small':
      all_head_scores = joblib.load('./saved_scores/{}/all_scores_gpt2_small.pkl')
    else:
      all_head_scores = joblib.load('./saved_scores/{}/induction_head_all_scores_{}.pkl'.format(model_name, model_name))
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
  
def plot_selected_fit(mses, mses_sem, labels, save_name_suffix=''):
  # plot the average CMR distance of the top heads in a model
  plt.figure(figsize=[6*cm, 3*cm], dpi=300)
  ax = plt.gca()
  ax.tick_params(axis='both', labelsize=5)
  plt.bar(np.arange(len(labels)), mses.flatten(), yerr=mses_sem.flatten(),
          color=[mpl.cm.gnuplot(i / mses.size) for i in range(mses.size)],
          ecolor='blue', capsize=1, error_kw={'elinewidth': 1})
  plt.xticks(np.arange(len(labels)), labels)
  plt.ylabel('CMR distance', fontsize=5)
  plt.tight_layout()
  plt.savefig('./figs/selected/fit_xtra_{}.pdf'.format(save_name_suffix), dpi=300)

def plot_selected_params(params, params_sem, labels, save_name_suffix=''):
  # plot the average beta_enc, beta_dec, and gamma_FT of the top heads in a model
  fig, axes = plt.subplots(1, 3, figsize=[14*cm, 3*cm], dpi=300)
  for ax in axes: 
    ax.tick_params(axis='both', labelsize=5)
  beta_enc, beta_dec, gamma_ft = params[:,0], params[:,1], params[:,2]
  beta_enc_sem, beta_dec_sem, gamma_ft_sem = params_sem[:,0], params_sem[:,1], params_sem[:,2]
  axes[0].bar(np.arange(len(labels)), beta_enc.flatten(), yerr=beta_enc_sem.flatten(),
              color=[mpl.cm.gnuplot(i / beta_enc.size) for i in range(beta_enc.size)],
              ecolor='blue', capsize=1, error_kw={'elinewidth': 1})
  axes[0].set_xticks(np.arange(len(labels)), labels=labels)
  axes[0].set_ylabel(r'$\beta_{\rm enc}$', fontsize=5)
  axes[0].set_ylim([0.5,1])
  axes[1].bar(np.arange(len(labels)), beta_dec.flatten(), yerr=beta_dec_sem.flatten(),
              color=[mpl.cm.gnuplot(i / beta_dec.size) for i in range(beta_dec.size)],
              ecolor='blue', capsize=1, error_kw={'elinewidth': 1})
  axes[1].set_xticks(np.arange(len(labels)), labels=labels)
  axes[1].set_ylabel(r'$\beta_{\rm rec}$', fontsize=5)
  axes[1].set_ylim([0.5,1])
  axes[2].bar(np.arange(len(labels)), gamma_ft.flatten(), yerr=gamma_ft_sem.flatten(),
              color=[mpl.cm.gnuplot(i / gamma_ft.size) for i in range(gamma_ft.size)],
              ecolor='blue', capsize=1, error_kw={'elinewidth': 1})
  axes[2].set_xticks(np.arange(len(labels)), labels=labels)
  axes[2].set_ylabel(r'$\gamma_{\rm FT}$', fontsize=5)
  axes[2].set_ylim([0,0.5])
  fig.tight_layout()
  plt.savefig('./figs/selected/params_xtra_{}.pdf'.format(save_name_suffix), dpi=300)

def plot_selected_scale_factors(scale_factors, scale_factors_sem, labels, save_name_suffix=''):
  # plot the scale factor (tau^-1) of the top heads in a model
  plt.figure(figsize=[5*cm, 3*cm], dpi=300)
  ax = plt.gca()
  ax.tick_params(axis='both', labelsize=5)
  plt.bar(np.arange(len(labels)), scale_factors.flatten(), yerr=scale_factors_sem.flatten(),
          color=[mpl.cm.gnuplot(i / scale_factors.size) for i in range(scale_factors.size)],
          ecolor='blue', capsize=1, error_kw={'elinewidth': 1})
  plt.xticks(np.arange(len(labels)), labels)
  plt.ylabel(r'$\tau^{-1}$', fontsize=5)
  plt.tight_layout()
  plt.savefig('./figs/selected/scale_factor_xtra_{}.pdf'.format(save_name_suffix), dpi=300)

def plot_selected_scores(head_scores, head_scores_sem, copy_scores, copy_scores_sem, labels, save_name_suffix=''):
  # plot the head and copy scores of the top heads in a model
  _, axes = plt.subplots(1, 2, figsize=[10*cm, 3*cm], dpi=300)
  for ax in axes: 
    ax.tick_params(axis='both', labelsize=5)

  axes[0].bar(np.arange(len(labels)), head_scores.flatten(), yerr=head_scores_sem.flatten(),
              color=[mpl.cm.gnuplot(i / head_scores.size) for i in range(head_scores.size)],
              ecolor='blue', capsize=1, error_kw={'elinewidth': 1})
  axes[0].set_xticks(np.arange(len(labels)), labels=labels)
  axes[0].set_ylabel('Head score', fontsize=5)
  axes[1].bar(np.arange(len(labels)), copy_scores.flatten(), yerr=copy_scores_sem.flatten(),
              color=[mpl.cm.gnuplot(i / copy_scores.size) for i in range(copy_scores.size)],
              ecolor='blue', capsize=1, error_kw={'elinewidth': 1})
  axes[1].set_xticks(np.arange(len(labels)), labels=labels)
  axes[1].set_ylabel('Copying score', fontsize=5)
  plt.tight_layout()
  plt.savefig('./figs/selected/scores_xtra_{}.pdf'.format(save_name_suffix), dpi=300)

def plot_selected_heads(selection, save_name_suffix=''):
  fit_res = fit_selected_head(selection, select_range=select_range)
  plot_selected_fit(fit_res['fitted_MSE'], fit_res['labels'], 
                    save_name_suffix=save_name_suffix)
  plot_selected_params(fit_res['fitted_params'], fit_res['labels'], 
                       save_name_suffix=save_name_suffix)
  plot_selected_scale_factors(fit_res['fitted_scale_factors'], fit_res['labels'], 
                              save_name_suffix=save_name_suffix)
  plot_selected_scores(fit_res['head_scores'], fit_res['copy_scores'], fit_res['labels'], 
                       save_name_suffix=save_name_suffix)

def plot_selection_cmp(selections, save_name_suffix=''):
  # compare the top n heads of a model given different values of n
  mses, params, inv_temps = [], [], []
  mses_sem, params_sem, inv_temps_sem = [], [], []
  head_scores, copy_scores = [], []
  head_scores_sem, copy_scores_sem = [], []
  labels = []
  for selection in selections:
    fit_res = fit_selected_head(selection['selection'])
    mses.append(fit_res['fitted_MSE'].mean())
    mses_sem.append(sem(fit_res['fitted_MSE'].squeeze()))
    params.append(fit_res['fitted_params'].squeeze().mean(axis=0))
    params_sem.append(sem(fit_res['fitted_params'].squeeze(), axis=0))
    inv_temps.append(fit_res['fitted_scale_factors'].mean())
    inv_temps_sem.append(sem(fit_res['fitted_scale_factors'].squeeze()))
    head_scores.append(fit_res['head_scores'].mean())
    head_scores_sem.append(sem(fit_res['head_scores']))
    copy_scores.append(fit_res['copy_scores'].mean())
    copy_scores_sem.append(sem(fit_res['copy_scores']))
    labels.append(selection['selection_name'])
  plot_selected_fit(np.row_stack(mses), np.row_stack(mses_sem), labels, save_name_suffix=save_name_suffix)
  plot_selected_params(np.stack(params, axis=0), np.stack(params_sem, axis=0), labels, save_name_suffix=save_name_suffix)
  plot_selected_scale_factors(np.row_stack(inv_temps), np.row_stack(inv_temps_sem), labels, save_name_suffix=save_name_suffix)
  plot_selected_scores(np.row_stack(head_scores), np.row_stack(head_scores_sem), 
                       np.row_stack(copy_scores), np.row_stack(copy_scores_sem), labels, save_name_suffix=save_name_suffix)

def plot_selected_fit_all(mses, mses_sem, labels, save_name_suffix=''):
  # plot the average CMR distance of the top heads across models
  # NOTE: current implementation only draws an odd number of models nicely
  plt.figure(figsize=[6*cm, 3*cm], dpi=300)
  ax = plt.gca()
  ax.tick_params(axis='both', labelsize=5)
  total_width = 0.8
  bar_width = total_width/mses.shape[0]
  x = np.arange(labels.shape[1])
  for i in range(mses.shape[0]):
    offset = bar_width * (i - mses.shape[0]//2)
    plt.bar(x+offset, mses[i], yerr=mses_sem[i],
            width=bar_width, label=[l.split()[0] for l in labels[i]],
            capsize=0.5, error_kw={'elinewidth': 0.5, 'capthick': 0.5})
  plt.xticks(x, [' '.join(l.split()[1:]) for l in labels[0]])
  plt.ylabel('CMR distance', fontsize=5)
  plt.tight_layout()
  handles, labels = plt.gca().get_legend_handles_labels()
  by_label = dict(zip(labels, handles))
  ax.legend(by_label.values(), by_label.keys(), fontsize=5)
  plt.savefig('./figs/selected/fit_all_xtra_{}.pdf'.format(save_name_suffix), dpi=300)

def plot_selected_params_all(params, params_sem, labels, save_name_suffix=''):
  # plot the average beta_enc, beta_dec, and gamma_FT of the top heads in a model
  # NOTE: current implementation only draws an odd number of models nicely
  fig, axes = plt.subplots(1, 3, figsize=[14*cm, 3*cm], dpi=300)
  for ax in axes: 
    ax.tick_params(axis='both', labelsize=5)
  
  total_width=0.8
  bar_width = total_width/params.shape[0]
  x = np.arange(labels.shape[1])
  for i in range(params.shape[0]):
    beta_enc, beta_dec, gamma_ft = params[i,:,0], params[i,:,1], params[i,:,2]
    beta_enc_sem, beta_dec_sem, gamma_ft_sem = params_sem[i,:,0], params_sem[i,:,1], params_sem[i,:,2]
    offset = bar_width * (i - params.shape[0]//2)
    axes[0].bar(x+offset, beta_enc, yerr=beta_enc_sem,
                width=bar_width, label=[l.split()[0] for l in labels[i]],
                capsize=0.5, error_kw={'elinewidth': 0.5, 'capthick': 0.5})
    axes[0].set_ylabel(r'$\beta_{\rm enc}$', fontsize=5)
    axes[0].set_ylim([0,1])
    axes[1].bar(x+offset, beta_dec.flatten(), yerr=beta_dec_sem.flatten(),
                width=bar_width, label=[l.split()[0] for l in labels[i]],
                capsize=0.5, error_kw={'elinewidth': 0.5, 'capthick': 0.5})
    axes[1].set_ylabel(r'$\beta_{\rm rec}$', fontsize=5)
    axes[1].set_ylim([0,1])
    axes[2].bar(x+offset, gamma_ft.flatten(), yerr=gamma_ft_sem.flatten(),
                width=bar_width, label=[l.split()[0] for l in labels[i]],
                capsize=0.5, error_kw={'elinewidth': 0.5, 'capthick': 0.5})
    axes[2].set_ylabel(r'$\gamma_{\rm FT}$', fontsize=5)
    axes[2].set_ylim([0,1])
  axes[0].set_xticks(x, labels=[' '.join(l.split()[1:]) for l in labels[0]])
  axes[1].set_xticks(x, labels=[' '.join(l.split()[1:]) for l in labels[0]])
  axes[2].set_xticks(x, labels=[' '.join(l.split()[1:]) for l in labels[0]])
  fig.tight_layout()
  handles, labels = plt.gca().get_legend_handles_labels()
  by_label = dict(zip(labels, handles))
  ax.legend(by_label.values(), by_label.keys(), fontsize=5)
  plt.savefig('./figs/selected/params_all_xtra_{}.pdf'.format(save_name_suffix), dpi=300)

def plot_selected_scale_factors_all(scale_factors, scale_factors_sem, labels, save_name_suffix=''):
  # plot the scale factor (tau^-1) of the top heads in a model
  # NOTE: current implementation only draws an odd number of models nicely
  plt.figure(figsize=[5*cm, 3*cm], dpi=300)
  ax = plt.gca()
  ax.tick_params(axis='both', labelsize=5)
  total_width = 0.8
  bar_width = total_width/scale_factors.shape[0]
  x = np.arange(labels.shape[1])
  for i in range(scale_factors.shape[0]):
    offset = bar_width * (i - scale_factors.shape[0]//2)
    plt.bar(x+offset, scale_factors[i], yerr=scale_factors_sem[i],
            width=bar_width, label=[l.split()[0] for l in labels[i]],
            capsize=0.5, error_kw={'elinewidth': 0.5, 'capthick': 0.5})
  plt.xticks(x, [' '.join(l.split()[1:]) for l in labels[0]])
  plt.ylabel(r'$\tau^{-1}$', fontsize=5)
  plt.tight_layout()
  handles, labels = plt.gca().get_legend_handles_labels()
  by_label = dict(zip(labels, handles))
  ax.legend(by_label.values(), by_label.keys(), fontsize=5)
  plt.savefig('./figs/selected/scale_factor_all_xtra_{}.pdf'.format(save_name_suffix), dpi=300)

def plot_selection_cmp_all(models, selections, save_name_suffix=''):
  # compare the top n heads of different models given different values of n
  # NOTE: current implementation only draws an odd number of models nicely
  mses, params, inv_temps = [[] for _ in range(len(models))], [[] for _ in range(len(models))], [[] for _ in range(len(models))]
  mses_sem, params_sem, inv_temps_sem = [[] for _ in range(len(models))], [[] for _ in range(len(models))], [[] for _ in range(len(models))]
  labels = [[] for _ in range(len(models))]
  for i, selection in enumerate(selections):
    for j, s in enumerate(selection):
      mses[j].append(s['MSE'].mean())
      mses_sem[j].append(sem(s['MSE']))
      params[j].append(s['params'].mean(axis=0))
      params_sem[j].append(sem(s['params'], axis=0))
      inv_temps[j].append(s['scale_factors'].mean())
      inv_temps_sem[j].append(sem(s['scale_factors']))
      labels[j].append(models[j] + ' ' + s['selection_name'])
  plot_selected_fit_all(np.array(mses), np.array(mses_sem), np.array(labels), save_name_suffix=save_name_suffix)
  plot_selected_params_all(np.array(params), np.array(params_sem), np.array(labels), save_name_suffix=save_name_suffix)
  plot_selected_scale_factors_all(np.array(inv_temps), np.array(inv_temps_sem), np.array(labels), save_name_suffix=save_name_suffix)

def plot_saved_selections(models=['qwen-7b', 'mistral-7b', 'llama3-8b']):
  # plot top heads from saved records
  top20s, top50s, top100s, top200s = [], [], [], []
  for model in models:
    top_crp_fit = joblib.load('saved_top_heads/cmr_top_crp_fit_{}.pkl'.format(model))
    top20s.append(top_crp_fit['top20'])
    top50s.append(top_crp_fit['top50'])
    top100s.append(top_crp_fit['top100'])
    top200s.append(top_crp_fit['top200'])
  plot_selection_cmp_all(models, [top20s, top50s, top100s, top200s], save_name_suffix='fit')

  top20is, top50is, top100is, top200is = [], [], [], []
  for model in models:
    top_crp_fit = joblib.load('saved_top_heads/cmr_top_head_score_{}.pkl'.format(model))
    top20is.append(top_crp_fit['top20'])
    top50is.append(top_crp_fit['top50'])
    top100is.append(top_crp_fit['top100'])
    top200is.append(top_crp_fit['top200'])
  plot_selection_cmp_all(models, [top20is, top50is, top100is, top200is], save_name_suffix='hs')

def select_subset_top(selection, n):
  # select top n heads from the selection
  mses = selection['MSE']
  top_heads = np.argpartition(mses, n)[:n]
  res = dict()
  for k, v in selection.items():
    if k == 'selection':
      res[k] = dict()
      for k1, v1 in v.items():
        res[k][k1] = np.array(v1)[top_heads]
    elif k == 'selection_name':
      res[k] = 'top {}'.format(n) 
    else:
      res[k] = np.array(v)[top_heads]
  return res

if __name__ == '__main__':
  # parse command line arguments
  args = parser.parse_args()
  models = args.models
  n_top = sorted(args.n_top, reverse=True)  # sort top selection sizes from the largest to the smallest
  use_saved = args.use_saved

  if use_saved:
    plot_saved_selections(models=models)
  else:
    # compute all top heads from scratch
    for model in models:
      print(model)
      # first find and plot top CMR-like heads in each model
      all_selection_cmr = []
      for i, n in enumerate(n_top):
        print('Selecting top {} CMR-like heads...'.format(n))
        if i == 0:  # only fit the largest subset
          selected = select_top_heads([model], n)
          all_selection_cmr.append(selected)
        else:       # all other selections are subsets of the largest subset
          assert(len(all_selection_cmr) > 0)
          selected = select_subset_top(all_selection_cmr[0], n)
          all_selection_cmr.append(selected)
      plot_selection_cmp(all_selection_cmr, save_name_suffix='fit_{}'.format(model))

      # save top CMR-like heads
      # d = dict()
      # for i, n in enumerate(n_top):
      #   d['top' + str(n)] = all_selection_cmr[i]
      # joblib.dump(d, 'saved_top_heads/cmr_top_crp_fit_{}.pkl'.format(model))
      # print('Selections saved.')

      # then find and plot top induction head over training
      all_selection_hs = []
      for i, n in enumerate(n_top):
        print('Selecting top {} induction heads...'.format(n))
        if i == 0:  # only fit the largest subset
          selected = select_top_induction_heads([model], n)
          all_selection_hs.append(selected)
        else:       # all other selections are subsets of the largest subset
          assert(len(all_selection_hs) > 0)
          selected = select_subset_top(all_selection_hs[0], n)
          all_selection_hs.append(selected)
      print('Done.')
      plot_selection_cmp(all_selection_hs, save_name_suffix='hs_{}'.format(model))

      # save top induction heads
      # d = dict()
      # for i, n in enumerate(n_top):
      #   d['top' + str(n)] = all_selection_hs[i]
      # joblib.dump(d, 'saved_top_heads/cmr_top_head_score_{}.pkl'.format(model))
      # print('Selections saved.')
