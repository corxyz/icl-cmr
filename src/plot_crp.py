# plot pre-computed CRP, simulated from CMR
# reproduces Figure 4 and Supplementary Figure S1

import math
import numpy as np
import matplotlib.pyplot as plt

from util import truncate_crp_to_range

########################################
# Define constants & hyperparameters
########################################

cm = 1/2.54

beta_gridsize = 1/20                    # size of each grid for grid search of beta
gamma_gridsize = 1/10                   # size of each grid for grid search of gamma

########################################
# Load pre-computed CRP
########################################

crp = np.load('./saved_crps/cmr_crp_avg_20_11.npy')
crp_sem = np.load('./saved_crps/cmr_crp_sem_20_11.npy')
crp_params = np.mgrid[beta_gridsize:1+beta_gridsize:beta_gridsize, 0:1+beta_gridsize:beta_gridsize, 
                      0:1+gamma_gridsize:gamma_gridsize]

########################################
# Plotting functions
########################################

def make_square_axes(ax):
    ax.set_aspect(1 / ax.get_data_ratio())

def plot_crp(crp, minp=0.1, max_lag=5, save_name='example_crp'):
  # plot a given crp curve
  x = np.arange(-max_lag, max_lag+1)  # restrict the maximum relative lag considered
  crp_range = np.arange(-(crp.size // 2), crp.size // 2 + 1)
  select_idx = (crp_range >= -max_lag) & (crp_range <= max_lag)

  plt.figure(figsize=[3.5*cm, 3.5*cm], dpi=300)
  plt.clf()
  plt.plot(x, crp[select_idx], 'ko-', markersize=0.7, linewidth=1)
  plt.xticks([-max_lag,0,max_lag], fontsize=5)
  if np.nanmax(crp) < 0.1:
    upperlim = round(max(np.nanmax(crp),minp), 2)
    nticks = 2
  else:
    upperlim = round(max(min(np.nanmax(crp),1),minp), 1)
    nticks = int(upperlim*10)+1 if upperlim < 1 else 3
  plt.yticks(np.linspace(0, upperlim, nticks), fontsize=5)
  plt.xlabel('Lag', fontsize=7)
  plt.ylabel('CRP', fontsize=7)
  ax = plt.gca()
  ax.tick_params(length=1, direction='in')
  make_square_axes(ax)
  plt.axvline(x = 0, color='k', linewidth=1, linestyle='--')
  plt.subplots_adjust(left=0.25, bottom=0.2, right=.95, top=.95, wspace=0, hspace=0)
  plt.savefig(save_name + '.pdf', dpi=300)

def plot_all_crp(crp, crp_sem, crp_params, plot_max_lag=5):
  # plot all crps in a given range (Fig. S1)
  n_beta_enc, n_beta_dec, _ = crp_params.shape[1:]
  x = np.arange(-plot_max_lag, plot_max_lag+1)
  _, axes = plt.subplots(n_beta_enc, n_beta_dec, figsize=[n_beta_dec*2*cm, n_beta_enc*2*cm], dpi=300, sharex=True, sharey=True)
  for i in range(n_beta_enc):
    for j in range(n_beta_dec):
      y_g0 = truncate_crp_to_range(crp[i,j,0], select_range=plot_max_lag)
      yerr_g0 = truncate_crp_to_range(crp_sem[i,j,0], select_range=plot_max_lag)
      axes[i,j].errorbar(x, y_g0, yerr=yerr_g0, color='k', lw=1)
      y_g1 = truncate_crp_to_range(crp[i,j,-1], select_range=plot_max_lag)
      yerr_g1 = truncate_crp_to_range(crp_sem[i,j,-1], select_range=plot_max_lag)
      axes[i,j].errorbar(x, y_g1, yerr=yerr_g1, color='grey', lw=1)
      axes[i,j].tick_params(axis='both', direction='in', labelsize=5)
      axes[i,j].set_title(r'$\beta_{{enc}}={:.2},\beta_{{rec}}={:.2}$'.format(crp_params[0,i,j,0], crp_params[1,i,j,0]),
                          fontsize=5)
  # label x-axis on the last row
  for i in range(n_beta_dec):
    axes[-1,i].set_xlabel('Lag', fontsize=7)
    axes[-1,i].set_xticks([-plot_max_lag, -1, 0, 1, plot_max_lag])
  # label y-axis on the first column
  for i in range(n_beta_enc):
    axes[i,0].set_ylabel('CRP', fontsize=7)
    axes[i,0].set_yticks([0, 0.5, 1])
  plt.tight_layout()
  plt.savefig('./figs/supp-crp-reference.pdf', dpi=300)

def main(plot_max_lag=5):
  # plot example CRPs (Figure 4)
  plot_crp(crp[-1,-1,0], save_name='crp_110')           # beta_enc = beta_rec = 1, gamma = 0
  plot_crp(crp[13,14,0], save_name='crp_07070')         # beta_enc = beta_rec = 0.7, gamma = 0  
  plot_crp(crp[13,14,5], save_name='crp_070705')        # beta_enc = beta_rec = 0.7, gamma = 0.5
  plot_crp(crp[13,14,-1], save_name='crp_07071')        # beta_enc = beta_rec = 0.7, gamma = 1   

  # plot many CRPs (Supplementary Figure S1)
  plot_gridsize = 1/10  # grid size to plot CRPs
  # offset the first param (beta_enc) by 1 since beta_enc > 0
  selected_crp = crp[1::math.floor(plot_gridsize/beta_gridsize),::math.floor(plot_gridsize/beta_gridsize)]
  selected_crp_sem = crp_sem[1::math.floor(plot_gridsize/beta_gridsize),::math.floor(plot_gridsize/beta_gridsize)]
  selected_crp_params = crp_params[:,1::math.floor(plot_gridsize/beta_gridsize),::math.floor(plot_gridsize/beta_gridsize)]
  plot_all_crp(selected_crp, selected_crp_sem, selected_crp_params, plot_max_lag=plot_max_lag)

main()
