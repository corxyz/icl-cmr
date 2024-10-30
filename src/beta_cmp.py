import numpy as np
import joblib
import matplotlib.pyplot as plt

########################################
# Define constant
########################################

cm = 1/2.54

def plot_beta_cmp(fitted_betas, fitted_labels, human_betas, human_labels, save_name_suffix=''):
  # plot average fitted beta_enc and beta_rec of top CMR-like heads (top 20, 50, 100, 200)
  # compare against average beta_enc and beta_rec from human free recall studies
  plt.figure(figsize=[4*cm, 4*cm], dpi=300)
  ax = plt.gca()
  ax.tick_params(axis='both', labelsize=5)
  plt.scatter(fitted_betas[:,0], fitted_betas[:,1], color='b', marker='o', s=4)
  for i, betas in enumerate(fitted_betas):
    if fitted_labels[i] == 'top100':
      plt.annotate(fitted_labels[i], betas, fontsize=4, xytext=(betas[0]-0.2,betas[1]))
    else:
      plt.annotate(fitted_labels[i], betas, fontsize=4)
  plt.scatter(human_betas[:,0], human_betas[:,1], color='r', marker='^', s=4)
  for i, betas in enumerate(human_betas):
    plt.annotate(human_labels[i], betas, fontsize=4)
  plt.axvline(x=human_betas[:,0].min(), color='grey', linestyle='--', linewidth=0.4)
  plt.axvline(x=human_betas[:,0].max(), color='grey', linestyle='--', linewidth=0.4)
  plt.axhline(y=human_betas[:,1].min(), color='grey', linestyle='--', linewidth=0.4)
  plt.axhline(y=human_betas[:,1].max(), color='grey', linestyle='--', linewidth=0.4)
  plt.xlabel(r'$\beta_{\rm enc}$', fontsize=5)
  plt.ylabel(r'$\beta_{\rm rec}$', fontsize=5)
  plt.xlim([0,1])
  plt.ylim([0,1])
  plt.tight_layout(pad=0.3)
  plt.savefig('./figs/human_cmp/beta_cmp_{}.pdf'.format(save_name_suffix), dpi=300)

def beta_cmp():
  # load fitted betas of top heads in pre-trained models
  top_cmr_head = joblib.load('./saved_top_heads/cmr_top_crp_fit.pkl')
  # fitted betas from human studies
  human_betas = np.array([
    [0.62676, 0.62676],     # Sederberg et al., 2008
    [0.481, 0.329],         # Lohnas et al., 2015 - Externalized free recall exp
    [0.182, 0.754],         # Lohnas et al., 2015 - Jang & Huber, 2008
    [0.635, 0.937],         # Lohnas et al., 2015 - Loess, 1967
    [0.745, 0.36],          # Polyn et al., 2009 - Murdock, 1962
    [0.621, 0.179],         # Polyn et al., 2009 - Murdock, 1970
    [0.79, 0.59]            # Zhang et al., 2023 - Kahana et al., 2002
  ])
  study_names = [
    'Sederberg2008', 'Lohnas2015', 'Jang2008', 
    'Loess1967', 'Murdock1962', 'Murdock1970', 'Kahana2002'
  ]

  fitted_betas = []
  selection_names = []
  for selection_name, selection in top_cmr_head.items():
    fitted_betas.append(selection['params'][:,:2].mean(axis=0))
    selection_names.append(selection_name)
  plot_beta_cmp(np.row_stack(fitted_betas), selection_names, human_betas, study_names, save_name_suffix='fit')

beta_cmp()
