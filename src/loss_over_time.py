# plot model loss over training for selected models
# reproduces Figure 7a

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

########################################
# Define constants & global attributes
########################################

cm = 1/2.54

file_prefix = 'loss_'                   # prefix of saved file containing loss over time
cp_interv = 10                          # checkpoint interval
n_cp = 143                              # number of checkpoints

# define color map
cm_name = "tab20"
cmap = mpl.colormaps[cm_name]
colors = cmap.colors

########################################

def load_model_loss(model_names):
  # load model loss of specified models
  model_loss = []
  for model_name in model_names:
    loss = np.load('./loss/{}{}.npy'.format(file_prefix, model_name))
    model_loss.append(loss)
  model_loss = np.row_stack(model_loss)
  return model_loss

def plot_model_loss(model_loss, model_names):
  # plot model loss over time
  cps = list(range(0, n_cp, cp_interv)) + [n_cp - 1]    # checkpoints
  n_model = model_loss.shape[0]
  plt.figure(figsize=[6*cm, 3*cm], dpi=300)
  ax = plt.gca()
  ax.tick_params(axis='both', labelsize=5)
  ax.set_prop_cycle(color=colors)
  for i in range(n_model):
    loss = model_loss[i]
    label = model_names[i].split('-')[1]
    plt.plot(cps, loss, lw=0.4, label=label)
  plt.xlabel('Checkpoint', fontsize=5)
  plt.ylabel('Loss', fontsize=5)
  plt.legend(bbox_to_anchor=(1.01, 0), loc='lower left', fontsize=5, ncol=2)
  plt.tight_layout(pad=0.3)
  plt.savefig('./figs/loss/loss_over_time.pdf')

def main(model_names):
  model_loss = load_model_loss(model_names)
  plot_model_loss(model_loss, model_names)

model_names = ['pythia-70m-deduped-v0','pythia-160m-deduped-v0','pythia-410m-deduped-v0',
               'pythia-1b-deduped-v0','pythia-1.4b-deduped-v0', 'pythia-2.8b-deduped-v0',
               'pythia-6.9b-deduped-v0','pythia-12b-deduped-v0']
main(model_names)

