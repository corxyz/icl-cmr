# plot the distribution of the relative position of the 2 layers w/ the highest proportion of CMR-like heads across models
# Models included: GPT2, all Pythia models, Qwen-7B, Mistral-7B, Llama3-8B
import joblib
import matplotlib.pyplot as plt
from fit_attn_score import fit
from util import *

########################################
# Define constants & hyperparameters
########################################

cm = 1/2.54

select_range = 5                        # magnitude of maximum CRP relative lag considered for fitting

mse_thres = 0.5                         # threshold of CMR distance (only plot results for heads under the threshold)

shape_dict = {                          # (n_layers, n_heads) of each model
  'gpt2-small': (12,12),
  'pythia-70m': (6,8),
  'pythia-160m': (12,12),
  'pythia-410m': (24,16),
  'pythia-1b': (16,8),
  'pythia-1.4b': (24,16),
  'pythia-2.8b': (32,32),
  'pythia-6.9b': (32,32),
  'pythia-12b': (36,40),
  'qwen-7b': (32,32),
  'mistral-7b': (32,32),
  'llama3-8b': (32,32)
}

########################################
# Plotting function
########################################

def plot_top2_layer_rel_pos(models=['gpt2-small', 'pythia-70m','pythia-160m', 'pythia-410m', 
                                    'pythia-1b', 'pythia-1.4b', 'pythia-2.8b', 'pythia-6.9b', 
                                    'pythia-12b', 'qwen-7b', 'mistral-7b', 'llama3-8b']):
  # Plot histogram of the two layers with the highest proportion of CMR-like heads across models
  n_layers_max, n_heads_max = max(shape_dict.values(), key=lambda sub: sub[0])
  mses = np.zeros((len(models), n_layers_max, n_heads_max))
  mses[:] = np.nan
  top2_rel_pos = []

  for i, model_name in enumerate(models):
    # load attention scores
    if 'pythia' in model_name:
      all_head_scores = joblib.load('./saved_scores/{}-deduped-v0/induction_head_all_scores_{}_cp142.pkl'.format(model_name, 
                                                                                                    model_name.replace('-', '_')))
    elif model_name == 'gpt2-small':
      all_head_scores = joblib.load('./saved_scores/gpt2-small/all_scores_gpt2_small.pkl')
    else:
      all_head_scores = joblib.load('./saved_scores/{}/induction_head_all_scores_{}.pkl'.format(model_name, model_name))
    # compute the CMR distance of each head
    n_layers, n_heads = all_head_scores['sorted_labels'].shape
    attn_scores = load_scores_in_range(all_head_scores, 'sorted_CRP_scores', select_range=select_range)
    fit_res = fit(attn_scores)
    mses[i, :n_layers, :n_heads] = fit_res['fitted_MSE']
    # compute % CMR distance < threshold for each layer
    mask = [~np.isnan(mses[i,l]) for l in range(mses.shape[1])]
    low_mse_pct = [(mses[i,l,mask[l]] < mse_thres).mean()*100 for l in range(mses.shape[1])]
    # record the relative position of the two layers with the highest fraction
    top2_ind = np.argpartition(low_mse_pct[:n_layers], -2)[-2:]
    top2_rel_pos = np.concatenate((top2_rel_pos, top2_ind / (shape_dict[model_name][0] - 1) * 100))
  
  # plot the distribution of top 2 layers
  plt.figure(figsize=[5*cm, 4*cm], dpi=300)
  ax = plt.gca()
  ax.tick_params(axis='both', labelsize=5)
  plt.hist(top2_rel_pos)
  plt.xlim([0, 110])
  plt.xticks(np.arange(0, 110, 20))
  plt.xlabel('Relative layer position (%)', fontsize=5)
  plt.ylabel('Count', fontsize=5)
  plt.tight_layout()
  plt.savefig('./figs/layer_pos_dist/layer_pos_dist.pdf')

plot_top2_layer_rel_pos()
