# ablate the top 10% CMR-like heads of the specified model
# compare against random ablation

import gc, time, random, argparse
from copy import deepcopy
from functools import partial

import numpy as np
from scipy.stats import ttest_rel
import torch, joblib
from transformer_lens import HookedTransformer, utils
from datasets import load_dataset

##################################################
# Define setting, access token & hyperparameters
##################################################

torch.set_grad_enabled(False)
access_token = ""                 # replace with your own huggingface access_token

# ICL score parameters
idx1, idx2 = 500, 50
# minimum prompt length in evaluation
min_len = 512
# number of examples to evaluate
n_eval = 1000

##################################################
# Load C4 dataset
##################################################

# load evaluation dataset
c4_dataset = load_dataset("allenai/c4", "en", streaming=True)

##################################################
# Define CLI parser
##################################################

parser = argparse.ArgumentParser(description='Fit attention scores with CMR (compare across sizes).')
parser.add_argument('-m', '--model_name', type=str, default='gpt2-small',
                    help='Model name')
parser.add_argument('-d', '--device', type=str, default='cpu',
                    help='Device name')

##################################################
# Ablation experiment
##################################################

def get_ablate_list(model_name):
  # parse and get a list of the top CMR-like heads of the given model
  # all head labels are determined using select_fit
  if model_name == 'llama3-8b':
    # Llama-3-8B
    a = ['L2H21', 'L17H26', 'L24H16', 'L24H18', 'L24H19', 'L24H20', 'L24H22', 'L24H23', 'L24H24', 'L24H25', 'L16H25', 'L24H27', 'L11H12', 'L25H5', 'L25H6', 'L13H9', 'L10H14', 'L25H15', 'L10H2', 'L10H1', 'L18H28', 'L19H1', 'L9H7', 'L16H20', 'L19H3', 'L13H20', 'L9H3', 'L16H19', 'L26H12', 'L26H13', 'L22H13', 'L23H7', 'L26H14', 'L26H15', 'L8H8', 'L8H1', 'L19H23', 'L27H4', 'L27H5', 'L22H11', 'L23H20', 'L27H6', 'L22H8', 'L27H7', 'L20H1', 'L20H2', 'L22H28', 'L20H3', 'L27H20', 'L27H21', 'L30H30', 'L30H29', 'L27H22', 'L14H15', 'L27H23', 'L23H5', 'L28H13', 'L22H29', 'L28H15', 'L20H13', 'L20H14', 'L20H15', 'L5H11', 'L5H10', 'L30H21', 'L30H20', 'L22H2', 'L23H21', 'L5H9', 'L22H0', 'L5H8', 'L4H16', 'L20H26', 'L21H2', 'L30H13', 'L23H22', 'L15H20', 'L15H30', 'L29H20', 'L29H21', 'L21H24', 'L2H25', 'L2H23', 'L2H22', 'L2H20', 'L17H27', 'L16H23', 'L18H20', 'L17H31', 'L22H31', 'L22H1', 'L25H14', 'L24H17', 'L14H14', 'L18H16', 'L15H26', 'L14H28', 'L16H8', 'L25H13', 'L19H0', 'L18H23', 'L20H9']
  elif model_name == 'qwen-7b':
    # Qwen-7B
    a = ['L14H14', 'L22H0', 'L22H16', 'L9H31', 'L21H24', 'L23H4', 'L10H11', 'L23H12', 'L21H18', 'L23H15', 'L23H23', 'L23H28', 'L24H0', 'L21H12', 'L10H25', 'L21H9', 'L24H20', 'L24H22', 'L24H27', 'L18H4', 'L20H29', 'L17H5', 'L20H27', 'L25H14', 'L20H8', 'L12H7', 'L18H6', 'L25H21', 'L25H22', 'L25H27', 'L12H12', 'L26H0', 'L20H7', 'L26H2', 'L5H11', 'L12H20', 'L26H13', 'L19H22', 'L26H28', 'L19H15', 'L19H5', 'L27H29', 'L28H5', 'L22H3', 'L4H0', 'L27H31', 'L21H30', 'L22H10', 'L22H11', 'L22H15', 'L18H12', 'L9H27', 'L16H13', 'L14H4', 'L10H4', 'L0H1', 'L22H26', 'L22H28', 'L22H29', 'L10H5', 'L15H23', 'L23H7', 'L18H18', 'L21H19', 'L23H9', 'L10H14', 'L23H13', 'L23H21', 'L18H3', 'L7H10', 'L24H12', 'L21H1', 'L28H4', 'L19H7', 'L18H31', 'L20H24', 'L20H23', 'L30H29', 'L29H7', 'L20H22', 'L12H5', 'L16H27', 'L25H23', 'L25H25', 'L18H7', 'L12H11', 'L14H21', 'L25H29', 'L26H8', 'L20H2', 'L14H19', 'L27H21', 'L19H30', 'L4H29', 'L14H15', 'L19H25', 'L13H0', 'L19H20', 'L3H2', 'L18H10', 'L19H9', 'L19H4']
  elif model_name == 'mistral-7b':
    # Mistral-7B
    a = ['L4H1', 'L15H31', 'L31H19', 'L31H6', 'L31H4', 'L30H10', 'L30H3', 'L30H2', 'L29H22', 'L29H9', 'L28H26', 'L28H25', 'L28H18', 'L28H16', 'L28H9', 'L27H29', 'L27H27', 'L27H26', 'L26H24', 'L26H6', 'L26H4', 'L25H29', 'L25H27', 'L23H15', 'L22H29', 'L22H1', 'L22H0', 'L21H11', 'L21H10', 'L21H9', 'L20H31', 'L20H29', 'L20H28', 'L20H23', 'L20H19', 'L20H18', 'L20H14', 'L20H13', 'L20H10', 'L19H11', 'L19H8', 'L19H6', 'L19H5', 'L19H4', 'L18H31', 'L18H30', 'L18H29', 'L18H23', 'L18H20', 'L18H14', 'L18H13', 'L18H12', 'L18H3', 'L18H2', 'L18H1', 'L18H0', 'L17H3', 'L17H1', 'L16H21', 'L16H20', 'L16H15', 'L16H7', 'L15H28', 'L15H23', 'L15H21', 'L15H9', 'L15H8', 'L15H7', 'L15H6', 'L15H1', 'L15H0', 'L14H26', 'L14H24', 'L14H0', 'L12H10', 'L12H9', 'L12H7', 'L12H6', 'L9H28', 'L9H26', 'L9H25', 'L9H20', 'L7H19', 'L7H18', 'L6H2', 'L2H21', 'L2H22', 'L2H23', 'L4H12', 'L31H31', 'L19H12', 'L11H14', 'L30H25', 'L19H16', 'L4H15', 'L24H5', 'L10H11', 'L22H3', 'L16H28', 'L11H20', 'L29H10', 'L19H9']
  elif model_name == 'pythia-12b':
    # Pythia-12B
    a = ['L15H11', 'L5H39', 'L5H35', 'L19H1', 'L6H4', 'L18H34', 'L5H32', 'L5H29', 'L6H13', 'L6H16', 'L6H17', 'L5H19', 'L5H11', 'L5H10', 'L6H23', 'L5H8', 'L6H27', 'L6H30', 'L17H34', 'L6H31', 'L20H10', 'L5H3', 'L17H30', 'L20H13', 'L7H0', 'L7H4', 'L7H11', 'L7H15', 'L7H26', 'L20H21', 'L4H23', 'L8H7', 'L8H8', 'L8H9', 'L8H22', 'L4H8', 'L17H5', 'L26H22', 'L4H6', 'L16H29', 'L29H24', 'L9H0', 'L3H29', 'L9H17', 'L21H6', 'L9H36', 'L9H38', 'L21H15', 'L16H10', 'L10H5', 'L25H15', 'L10H19', 'L11H2', 'L16H0', 'L11H27', 'L21H25', 'L24H19', 'L23H34', 'L12H37', 'L15H26', 'L2H13', 'L23H18', 'L13H12', 'L14H10', 'L21H38', 'L6H1', 'L20H35', 'L15H37', 'L9H16', 'L25H23', 'L6H3', 'L18H37', 'L15H35', 'L14H22', 'L10H2', 'L26H13', 'L12H3', 'L18H20', 'L3H32', 'L24H9', 'L17H12', 'L20H25', 'L22H39', 'L4H9', 'L23H3', 'L23H6', 'L10H33', 'L23H12', 'L23H13', 'L20H20', 'L21H26', 'L21H30', 'L18H2', 'L20H8', 'L23H24', 'L6H39', 'L21H2', 'L4H31', 'L20H38', 'L2H36', 'L4H1', 'L15H21']
  elif model_name == 'pythia-6.9b':
    # Pythia-6.9B
    a = ['L8H24', 'L5H23', 'L5H24', 'L12H13', 'L5H27', 'L5H15', 'L5H29', 'L12H7', 'L12H6', 'L13H4', 'L5H11', 'L21H12', 'L11H30', 'L6H3', 'L13H12', 'L5H6', 'L19H7', 'L19H3', 'L5H1', 'L19H1', 'L4H30', 'L13H24', 'L4H29', 'L6H13', 'L18H23', 'L6H18', 'L18H5', 'L18H3', 'L4H22', 'L6H29', 'L22H25', 'L6H31', 'L7H1', 'L27H27', 'L22H30', 'L23H1', 'L14H9', 'L4H7', 'L7H2', 'L14H16', 'L10H30', 'L10H28', 'L4H5', 'L3H31', 'L14H20', 'L17H12', 'L14H27', 'L7H16', 'L7H19', 'L7H20', 'L7H23', 'L23H17', 'L10H13', 'L7H30', 'L15H1', 'L15H4', 'L7H31', 'L23H21', 'L15H15', 'L10H1', 'L17H3', 'L10H0', 'L9H30', 'L15H17', 'L9H25', 'L23H23', 'L8H4', 'L15H19', 'L23H25', 'L15H20', 'L16H30', 'L9H17', 'L3H8', 'L23H30', 'L9H14', 'L8H11', 'L16H1', 'L16H23', 'L9H8', 'L8H16', 'L9H5', 'L8H17', 'L8H18', 'L16H17', 'L8H29', 'L24H10', 'L16H20', 'L5H20', 'L12H16', 'L11H8', 'L28H6', 'L4H18', 'L8H28', 'L10H17', 'L10H12', 'L6H15', 'L13H14', 'L11H1', 'L7H0', 'L17H22', 'L14H6', 'L14H1']
  elif model_name == 'pythia-2.8b':
    # Pythia-2.8B
    a = ['L2H15', 'L16H0', 'L15H30', 'L15H24', 'L15H21', 'L15H14', 'L15H12', 'L16H11', 'L15H1', 'L14H23', 'L16H22', 'L16H23', 'L16H24', 'L16H25', 'L14H10', 'L14H9', 'L13H30', 'L13H29', 'L13H24', 'L13H14', 'L13H10', 'L13H4', 'L17H8', 'L12H8', 'L11H30', 'L17H15', 'L11H19', 'L11H18', 'L11H15', 'L17H22', 'L17H24', 'L17H28', 'L10H31', 'L18H4', 'L18H10', 'L18H12', 'L9H3', 'L19H2', 'L8H24', 'L19H12', 'L8H19', 'L7H28', 'L20H8', 'L20H22', 'L6H27', 'L5H5', 'L5H3', 'L3H22', 'L3H17', 'L3H14', 'L3H11', 'L3H4', 'L2H25', 'L16H2', 'L15H19', 'L6H29', 'L16H4', 'L11H26', 'L17H30', 'L10H28', 'L18H7', 'L12H26', 'L13H1', 'L20H0', 'L17H0', 'L20H18', 'L20H25', 'L21H24', 'L5H25', 'L14H4', 'L5H14', 'L14H11', 'L16H14', 'L14H31', 'L15H2', 'L15H7', 'L15H16', 'L2H13', 'L17H14', 'L22H29', 'L17H7', 'L12H7', 'L13H22', 'L2H19', 'L2H20', 'L12H21', 'L13H18', 'L11H27', 'L14H26', 'L6H28', 'L22H21', 'L19H18', 'L6H4', 'L26H21', 'L15H6', 'L11H31', 'L14H25', 'L25H6', 'L12H10', 'L22H16', 'L13H31', 'L12H25']
  elif model_name == 'pythia-1.4b':
    # Pythia-1.4B
    a = ['L10H0', 'L7H15', 'L6H3', 'L4H0', 'L12H15', 'L12H11', 'L7H3', 'L15H3', 'L12H6', 'L10H9', 'L10H8', 'L8H0', 'L9H8', 'L9H12', 'L11H6', 'L18H8', 'L13H15', 'L13H1', 'L10H10', 'L7H12', 'L1H4', 'L10H11', 'L4H4', 'L15H1', 'L11H3', 'L16H6', 'L4H5', 'L16H3', 'L6H13', 'L11H2', 'L11H14', 'L9H0', 'L13H7', 'L14H6', 'L17H8', 'L14H11', 'L10H5', 'L15H9']
  elif model_name == 'pythia-1b':
    # Pythia-1B
    a = ['L10H2', 'L11H0', 'L4H5', 'L4H4', 'L11H5', 'L12H1', 'L4H1', 'L12H5', 'L5H7', 'L10H5', 'L2H4', 'L12H3']
  elif model_name == 'pythia-160m':
    # Pythia-160M
    a = ['L10H9', 'L4H8', 'L7H1', 'L6H6', 'L9H4', 'L9H2', 'L6H2', 'L8H11', 'L4H10', 'L8H2', 'L5H10', 'L5H9', 'L5H6', 'L4H6']
  elif model_name == 'pythia-70m':
    # Pythia-70M
    a = ['L3H0', 'L3H6', 'L3H5', 'L3H1']
  elif model_name == 'gpt2-small':
    # GPT2
    a = ['L1H11', 'L5H5', 'L5H1', 'L3H0', 'L5H8', 'L8H6', 'L0H5', 'L5H0', 'L0H1', 'L6H9', 'L0H10', 'L6H10', 'L7H2', 'L8H3']
  else:
    raise Exception('Model {} is not supported. Please check the spelling.'.format(model_name))

  ablate_list = []
  for s in a:
    layer_to_ablate = int(s[s.index('L')+1:s.index('H')])
    head_index_to_ablate = int(s[s.index('H')+1:])
    ablate_list.append((layer_to_ablate, head_index_to_ablate))
  return ablate_list

def head_ablation_hook_full(value, hook, head_index_to_ablate):
  value[:, :, head_index_to_ablate, :] = 0.
  return value

def load_ablate_model(base_model_name, ablate_list, device='cpu'):
  # load original pretrained model
  model = HookedTransformer.from_pretrained(base_model_name, device=device)

  # randomly choose the same number of heads as specified in ablate_list to ablate
  ablate_rand_list = set()
  while len(ablate_rand_list) < len(ablate_list):
      head_index_to_ablate = random.randint(0, model.cfg.n_heads-1)
      layer_to_ablate = random.randint(0, model.cfg.n_layers-1)
      ablate_rand_list.add((layer_to_ablate, head_index_to_ablate))

  model.reset_hooks()
  # ablate top CMR-like heads
  model_ablated = deepcopy(model)
  model_ablated.reset_hooks()
  for layer_to_ablate, head_index_to_ablate in ablate_list:
    head_ablation_hook = partial(head_ablation_hook_full, head_index_to_ablate=head_index_to_ablate)
    model_ablated.add_hook(utils.get_act_name("v", layer_to_ablate), head_ablation_hook)
  # ablate random heads (control)
  model_ablated_rand = deepcopy(model)
  model_ablated_rand.reset_hooks()
  for layer_to_ablate, head_index_to_ablate in ablate_rand_list:
    head_ablation_hook = partial(head_ablation_hook_full, head_index_to_ablate=head_index_to_ablate)
    model_ablated_rand.add_hook(utils.get_act_name("v", layer_to_ablate), head_ablation_hook)
  return model, model_ablated, model_ablated_rand

def ablation_exp(model_name, base_model_name, device='cpu'):
  original_icl_score = np.zeros(n_eval)
  ablated_icl_score = np.zeros(n_eval)
  ablated_rand_icl_score = np.zeros(n_eval)

  ablate_list = get_ablate_list(model_name)
  model, model_ablated, model_ablated_rand = load_ablate_model(base_model_name, ablate_list, device=device)

  i = 0
  for v in c4_dataset['validation']:
    prompt_text = v['text']
    prompt_tokens = model.to_tokens(prompt_text)
    if prompt_tokens.shape[1] < min_len:    # ensure minimum prompt length
      continue
    else:
      print('Trial {}...'.format(i), end='\t')
      start_time = time.time()
      prompt_tokens = prompt_tokens[:,:min_len]
      # compute in-context learning score similar to the one defined in 
      # https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html#results-in-context-score
      # i.e., ICL score = avg loss at token index 500 - avg loss at token index 50 
      original_logits = model(prompt_tokens, return_type="logits")
      loss_by_index = utils.lm_cross_entropy_loss(original_logits, prompt_tokens, per_token=True).numpy()[0]
      original_icl_score[i] = loss_by_index[idx1] - loss_by_index[idx2]
      ablated_logits = model_ablated(prompt_tokens, return_type="logits")
      loss_by_index = utils.lm_cross_entropy_loss(ablated_logits, prompt_tokens, per_token=True).numpy()[0]
      ablated_icl_score[i] = loss_by_index[idx1] - loss_by_index[idx2]
      ablated_rand_logits = model_ablated_rand(prompt_tokens, return_type="logits")
      loss_by_index = utils.lm_cross_entropy_loss(ablated_rand_logits, prompt_tokens, per_token=True).numpy()[0]
      ablated_rand_icl_score[i] = loss_by_index[idx1] - loss_by_index[idx2]
      i += 1
      gc.collect()
      end_time = time.time()
      print('Time elapsed: {:.3f}s.'.format(end_time-start_time))
    if i >= n_eval:
      break
  
  # paired t-test
  print('original vs. ablating top CMR-like heads', ttest_rel(original_icl_score, ablated_icl_score))
  print('original vs. ablating random heads', ttest_rel(original_icl_score, ablated_rand_icl_score))
  print('ablating top CMR-like vs. random heads', ttest_rel(ablated_icl_score, ablated_rand_icl_score))

  # save results
  joblib.dump({
    'original_icl_score': original_icl_score,
    'ablated_icl_score': ablated_icl_score,
    'ablated_rand_icl_score': ablated_rand_icl_score,
}, './saved_ablation/ablation_experiment_{}.pkl'.format(model_name))

if __name__ == '__main__':
  # parse command line arguments
  args = parser.parse_args()
  model_name = args.model_name.lower()
  device = args.device
  # get model key to load from TransformerLens
  if 'pythia' in model_name:
    base_model_name = model_name + '-deduped-v0'
  elif 'mistral' in model_name:
    base_model_name = 'mistralai/Mistral-7B-v0.1'
  elif 'llama' in model_name:
    base_model_name = 'meta-llama/Meta-Llama-3-8B'
  elif 'qwen' in model_name:
    base_model_name = 'Qwen/Qwen-7B'
  else:
    base_model_name = model_name.lower()

  ablation_exp(model_name, base_model_name, device=device)
