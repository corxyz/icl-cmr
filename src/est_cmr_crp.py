# Simulate free recall with CMR and compute CRP

import multiprocessing

import numpy as np
from numpy.random import rand
from scipy.stats import sem

########################################
# Define hyperparameters & switches
########################################

n = 100                                 # number of distinct tokens
max_lag = 8                             # magnitude of maximum CRP relative lag

n_exps = 1000                           # number of recall experiments (simulations)
max_samps = 100                         # maximum number of memory samples

beta_gridsize = 1/20                    # size of each grid for grid search of beta
nBeta = int(1/beta_gridsize)            # number of beta_enc
gamma_gridsize = 1/10                   # size of each grid for grid search of gamma
nGamma = int(1/gamma_gridsize) + 1      # number of gamma
n_s0 = 20                               # number of initial recall positions

omit_zero = False                       # omit CRP at lag=0
verbose = False                         # printing during simulation

########################################
# CMR Simulation
########################################

def encode(beta, n, addAbsorbingSt=True):
  # construct transition matrix, assuming one-hot encodings and uniqueness of the stimuli
  # add an absorbing state to prevent edge effects
  T = np.zeros((n + addAbsorbingSt, n + addAbsorbingSt))
  for i in range(n + addAbsorbingSt - 1):
    T[i, i+1] = 1
  # compute MTF
  M = np.linalg.inv(np.identity(T.shape[0]) - (1-beta)*T)
  M = np.matmul(M, T)
  # compute MFT
  MFT = M
  return M, MFT

def recall(beta, gamma_ft, M, MFT, s0, n_exps, max_samps,
           addAbsorbingSt=True, ensureCnormIs1=True, verbose=True):
  # M: context-to-item associative matrix, specifying the temporal context associated with the item during study
  # MFT: item-to-context associative matrix, specifying which temporal context is used to update the current context vector once the item is recalled
  if MFT is None:
    MFT = M
  samps = np.negative(np.ones((n_exps, max_samps), dtype=int))

  for e in range(n_exps):
    if verbose and n_exps >= 10 and (e+1) % (n_exps//10) == 0: print(str((e+1) * 100 // n_exps) + '%')

    stateIdx = np.arange(0, M.shape[0])    # state indices
    stim = np.identity(M.shape[0])
    t = stim[:,stateIdx[s0]]  # starting context vector (set to the starting state)

    i = 0   # trial-wise sample counter
    while i < max_samps:
      # retrieve distribution over features
      F = np.matmul(M.T,t)
      P = F / np.sum(F)   # NOTE: an alternative is to apply a softmax function over F

      # recall by sampling
      tmp = np.where(rand() <= np.cumsum(P))[0]
      # break if reached absorbing state (similar to unable to produce any further recall)
      if tmp[0] >= M.shape[0] - addAbsorbingSt:
        break
      samps[e,i] = tmp[0]
      f = stim[:, samps[e,i]]
      i += 1

      # get new incoming context vector to update the temporal context
      tIN = (1-gamma_ft) * f + gamma_ft * np.matmul(MFT,f)
      tIN = tIN/np.linalg.norm(tIN)
      assert np.abs(np.linalg.norm(tIN)-1) < 1e-10, 'Norm of cIN is not one'

      # update temporal context
      t = (1-beta) * t + beta * tIN
      t = t/np.linalg.norm(t)
      if ensureCnormIs1:
        assert np.abs(np.linalg.norm(t)-1) < 1e-10, 'Norm of c is not one'

  return samps

########################################
# Compute CRP
########################################

def estimateCRP(samps, s0, max_lag=9, omit_zero=False):
  n_exps, _ = samps.shape
  count = np.zeros((n_exps, max_lag * 2 + 1))  # compute CRP curve for each experiment
  totalTransitions = np.sum(np.array(samps) >= 0, axis=1)
  # accumulate counts of relative positions from consecutive samples
  for e in range(n_exps):
    trial = samps[e,:]
    for i, samp in enumerate(trial):
      if samp < 0: break    # recall ended
      if i == 0:  # the first sample
        relPos = samp - s0
      else:
        relPos = samp - trial[i-1]
      if omit_zero and relPos == 0 or abs(relPos) > max_lag: continue
      count[e, relPos + max_lag] += 1

  # compute mean crp across experiments
  y = count / totalTransitions[:, np.newaxis]
  if omit_zero: y[:, max_lag] = np.nan
  res = np.nanmean(y, axis=0)
  return res

def getCRP(beta_rec, gamma_ft, n, samps, s0, M, max_lag=9, omit_zero=False):
  if beta_rec == 1 and gamma_ft == 0:
    # close form exists
    crp = np.zeros(max_lag * 2 + 1)
    Mpower = np.identity(M.shape[0])
    for d in range(1, max_lag + 1):
      Mpower = np.matmul(Mpower, M)
      P1 = Mpower[s0,:]/Mpower[s0,:].sum()
      P2 = np.append(np.diagonal(M, offset=d), np.zeros(d)) / M.sum(axis=1)
      crp[d + max_lag] = np.nansum(P1 * P2)
  elif beta_rec == 0 and gamma_ft == 0:
    # close form exists
    crp = np.zeros(max_lag * 2 + 1)
    # crp(0) is the probability to drawing two samples with the same index
    P1 = M[s0,:]/M[s0,:].sum()
    crp[max_lag] = np.square(P1).sum()
    for d in range(1, min(n, max_lag+1)):
      # for d > 0, crp(d) is the probability of drawing two samples s1, s2 such that s2 - s1 = d
      P2 = np.append(P1[d:], np.zeros(d))
      crp[d + max_lag] = (P1 * P2).sum()
      # similarly, for d < 0, crp(d) is the probably of drawing two samples s1, s2 such that s2 - s1 = -d
      P2 = np.append(np.zeros(d), P1[:-d])
      crp[-d + max_lag] = (P1 * P2).sum()
  else:
    # use empirically estimated crp instead
    crp = estimateCRP(samps, s0, max_lag=max_lag, omit_zero=omit_zero)
  return crp / crp.sum()

########################################
# Estimate CRPs wrt CMR parameters
########################################

def main(beta_enc, beta_rec, gamma_ft):
  res = np.zeros((n_s0, max_lag * 2 + 1))
  M, MFT = encode(beta_enc, n)
  for s0 in range(n_s0):
    print('beta_enc={}, beta_rec={}, gamma_ft={}, s0={}'.format(beta_enc, beta_rec, gamma_ft, s0))
    # close form not derived; empirically estimate crp
    samps = recall(beta_rec, gamma_ft, M, MFT, s0, n_exps, max_samps, 
                   verbose=verbose)
    res[s0] = getCRP(beta_rec, gamma_ft, n, samps, s0, M, 
                     max_lag=max_lag, omit_zero=omit_zero)
  return res

if __name__ == '__main__':
  pool = multiprocessing.Pool(10)
  # create parameter grid
  # aboid beta = 0 in encoding (equiv. to a discount factor of 1)
  B = np.mgrid[beta_gridsize:1+beta_gridsize:beta_gridsize, 0:1+beta_gridsize:beta_gridsize,
               0:1+gamma_gridsize:gamma_gridsize]
  processes = [[[None] * nGamma for _ in range(nBeta+1)] for _ in range(nBeta)]
  for i in range(nBeta):
    for j in range(nBeta+1):
      for k in range(nGamma):
        processes[i][j][k] = pool.apply_async(main, args=(B[0,i,j,k],B[1,i,j,k],B[2,i,j,k]))
  result = [[[b2.get() for b2 in b1] for b1 in p] for p in processes]
  all_crps = np.array(result)
  np.save('./saved_crps/cmr_crp_avg_{}_{}.npy'.format(nBeta, nGamma), all_crps.mean(axis=2))
  np.save('./saved_crps/cmr_crp_sem_{}_{}.npy'.format(nBeta, nGamma), sem(all_crps, axis=2))
