"""
The human recordings benchmark compares the resynthesis quality for a small set
of words (Problem, Studium, Wissenschaft, Liebe, genau, Klasse for now) between
the recovering of the segment based synthesis and the resynthesis of the human
recording.

"""

import pickle
import os
import time

import pandas as pd
from tqdm import tqdm
from paule import grad_plan
from paule import util
from paule import models
import torch
import numpy as np

tqdm.pandas()

torch.set_num_threads(4)

CONDITION = 'default'
#CONDITION = 'init-seg_baseline'
#CONDITION = 'pred_baseline'
#CONDITION = 'baseline_segment-embedder'
#CONDITION = 'segment-embedder'

if CONDITION == 'default':
    paule = grad_plan.Paule(device=torch.device('cpu'))
elif CONDITION == 'pred_baseline':
    pred_model = models.Non_Linear_Model(mode='pred', on_full_sequence=True).double()
    pred_model.load_state_dict(torch.load("../paule/paule/pretrained_models/baseline_nonlinear/non_linear_pred_model_fullseq_8192_lr_0001_50_00001_50_000001_50_0000001_200.pt", map_location=torch.device('cpu')))
    paule = grad_plan.Paule(pred_model=pred_model, device=torch.device('cpu'))
elif CONDITION == 'init-seg_baseline':
    pred_model = models.Non_Linear_Model(mode='pred', on_full_sequence=True).double()
    pred_model.load_state_dict(torch.load("../paule/paule/pretrained_models/baseline_nonlinear/non_linear_pred_model_fullseq_8192_lr_0001_50_00001_50_000001_50_0000001_200.pt", map_location=torch.device('cpu')))
    paule = grad_plan.Paule(pred_model=pred_model, device=torch.device('cpu'))
elif CONDITION == 'baseline_segment-embedder':
    pred_model = models.Non_Linear_Model(mode='pred', on_full_sequence=True).double()
    pred_model.load_state_dict(torch.load("../paule/paule/pretrained_models/baseline_nonlinear/non_linear_pred_model_fullseq_8192_lr_0001_50_00001_50_000001_50_0000001_200.pt", map_location=torch.device('cpu')))
    embedder = models.MelEmbeddingModel_MelSmoothResidualUpsampling(mel_smooth_layers=3).double()
    embedder.load_state_dict(torch.load("../paule/paule/pretrained_models/embedder/model_synthesized_embed_model_3_4_180_8192_rmse_lr_00001_400.pt", map_location=torch.device('cpu')))
    paule = grad_plan.Paule(pred_model=pred_model, embedder=embedder, device=torch.device('cpu'))
elif CONDITION == 'segment-embedder':
    embedder = models.MelEmbeddingModel_MelSmoothResidualUpsampling(mel_smooth_layers=3).double()
    embedder.load_state_dict(torch.load("../paule/paule/pretrained_models/embedder/model_synthesized_embed_model_3_4_180_8192_rmse_lr_00001_400.pt", map_location=torch.device('cpu')))
    paule = grad_plan.Paule(embedder=embedder, device=torch.device('cpu'))
else:
    raise ValueError('condition is wrong')

RESULT_DIR = f'results_{time.strftime("%Y%m%d")}_{CONDITION}/'

os.mkdir(RESULT_DIR)

dat = pd.read_pickle('data/geco_df_test_final_subset.pkl')

N_INNER = 100
N_OUTER = 40 

dat = dat[-35:]
## for testing run it on the first two
#dat = dat[3:5]
dat = dat.iloc[[90, 100, 120]]
#N_INNER = 20 
#N_OUTER = 2

if 'total_count_train_valid' in dat.columns:
    dat_raw = dat
    dat = dat_raw[['label', 'file_rec',
        'vector',
        'wav_rec', 'sampling_rate_rec', 'melspec_norm_rec', 'melspec_norm_min_rec',
        'cp_norm',
        'wav_syn', 'sampling_rate_syn', 'melspec_norm_syn', 'melspec_norm_min_syn']]
    dat.columns = ['label', 'file', 'vector', 'rec_sig', 'rec_sr',
       'rec_mel', 'rec_mel_min', 'seg_cp', 'seg_sig', 'seg_sr', 'seg_mel',
       'seg_mel_min']
else:
    # add segment synthesis
    dat['seg_sig'] = dat['seg_cp'].progress_apply(lambda cp: util.speak(util.inv_normalize_cp(cp))[0])
    dat['seg_sr'] = 44100
    dat['seg_mel'] = dat['seg_sig'].progress_apply(lambda sig: util.normalize_mel_librosa(util.librosa_melspec(sig, 44100)))  # WARNING this only works for synthesised audio as the sample rate is always 44100
    dat['seg_mel_min'] = dat['seg_mel'].apply(lambda mel: mel.min())
    dat['seg_mel'] = dat['seg_mel'] - dat['seg_mel_min']


# 1. Acoustic resynthesis

# 1.1 only acoustic objective

acoustic_results = list()
planned_cps = list()
inv_cps = list()
prod_signals = list()
prod_mels = list()
for index, row in tqdm(dat.iterrows(), total=dat.shape[0], desc='acoustic_acoustics'):
    target_sig = row['rec_sig'].copy()
    target_sig /= np.max(np.abs(target_sig))
    target_sig *= 0.1
    target_semvec = torch.tensor(row['vector'].copy()).view(1, 300)
    result = paule.plan_resynth(
            target_acoustic=(target_sig, row['rec_sr']),
            target_semvec=target_semvec,
            initialize_from='acoustic',
            objective="acoustic",
            log_semantics=True, n_inner=N_INNER, n_outer=N_OUTER)
    acoustic_results.append(result)
    (planned_cp, inv_cp, target_sig, target_mel, prod_sig, prod_mel, pred_mel, loss_steps,
            loss_mel_steps, loss_semvec_steps, loss_jerk_steps, loss_velocity_steps, loss_prod_steps) = result
    planned_cps.append(planned_cp)
    inv_cps.append(inv_cp)
    prod_signals.append(prod_sig)
    prod_mels.append(prod_mel)
    print(f"last semvec loss: {loss_semvec_steps[-1]:.2e}; last mel loss: {loss_mel_steps[-1]:.2e}\n\n")


dat['inv_cp'] = None
dat['inv_cp'] = inv_cps

dat['planned_cp_acoustic'] = None
dat['prod_sig_acoustic'] = None
dat['prod_mel_acoustic'] = None

dat['planned_cp_acoustic'] = planned_cps
dat['prod_sig_acoustic'] = prod_signals
dat['prod_mel_acoustic'] = prod_mels


# add inv synthesis
dat['inv_sig'] = dat['inv_cp'].progress_apply(lambda cp: util.speak(util.inv_normalize_cp(cp))[0])
dat['inv_sr'] = 44100
dat['inv_mel'] = dat['inv_sig'].progress_apply(lambda sig: util.normalize_mel_librosa(util.librosa_melspec(sig, 44100)))  # WARNING this only works for synthesised audio as the sample rate is always 44100
dat['inv_mel_min'] = dat['inv_mel'].apply(lambda mel: mel.min())
dat['inv_mel'] = dat['inv_mel'] - dat['inv_mel_min']



with open(os.path.join(RESULT_DIR, 'acoustics_acoustic_results.pickle'), 'wb') as pfile:
    pickle.dump(acoustic_results, pfile)


# 1.2 semantic and acoustic objective

acoustic_semvec_results = list()
planned_cps = list()
prod_signals = list()
prod_mels = list()
inv_cps = list()
for index, row in tqdm(dat.iterrows(), total=dat.shape[0], desc='acoustic_acoustic-semvec'):
    target_sig = row['rec_sig'].copy()
    target_sig /= np.max(np.abs(target_sig))
    target_sig *= 0.1
    target_semvec = torch.tensor(row['vector'].copy()).view(1, 300)
    result = paule.plan_resynth(
            target_acoustic=(target_sig, row['rec_sr']),
            target_semvec=target_semvec,
            initialize_from='acoustic',
            objective="acoustic_semvec",
            log_semantics=True, n_inner=N_INNER, n_outer=N_OUTER)
    acoustic_semvec_results.append(result)
    (planned_cp, inv_cp, target_sig, target_mel, prod_sig, prod_mel, pred_mel, loss_steps,
            loss_mel_steps, loss_semvec_steps, loss_jerk_steps, loss_velocity_steps, loss_prod_steps) = result
    planned_cps.append(planned_cp)
    prod_signals.append(prod_sig)
    prod_mels.append(prod_mel)
    inv_cps.append(inv_cp)
    print(f"last semvec loss: {loss_semvec_steps[-1]:.2e}; last mel loss: {loss_mel_steps[-1]:.2e}")

dat['planned_cp_acoustic-semvec'] = None
dat['prod_sig_acoustic-semvec'] = None
dat['prod_mel_acoustic-semvec'] = None

dat['planned_cp_acoustic-semvec'] = planned_cps
dat['prod_sig_acoustic-semvec'] = prod_signals
dat['prod_mel_acoustic-semvec'] = prod_mels

with open(os.path.join(RESULT_DIR, 'acoustics_acoustic-semvec_results.pickle'), 'wb') as pfile:
    pickle.dump(acoustic_semvec_results, pfile)


# 1.3 only semantic objective

semvec_results = list()
planned_cps = list()
prod_signals = list()
prod_mels = list()
inv_cps = list()
for index, row in tqdm(dat.iterrows(), total=dat.shape[0], desc='acoustic_semvec'):
    target_sig = row['rec_sig'].copy()
    target_sig /= np.max(np.abs(target_sig))
    target_sig *= 0.1
    target_semvec = torch.tensor(row['vector'].copy()).view(1, 300)
    result = paule.plan_resynth(
            target_acoustic=(target_sig, row['rec_sr']),
            target_semvec=target_semvec,
            initialize_from='acoustic',
            objective="semvec",
            log_semantics=True, n_inner=N_INNER, n_outer=N_OUTER)
    semvec_results.append(result)
    (planned_cp, inv_cp, target_sig, target_mel, prod_sig, prod_mel, pred_mel, loss_steps,
            loss_mel_steps, loss_semvec_steps, loss_jerk_steps, loss_velocity_steps, loss_prod_steps) = result
    planned_cps.append(planned_cp)
    prod_signals.append(prod_sig)
    prod_mels.append(prod_mel)
    inv_cps.append(inv_cp)
    print(f"last semvec loss: {loss_semvec_steps[-1]:.2e}; last mel loss: {loss_mel_steps[-1]:.2e}")

dat['planned_cp_semvec'] = None
dat['prod_sig_semvec'] = None
dat['prod_mel_semvec'] = None

dat['planned_cp_semvec'] = planned_cps
dat['prod_sig_semvec'] = prod_signals
dat['prod_mel_semvec'] = prod_mels

with open(os.path.join(RESULT_DIR, 'acoustics_semvec_results.pickle'), 'wb') as pfile:
    pickle.dump(semvec_results, pfile)

dat.to_pickle(os.path.join(RESULT_DIR, 'dat_acoustics.pickle'))

# TODO save results and data

# 1.4 Conditions

# segment based synthesis
# inverse prduction
# (after first planning cycle)
# after 40 planning cycles

# 1.5 Metrics

# rmse log-mel-spectrogramm from production compared to target acoustics 
# Wasserstein distance log-mel-spectrogramm from production compared to target acoustics 
# rmse semantic vector from production to target acoustics
# classification rank


# 2. Semvec resyntesis

# 2.1 acoustic semvec objective
vector_results = list()
planned_cps = list()
inv_cps = list()
prod_signals = list()
prod_mels = list()
for index, row in tqdm(dat.iterrows(), total=dat.shape[0], desc='vector_acoustics-semvec'):
    target_seq_length = row['rec_mel'].shape[0]
    target_semvec = torch.tensor(row['vector'].copy()).view(1, 300)
    result = paule.plan_resynth(
            target_semvec=target_semvec,
            target_seq_length=target_seq_length,
            initialize_from='semvec',
            objective="acoustic_semvec",
            log_semantics=True, n_inner=N_INNER, n_outer=N_OUTER, seed=20210811 + index)
    vector_results.append(result)
    (planned_cp, inv_cp, target_sig, target_mel, prod_sig, prod_mel, pred_mel, loss_steps,
            loss_mel_steps, loss_semvec_steps, loss_jerk_steps, loss_velocity_steps, loss_prod_steps) = result
    planned_cps.append(planned_cp)
    inv_cps.append(inv_cp)
    prod_signals.append(prod_sig)
    prod_mels.append(prod_mel)
    print(f"last semvec loss: {loss_semvec_steps[-1]:.2e}; last mel loss: {loss_mel_steps[-1]:.2e}\n\n")


dat['planned_cp_vector_acoustic-semvec'] = None
dat['prod_sig_vector_acoustic-semvec'] = None
dat['prod_mel_vector_acoustic-semvec'] = None

dat['planned_cp_vector_acoustic-semvec'] = planned_cps
dat['prod_sig_vector_acoustic-semvec'] = prod_signals
dat['prod_mel_vector_acoustic-semvec'] = prod_mels

with open(os.path.join(RESULT_DIR, 'vector_acoustic-semvec_results.pickle'), 'wb') as pfile:
    pickle.dump(vector_results, pfile)

dat['gan_cp'] = None
dat['gan_cp'] = inv_cps

# add gan synthesis
dat['gan_sig'] = dat['gan_cp'].progress_apply(lambda cp: util.speak(util.inv_normalize_cp(cp))[0])
dat['gan_sr'] = 44100
dat['gan_mel'] = dat['gan_sig'].progress_apply(lambda sig: util.normalize_mel_librosa(util.librosa_melspec(sig, 44100)))  # NOTE: this only works for synthesised audio as the sample rate is always 44100
dat['gan_mel_min'] = dat['gan_mel'].apply(lambda mel: mel.min())
dat['gan_mel'] = dat['gan_mel'] - dat['gan_mel_min']


# 2.2 acoustic only objective
vector_results = list()
planned_cps = list()
inv_cps = list()
prod_signals = list()
prod_mels = list()
for index, row in tqdm(dat.iterrows(), total=dat.shape[0], desc='vector_acoustics'):
    target_seq_length = row['rec_mel'].shape[0]
    target_semvec = torch.tensor(row['vector'].copy()).view(1, 300)
    result = paule.plan_resynth(
            target_semvec=target_semvec,
            target_seq_length=target_seq_length,
            initialize_from='semvec',
            objective="acoustic",
            log_semantics=True, n_inner=N_INNER, n_outer=N_OUTER, seed=20210811 + index)
    vector_results.append(result)
    (planned_cp, inv_cp, target_sig, target_mel, prod_sig, prod_mel, pred_mel, loss_steps,
            loss_mel_steps, loss_semvec_steps, loss_jerk_steps, loss_velocity_steps, loss_prod_steps) = result
    planned_cps.append(planned_cp)
    inv_cps.append(inv_cp)
    prod_signals.append(prod_sig)
    prod_mels.append(prod_mel)
    print(f"last semvec loss: {loss_semvec_steps[-1]:.2e}; last mel loss: {loss_mel_steps[-1]:.2e}\n\n")


dat['planned_cp_vector_acoustic'] = None
dat['prod_sig_vector_acoustic'] = None
dat['prod_mel_vector_acoustic'] = None

dat['planned_cp_vector_acoustic'] = planned_cps
dat['prod_sig_vector_acoustic'] = prod_signals
dat['prod_mel_vector_acoustic'] = prod_mels

with open(os.path.join(RESULT_DIR, 'vector_acoustic_results.pickle'), 'wb') as pfile:
    pickle.dump(vector_results, pfile)


# 2.3 semvec only objective
vector_results = list()
planned_cps = list()
inv_cps = list()
prod_signals = list()
prod_mels = list()
for index, row in tqdm(dat.iterrows(), total=dat.shape[0], desc='vector_semvec'):
    target_seq_length = row['rec_mel'].shape[0]
    target_semvec = torch.tensor(row['vector'].copy()).view(1, 300)
    result = paule.plan_resynth(
            target_semvec=target_semvec,
            target_seq_length=target_seq_length,
            initialize_from='semvec',
            objective="semvec",
            log_semantics=True, n_inner=N_INNER, n_outer=N_OUTER, seed=20210811 + index)
    vector_results.append(result)
    (planned_cp, inv_cp, target_sig, target_mel, prod_sig, prod_mel, pred_mel, loss_steps,
            loss_mel_steps, loss_semvec_steps, loss_jerk_steps, loss_velocity_steps, loss_prod_steps) = result
    planned_cps.append(planned_cp)
    inv_cps.append(inv_cp)
    prod_signals.append(prod_sig)
    prod_mels.append(prod_mel)
    print(f"last semvec loss: {loss_semvec_steps[-1]:.2e}; last mel loss: {loss_mel_steps[-1]:.2e}\n\n")


dat['planned_cp_vector_semvec'] = None
dat['prod_sig_vector_semvec'] = None
dat['prod_mel_vector_semvec'] = None

dat['planned_cp_vector_semvec'] = planned_cps
dat['prod_sig_vector_semvec'] = prod_signals
dat['prod_mel_vector_semvec'] = prod_mels

with open(os.path.join(RESULT_DIR, 'vector_semvec.pickle'), 'wb') as pfile:
    pickle.dump(vector_results, pfile)

dat.to_pickle(os.path.join(RESULT_DIR, 'dat_vector.pickle'))


# 3 Initialise with segment based cps


# 3.1 only acoustic objective

acoustic_results = list()
planned_cps = list()
inv_cps = list()
prod_signals = list()
prod_mels = list()
for index, row in tqdm(dat.iterrows(), total=dat.shape[0], desc='init-seg_acoustic'):
    target_sig = row['rec_sig'].copy()
    target_sig /= np.max(np.abs(target_sig))
    target_sig *= 0.1
    target_semvec = torch.tensor(row['vector'].copy()).view(1, 300)
    inv_cp = row['seg_cp'].copy()
    seg_padding = int((row['seg_cp'].shape[0] - row['rec_mel'].shape[0] * 2) / 2)
    inv_cp = inv_cp[seg_padding:-seg_padding]
    result = paule.plan_resynth(
            target_acoustic=(target_sig, row['rec_sr']),
            target_semvec=target_semvec,
            inv_cp=inv_cp,
            initialize_from=None,
            objective="acoustic",
            log_semantics=True, n_inner=N_INNER, n_outer=N_OUTER)
    acoustic_results.append(result)
    (planned_cp, inv_cp, target_sig, target_mel, prod_sig, prod_mel, pred_mel, loss_steps,
            loss_mel_steps, loss_semvec_steps, loss_jerk_steps, loss_velocity_steps, loss_prod_steps) = result
    planned_cps.append(planned_cp)
    inv_cps.append(inv_cp)
    prod_signals.append(prod_sig)
    prod_mels.append(prod_mel)
    print(f"last semvec loss: {loss_semvec_steps[-1]:.2e}; last mel loss: {loss_mel_steps[-1]:.2e}\n\n")


dat['inv_cp'] = None
dat['inv_cp'] = inv_cps

dat['planned_cp_acoustic'] = None
dat['prod_sig_acoustic'] = None
dat['prod_mel_acoustic'] = None

dat['planned_cp_acoustic'] = planned_cps
dat['prod_sig_acoustic'] = prod_signals
dat['prod_mel_acoustic'] = prod_mels


# add inv synthesis
dat['inv_sig'] = dat['inv_cp'].progress_apply(lambda cp: util.speak(util.inv_normalize_cp(cp))[0])
dat['inv_sr'] = 44100
dat['inv_mel'] = dat['inv_sig'].progress_apply(lambda sig: util.normalize_mel_librosa(util.librosa_melspec(sig, 44100)))  # WARNING this only works for synthesised audio as the sample rate is always 44100
dat['inv_mel_min'] = dat['inv_mel'].apply(lambda mel: mel.min())
dat['inv_mel'] = dat['inv_mel'] - dat['inv_mel_min']



with open(os.path.join(RESULT_DIR, 'init-seg_acoustic_results.pickle'), 'wb') as pfile:
    pickle.dump(acoustic_results, pfile)


# 3.2 semantic and acoustic objective

acoustic_semvec_results = list()
planned_cps = list()
prod_signals = list()
prod_mels = list()
inv_cps = list()
for index, row in tqdm(dat.iterrows(), total=dat.shape[0], desc='init-seg_acoustic-semvec'):
    target_sig = row['rec_sig'].copy()
    target_sig /= np.max(np.abs(target_sig))
    target_sig *= 0.1
    target_semvec = torch.tensor(row['vector'].copy()).view(1, 300)
    inv_cp = row['seg_cp'].copy()
    seg_padding = int((row['seg_cp'].shape[0] - row['rec_mel'].shape[0] * 2) / 2)
    inv_cp = inv_cp[seg_padding:-seg_padding]
    result = paule.plan_resynth(
            target_acoustic=(target_sig, row['rec_sr']),
            target_semvec=target_semvec,
            inv_cp=inv_cp,
            initialize_from=None,
            objective="acoustic_semvec",
            log_semantics=True, n_inner=N_INNER, n_outer=N_OUTER)
    acoustic_semvec_results.append(result)
    (planned_cp, inv_cp, target_sig, target_mel, prod_sig, prod_mel, pred_mel, loss_steps,
            loss_mel_steps, loss_semvec_steps, loss_jerk_steps, loss_velocity_steps, loss_prod_steps) = result
    planned_cps.append(planned_cp)
    prod_signals.append(prod_sig)
    prod_mels.append(prod_mel)
    inv_cps.append(inv_cp)
    print(f"last semvec loss: {loss_semvec_steps[-1]:.2e}; last mel loss: {loss_mel_steps[-1]:.2e}")

dat['planned_cp_acoustic-semvec'] = None
dat['prod_sig_acoustic-semvec'] = None
dat['prod_mel_acoustic-semvec'] = None

dat['planned_cp_acoustic-semvec'] = planned_cps
dat['prod_sig_acoustic-semvec'] = prod_signals
dat['prod_mel_acoustic-semvec'] = prod_mels

with open(os.path.join(RESULT_DIR, 'init-seg_acoustic-semvec_results.pickle'), 'wb') as pfile:
    pickle.dump(acoustic_semvec_results, pfile)


# 3.3 only semantic objective

semvec_results = list()
planned_cps = list()
prod_signals = list()
prod_mels = list()
inv_cps = list()
for index, row in tqdm(dat.iterrows(), total=dat.shape[0], desc='init-seg_semvec'):
    target_sig = row['rec_sig'].copy()
    target_sig /= np.max(np.abs(target_sig))
    target_sig *= 0.1
    target_semvec = torch.tensor(row['vector'].copy()).view(1, 300)
    inv_cp = row['seg_cp'].copy()
    seg_padding = int((row['seg_cp'].shape[0] - row['rec_mel'].shape[0] * 2) / 2)
    inv_cp = inv_cp[seg_padding:-seg_padding]
    result = paule.plan_resynth(
            target_acoustic=(target_sig, row['rec_sr']),
            target_semvec=target_semvec,
            inv_cp=inv_cp,
            initialize_from=None,
            objective="semvec",
            log_semantics=True, n_inner=N_INNER, n_outer=N_OUTER)
    semvec_results.append(result)
    (planned_cp, inv_cp, target_sig, target_mel, prod_sig, prod_mel, pred_mel, loss_steps,
            loss_mel_steps, loss_semvec_steps, loss_jerk_steps, loss_velocity_steps, loss_prod_steps) = result
    planned_cps.append(planned_cp)
    prod_signals.append(prod_sig)
    prod_mels.append(prod_mel)
    inv_cps.append(inv_cp)
    print(f"last semvec loss: {loss_semvec_steps[-1]:.2e}; last mel loss: {loss_mel_steps[-1]:.2e}")

dat['planned_cp_semvec'] = None
dat['prod_sig_semvec'] = None
dat['prod_mel_semvec'] = None

dat['planned_cp_semvec'] = planned_cps
dat['prod_sig_semvec'] = prod_signals
dat['prod_mel_semvec'] = prod_mels

with open(os.path.join(RESULT_DIR, 'init-seg_semvec_results.pickle'), 'wb') as pfile:
    pickle.dump(semvec_results, pfile)

dat.to_pickle(os.path.join(RESULT_DIR, 'dat_init-seg.pickle'))


# 4. "Wer ist eigentlich Paule?"


