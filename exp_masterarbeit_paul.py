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

torch.set_num_threads(2)

embedder = models.MelEmbeddingModel_MelSmoothResidualUpsampling(mel_smooth_layers=3).double()
embedder.load_state_dict(torch.load("../paule/paule/pretrained_models/embedder/model_synthesized_embed_model_3_4_180_8192_rmse_lr_00001_400.pt", map_location=torch.device('cpu')))

pred_model = models.Non_Linear_Model(mode='pred', on_full_sequence=True).double()
pred_model.load_state_dict(torch.load("../paule/paule/pretrained_models/baseline_nonlinear/non_linear_pred_model_fullseq_8192_lr_0001_50_00001_50_000001_50_0000001_200.pt", map_location=torch.device('cpu')))

paule = grad_plan.Paule(pred_model=pred_model, embedder=embedder, device=torch.device('cpu'))

RESULT_DIR = f'results_{time.strftime("%Y%m%d_%H%M")}_masterarbeit_paul/'

os.mkdir(RESULT_DIR)

dat = pd.read_pickle('data/geco_df_test_final_subset.pkl')
dat = dat.iloc[-35:]
N_INNER = 100
N_OUTER = 20

#dat = dat.iloc[-3:]
#N_INNER = 10
#N_OUTER = 2

dat_raw = dat
dat = dat_raw[['label', 'file_rec',
    'vector',
    'wav_rec', 'sampling_rate_rec', 'melspec_norm_rec', 'melspec_norm_min_rec',
    'cp_norm',
    'wav_syn', 'sampling_rate_syn', 'melspec_norm_syn', 'melspec_norm_min_syn']]
dat.columns = ['label', 'file', 'vector', 'rec_sig', 'rec_sr',
   'rec_mel', 'rec_mel_min', 'seg_cp', 'seg_sig', 'seg_sr', 'seg_mel',
   'seg_mel_min']

if dat.rec_sig.apply(max).max() > 1.0:
    dat.rec_sig = dat.rec_sig / 2 ** 15 * 6

INITIALIZES = ('acoustic', 'semvec', 'init-seg')
OBJECTIVES = ('acoustic', 'acoustic_semvec', 'semvec')

for init in INITIALIZES:
    for objective in OBJECTIVES:
        results = list()
        planned_cps = list()
        prod_signals = list()
        prod_mels = list()
        inv_cps = list()
        for index, row in tqdm(dat.iterrows(), total=dat.shape[0], desc=f'{init}_{objective}'):
            target_sig = row['rec_sig'].copy()
            target_sig /= np.max(np.abs(target_sig))
            target_sig *= 0.1
            target_semvec = torch.tensor(row['vector'].copy()).view(1, 300)
            if init == 'init-seg':
                initialize_from = None
                inv_cp = row['seg_cp'].copy()
                seg_padding = int((row['seg_cp'].shape[0] - row['rec_mel'].shape[0] * 2) / 2)
                inv_cp = inv_cp[seg_padding:-seg_padding]
            else:
                initialize_from = init
                inv_cp = None
            result = paule.plan_resynth(
                    target_acoustic=(target_sig, row['rec_sr']),
                    target_semvec=target_semvec,
                    inv_cp=inv_cp,
                    initialize_from=initialize_from,
                    objective=objective,
                    log_semantics=False, n_inner=N_INNER, n_outer=N_OUTER,
                    verbose=False)
            results.append(result)
            (planned_cp, inv_cp, target_sig, target_mel, prod_sig, prod_mel, pred_mel, loss_steps,
                    loss_mel_steps, loss_semvec_steps, loss_jerk_steps, loss_velocity_steps, loss_prod_steps) = result
            planned_cps.append(planned_cp)
            prod_signals.append(prod_sig)
            prod_mels.append(prod_mel)
            inv_cps.append(inv_cp)
            if loss_semvec_steps:
                print(f"last semvec loss: {loss_semvec_steps[-1]:.2e}; last mel loss: {loss_mel_steps[-1]:.2e}")
            else:
                print(f"last mel loss: {loss_mel_steps[-1]:.2e}")

        dat[f'planned_cp_{objective}'] = None
        dat[f'prod_sig_{objective}'] = None
        dat[f'prod_mel_{objective}'] = None

        dat[f'planned_cp_{objective}'] = planned_cps
        dat[f'prod_sig_{objective}'] = prod_signals
        dat[f'prod_mel_{objective}'] = prod_mels

        with open(os.path.join(RESULT_DIR, f'{init}_{objective}_results.pickle'), 'wb') as pfile:
            pickle.dump(results, pfile)

    # add inv synthesis
    dat['inv_cp'] = None
    dat['inv_cp'] = inv_cps
    dat['inv_sig'] = dat['inv_cp'].progress_apply(lambda cp: util.speak(util.inv_normalize_cp(cp))[0])
    dat['inv_sr'] = 44100
    dat['inv_mel'] = dat['inv_sig'].progress_apply(lambda sig: util.normalize_mel_librosa(util.librosa_melspec(sig, 44100)))  # WARNING this only works for synthesised audio as the sample rate is always 44100
    dat['inv_mel_min'] = dat['inv_mel'].apply(lambda mel: mel.min())
    dat['inv_mel'] = dat['inv_mel'] - dat['inv_mel_min']


    # SAVE
    dat.to_pickle(os.path.join(RESULT_DIR, f'dat_{init}.pickle'))

