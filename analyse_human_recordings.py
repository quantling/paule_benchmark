import pandas as pd
from paule import grad_plan
import torch
import numpy as np

from paule import models

CONDITION = 'acoustic-default'
#CONDITION = 'acoustic-baseline'
#CONDITION = 'acoustic-init-seg'
#CONDITION = 'semantic-default'

#embedder = models.MelEmbeddingModel_MelSmoothResidualUpsampling(mel_smooth_layers=3).double()
#embedder.load_state_dict(torch.load("/home/tino/Documents/phd/projects/paule/paule/pretrained_models/embedder/model_recorded_embed_model_3_4_180_8192_rmse_lr_00001_400.pt", map_location=torch.device('cpu')))

paule = grad_plan.Paule()


if CONDITION == 'acoustic-baseline':
    dat = pd.read_pickle('results20210809_baseline_pred/dat_acoustics.pickle')
elif CONDITION == 'acoustic-default':
    dat = pd.read_pickle('results20210809/dat_acoustics.pickle')
elif CONDITION == 'acoustic-init-seg':
    pass
elif CONDITION == 'semantic-default':
    pass
else:
    raise ValueError(f"unkown condition {CONDITION}")

# rename legacy columns
if 'prod_sig_nosemvec' in dat.columns:
    dat['prod_sig_acoustic'] = dat['prod_sig_nosemvec']
    dat['prod_mel_acoustic'] = dat['prod_mel_nosemvec']
    dat['planned_cp_acoustic'] = dat['planned_cp_nosemvec']

dat['rec_vec'] = None
dat['seg_vec'] = None
dat['inv_vec'] = None
dat['prod_vec_semvec'] = None
dat['prod_vec_acoustic'] = None
dat['prod_vec_acoustic-semvec'] = None
dat['rmse_mel_rec'] = None
dat['rmse_mel_seg'] = None
dat['rmse_mel_inv'] = None
dat['rmse_mel_semvec'] = None
dat['rmse_mel_acoustic'] = None
dat['rmse_mel_acoustic-semvec'] = None
dat['rmse_vec_rec'] = None
dat['rmse_vec_seg'] = None
dat['rmse_vec_inv'] = None
dat['rmse_vec_semvec'] = None
dat['rmse_vec_acoustic'] = None
dat['rmse_vec_acoustic-semvec'] = None


def get_semvec(mel):
    mel = mel.copy()
    mel.shape = (1,) + mel.shape
    mel = torch.from_numpy(mel)
    seq_length = mel.shape[1]
    with torch.no_grad():
        semvec = paule.embedder(mel.to(paule.device), (torch.tensor(seq_length),))
    semvec = semvec.cpu().numpy()
    semvec.shape = (semvec.shape[1],)
    return semvec


# rec_vec
# seg_vec
# prod_vec_semvec
# prod_vec_acoustic

dat['rec_vec'] = dat.rec_mel.apply(get_semvec)
dat['seg_vec'] = dat.seg_mel.apply(get_semvec)
dat['inv_vec'] = dat.inv_mel.apply(get_semvec)
dat['prod_vec_semvec'] = dat.prod_mel_semvec.apply(get_semvec)
dat['prod_vec_acoustic'] = dat.prod_mel_acoustic.apply(get_semvec)
dat['prod_vec_acoustic-semvec'] = dat['prod_mel_acoustic-semvec'].apply(get_semvec)


def rmse(array1, array2):
    #  eps = 1e-6
    return np.sqrt(np.mean((array2 - array1) ** 2) + 1e-6)

# rmse_mel_semvec
# rmse_mel_acoustic
# rmse_mel_seg
#dat['rmse_mel_seg'] = dat.apply(lambda row: rmse(row.rec_mel, row.seg_mel), axis=1)
dat['rmse_mel_rec'] = dat.apply(lambda row: rmse(row.rec_mel, row.rec_mel), axis=1)
#dat.rmse_mel_rec += np.random.random(36) * 0.001
dat['rmse_mel_inv'] = dat.apply(lambda row: rmse(row.rec_mel, row.inv_mel), axis=1)
dat['rmse_mel_semvec'] = dat.apply(lambda row: rmse(row.rec_mel, row.prod_mel_semvec), axis=1)
dat['rmse_mel_acoustic'] = dat.apply(lambda row: rmse(row.rec_mel, row.prod_mel_acoustic), axis=1)
dat['rmse_mel_acoustic-semvec'] = dat.apply(lambda row: rmse(row.rec_mel, row['prod_mel_acoustic-semvec']), axis=1)

def rmse_mel_seg(row):
    rec_mel = row.rec_mel
    seg_mel = row.seg_mel
    half_shift = int((seg_mel.shape[0] - rec_mel.shape[0]) / 2)
    seg_mel = seg_mel[half_shift:-half_shift]
    if rec_mel.shape[0] < seg_mel.shape[0]:
        seg_mel = seg_mel[:rec_mel.shape[0]]
    return rmse(seg_mel, rec_mel)

dat['rmse_mel_seg'] = dat.apply(rmse_mel_seg, axis=1)

# rmse_vec_semvec
# rmse_vec_acoustic
# rmse_vec_seg0

dat['rmse_vec_rec'] = dat.apply(lambda row: rmse(row.vector, row.rec_vec), axis=1)
dat['rmse_vec_inv'] = dat.apply(lambda row: rmse(row.vector, row.inv_vec), axis=1)
dat['rmse_vec_semvec'] = dat.apply(lambda row: rmse(row.vector, row.prod_vec_semvec), axis=1)
dat['rmse_vec_acoustic'] = dat.apply(lambda row: rmse(row.vector, row.prod_vec_acoustic), axis=1)
dat['rmse_vec_acoustic-semvec'] = dat.apply(lambda row: rmse(row.vector, row['prod_vec_acoustic-semvec']), axis=1)
dat['rmse_vec_seg'] = dat.apply(lambda row: rmse(row.vector, row.seg_vec), axis=1)




dat.to_pickle(f'data/dat_human_recordings_metrics_{CONDITION}.pickle')




import matplotlib.pyplot as plt
import ptitprince as pt
import numpy as np 
import pandas as pd
dat = pd.read_pickle(f'data/dat_human_recordings_metrics_{CONDITION}.pickle')

df = pd.DataFrame({
    'file': np.tile(dat['file'], 6),
    'group': np.repeat(['rec', 'inv', 'semvec', 'acoustic semvec', 'acoustic', 'segment'], 36),
    'rMSE loss between true and resynth semantic vectors': dat['rmse_vec_rec'].to_list() + dat['rmse_vec_inv'].to_list() + dat['rmse_vec_semvec'].to_list() + dat['rmse_vec_acoustic-semvec'].to_list() + dat['rmse_vec_acoustic'].to_list() + dat['rmse_vec_seg'].to_list(),
    'rMSE loss between true and resynth acoustics': dat['rmse_mel_rec'].to_list() + dat['rmse_mel_inv'].to_list() + dat['rmse_mel_semvec'].to_list() + dat['rmse_mel_acoustic-semvec'].to_list() + dat['rmse_mel_acoustic'].to_list() + dat['rmse_mel_seg'].to_list()})


# plots including all data points ----

# both plots in one figure
plt.clf()
f, (ax1, ax2) = plt.subplots(1,2, figsize=(15, 5))
ort = "h"; pal = "Set2"; sigma = .2
pt.RainCloud(x='group', y='rMSE loss between true and resynth acoustics', data=df, palette=pal, bw=sigma, width_viol=.6, ax=ax1, orient=ort, box_showfliers=False)
pt.RainCloud(x='group', y='rMSE loss between true and resynth semantic vectors', data=df, palette=pal, bw=sigma, width_viol=.6, ax=ax2, orient=ort, box_showfliers=False)
ax1.set_xlim((-0.0001, 0.5))
ax2.set_xlim((-0.0001, 0.07))
plt.savefig(f'plots/boxplots-1-2_{CONDITION}.png')
plt.savefig(f'plots/boxplots-1-2_{CONDITION}.pdf')


# play all words

import sounddevice as sd
import time

print(f"condition: {CONDITION}")
for index, row in dat.iterrows():
    print(f"{row['label']} (inv, planned acoustic-semvec, rec, seg)")
    sd.play(row['inv_sig'] * 4, row['inv_sr'], blocking=True)
    time.sleep(0.1)
    sd.play(row['prod_sig_acoustic-semvec'] * 4, 44100, blocking=True)
    time.sleep(0.1)
    sd.play(row['rec_sig'], row['rec_sr'], blocking=True)
    time.sleep(0.1)
    sd.play(row['seg_sig'], row['seg_sr'], blocking=True)
    time.sleep(0.5)


