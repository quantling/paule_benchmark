import pandas as pd
from paule import grad_plan
import torch
import numpy as np

from paule import models

#embedder = models.MelEmbeddingModel_MelSmoothResidualUpsampling(mel_smooth_layers=3).double()
#embedder.load_state_dict(torch.load("/home/tino/Documents/phd/projects/paule/paule/pretrained_models/embedder/model_recorded_embed_model_3_4_180_8192_rmse_lr_00001_400.pt", map_location=torch.device('cpu')))

paule = grad_plan.Paule()

#dat = pd.read_pickle('results20210809_baseline_pred/dat_acoustics.pickle')
dat = pd.read_pickle('results20210805_marser/dat_acoustics.pickle')

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
    mel1 = row.rec_mel
    mel2 = row.seg_mel[:mel1.shape[0],:]
    return np.mean((mel1 - mel2) ** 2)

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

dat.to_pickle('dat_human_recordings_results.pkl')




import matplotlib.pyplot as plt
import ptitprince as pt
import numpy as np 
import pandas as pd
dat = pd.read_pickle('dat_human_recordings_results.pkl')

df = pd.DataFrame({
    'file': np.tile(dat['file'], 6),
    'group': np.repeat(['rec', 'inv', 'semvec', 'semvec acoustic', 'acoustic', 'segment'], 36),
    'rMSE loss between true and resynth semantic vectors': dat['rmse_vec_rec'].to_list() + dat['rmse_vec_inv'].to_list() + dat['rmse_vec_semvec'].to_list() + dat['rmse_vec_acoustic-semvec'].to_list() + dat['rmse_vec_acoustic'].to_list() + dat['rmse_vec_seg'].to_list(),
    'rMSE loss between true and resynth acoustics': dat['rmse_mel_rec'].to_list() + dat['rmse_mel_inv'].to_list() + dat['rmse_mel_semvec'].to_list() + dat['rmse_mel_acoustic-semvec'].to_list() + dat['rmse_mel_acoustic'].to_list() + dat['rmse_mel_seg'].to_list()})


# plots including all data points ----

# both plots in one figure
plt.clf()
f, (ax1, ax2) = plt.subplots(1,2, figsize=(15, 5))
ort = "h"; pal = "Set2"; sigma = .2
pt.RainCloud(x='group', y='rMSE loss between true and resynth acoustics', data=df, palette=pal, bw=sigma, width_viol=.6, ax=ax1, orient=ort, box_showfliers=False)
pt.RainCloud(x='group', y='rMSE loss between true and resynth semantic vectors', data=df, palette=pal, bw=sigma, width_viol=.6, ax=ax2, orient=ort, box_showfliers=False)
#ax2.set_xlim((-0.0001, 0.004))
plt.savefig('boxplots-1-2-noxlim.png')
plt.savefig('boxplots-1-2-noxlim.pdf')



