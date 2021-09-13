import pandas as pd
import torch
import numpy as np
import scipy as sp
import scipy.stats

from paule import models

#CONDITION = 'recording-embedder'
CONDITION = 'segment-embedder'

OBJECTIVES = ('acoustic', 'semvec', 'init-seg')

LABEL_VECTORS = pd.read_pickle('data/label_vectors_vectors_checked.pkl')

# load embedder for evaluation
embedder = models.MelEmbeddingModel_MelSmoothResidualUpsampling(mel_smooth_layers=3).double()
#embedder.load_state_dict(torch.load("../paule/paule/pretrained_models/embedder/model_recorded_embed_model_3_4_180_8192_rmse_lr_00001_400.pt", map_location=torch.device('cpu')))
embedder.load_state_dict(torch.load("../paule/paule/pretrained_models/embedder/model_synthesized_embed_model_3_4_180_8192_rmse_lr_00001_400.pt", map_location=torch.device('cpu')))


# load and concatenate data
dats = dict()
for objective in OBJECTIVES:
    dat1 = pd.read_pickle(f'results_20210911_0838_masterarbeit_paul/dat_{objective}.pickle')
    dat2 = pd.read_pickle(f'results_20210911_1012_masterarbeit_paul/dat_{objective}.pickle')
    dat3 = pd.read_pickle(f'results_20210910_2201_masterarbeit_paul/dat_{objective}.pickle')
    dats[objective] = pd.concat((dat1, dat2, dat3))
del dat1, dat2, dat3


def mel_wasserstein_distance(mel1, mel2):
    """
    perform the 1d Wasserstein distance function over mel bands, time points and energy

    :param mel1: np.array
        log-mel spectrogram (seq_length, mel channels)
    :param: mel1: np.array
        log-mel spectrogram (seq_length, mel channels)

    :return mean_time_dist, meean_mel_dist, energy_dist: np.float, np.float, np.float
        average 1d distance over time, mel_channel and energy
    """
    assert mel1.shape == mel2.shape

    if isinstance(mel1, np.ndarray):
        mel1 = torch.from_numpy(mel1)
    if isinstance(mel2, np.ndarray):
        mel2 = torch.from_numpy(mel2)

    time_dist = []
    mel_dist = []

    for time_point in range(mel1.shape[0]):
        time_dist.append(sp.stats.wasserstein_distance(mel1[time_point],mel2[time_point]))
    for mel_channel in range(mel1.shape[1]):
        mel_dist.append(sp.stats.wasserstein_distance(mel1[:,mel_channel], mel2[:,mel_channel]))

    energy1 = torch.mean(mel1, axis=1)
    energy2 = torch.mean(mel2, axis=1)

    energy_dist = sp.stats.wasserstein_distance(energy1, energy2)

    #return np.mean(time_dist), np.mean(mel_dist), energy_dist
    return np.mean((np.mean(time_dist), np.mean(mel_dist), energy_dist))


def get_semvec(mel):
    mel = mel.copy()
    mel.shape = (1,) + mel.shape
    mel = torch.from_numpy(mel)
    seq_length = mel.shape[1]
    with torch.no_grad():
        semvec = embedder(mel, (torch.tensor(seq_length),))
    semvec = semvec.cpu().numpy()
    semvec.shape = (semvec.shape[1],)
    return semvec


def rmse(array1, array2):
    #  eps = 1e-6
    return np.sqrt(np.mean((array2 - array1) ** 2) + 1e-6)


def rmse_mel_seg(row):
    rec_mel = row.rec_mel
    seg_mel = row.seg_mel
    half_shift = int((seg_mel.shape[0] - rec_mel.shape[0]) / 2)
    seg_mel = seg_mel[half_shift:-half_shift]
    if rec_mel.shape[0] < seg_mel.shape[0]:
        seg_mel = seg_mel[:rec_mel.shape[0]]
    return rmse(seg_mel, rec_mel)


def mel_wasserstein_distance_mel_seg(row):
    rec_mel = row.rec_mel
    seg_mel = row.seg_mel
    half_shift = int((seg_mel.shape[0] - rec_mel.shape[0]) / 2)
    seg_mel = seg_mel[half_shift:-half_shift]
    if rec_mel.shape[0] < seg_mel.shape[0]:
        seg_mel = seg_mel[:rec_mel.shape[0]]
    return mel_wasserstein_distance(seg_mel, rec_mel)


def rank(label, vector):
    corr = np.array([np.correlate(vector, vec2) for vec2 in LABEL_VECTORS['vector']])
    return int(sp.stats.rankdata(-corr, method='min')[LABEL_VECTORS.label == 'Problem'])


for dat in dats.values():
    dat['rec_vec'] = None
    dat['seg_vec'] = None
    dat['inv_vec'] = None
    dat['prod_vec_acoustic'] = None
    dat['prod_vec_acoustic_semvec'] = None
    dat['prod_vec_semvec'] = None
    dat['rmse_mel_rec'] = None
    dat['rmse_mel_seg'] = None
    dat['rmse_mel_inv'] = None
    dat['rmse_mel_acoustic'] = None
    dat['rmse_mel_acoustic_semvec'] = None
    dat['rmse_mel_semvec'] = None
    dat['rmse_vec_rec'] = None
    dat['rmse_vec_seg'] = None
    dat['rmse_vec_inv'] = None
    dat['rmse_vec_acoustic'] = None
    dat['rmse_vec_acoustic_semvec'] = None
    dat['rmse_vec_semvec'] = None
    dat['wasser_mel_rec'] = None
    dat['wasser_mel_seg'] = None
    dat['wasser_mel_inv'] = None
    dat['wasser_mel_acoustic'] = None
    dat['wasser_mel_acoustic_semvec'] = None
    dat['wasser_mel_semvec'] = None

    # vec (semantic vectors)
    dat['rec_vec'] = dat.rec_mel.apply(get_semvec)
    dat['seg_vec'] = dat.seg_mel.apply(get_semvec)
    dat['inv_vec'] = dat.inv_mel.apply(get_semvec)
    dat['prod_vec_acoustic'] = dat.prod_mel_acoustic.apply(get_semvec)
    dat['prod_vec_acoustic_semvec'] = dat.prod_mel_acoustic_semvec.apply(get_semvec)
    dat['prod_vec_semvec'] = dat.prod_mel_semvec.apply(get_semvec)

    # rmse mel
    dat['rmse_mel_rec'] = dat.apply(lambda row: rmse(row.rec_mel, row.rec_mel), axis=1)
    dat['rmse_mel_inv'] = dat.apply(lambda row: rmse(row.rec_mel, row.inv_mel), axis=1)
    dat['rmse_mel_acoustic'] = dat.apply(lambda row: rmse(row.rec_mel, row.prod_mel_acoustic), axis=1)
    dat['rmse_mel_acoustic_semvec'] = dat.apply(lambda row: rmse(row.rec_mel, row.prod_mel_acoustic_semvec), axis=1)
    dat['rmse_mel_semvec'] = dat.apply(lambda row: rmse(row.rec_mel, row.prod_mel_semvec), axis=1)
    dat['rmse_mel_seg'] = dat.apply(rmse_mel_seg, axis=1)

    # rmse vec
    dat['rmse_vec_rec'] = dat.apply(lambda row: rmse(row.vector, row.rec_vec), axis=1)
    dat['rmse_vec_inv'] = dat.apply(lambda row: rmse(row.vector, row.inv_vec), axis=1)
    dat['rmse_vec_acoustic'] = dat.apply(lambda row: rmse(row.vector, row.prod_vec_acoustic), axis=1)
    dat['rmse_vec_acoustic_semvec'] = dat.apply(lambda row: rmse(row.vector, row.prod_vec_acoustic_semvec), axis=1)
    dat['rmse_vec_semvec'] = dat.apply(lambda row: rmse(row.vector, row.prod_vec_semvec), axis=1)
    dat['rmse_vec_seg'] = dat.apply(lambda row: rmse(row.vector, row.seg_vec), axis=1)

    # wasserstein mel
    dat['wasser_mel_rec'] = dat.apply(lambda row: mel_wasserstein_distance(row.rec_mel, row.rec_mel), axis=1)
    dat['wasser_mel_inv'] = dat.apply(lambda row: mel_wasserstein_distance(row.rec_mel, row.inv_mel), axis=1)
    dat['wasser_mel_acoustic'] = dat.apply(lambda row: mel_wasserstein_distance(row.rec_mel, row.prod_mel_acoustic), axis=1)
    dat['wasser_mel_acoustic_semvec'] = dat.apply(lambda row: mel_wasserstein_distance(row.rec_mel, row.prod_mel_acoustic_semvec), axis=1)
    dat['wasser_mel_semvec'] = dat.apply(lambda row: mel_wasserstein_distance(row.rec_mel, row.prod_mel_semvec), axis=1)
    dat['wasser_mel_seg'] = dat.apply(mel_wasserstein_distance_mel_seg, axis=1)

    # correlations and rank of target word
    dat['corr_vec_rec'] = dat.apply(lambda row: 1 - float(np.correlate(row.vector, row.rec_vec)), axis=1)
    dat['corr_vec_inv'] = dat.apply(lambda row: 1 - float(np.correlate(row.vector, row.inv_vec)), axis=1)
    dat['corr_vec_acoustic'] = dat.apply(lambda row: 1 - float(np.correlate(row.vector, row.prod_vec_acoustic)), axis=1)
    dat['corr_vec_acoustic_semvec'] = dat.apply(lambda row: 1 - float(np.correlate(row.vector, row.prod_vec_acoustic_semvec)), axis=1)
    dat['corr_vec_semvec'] = dat.apply(lambda row: 1 - float(np.correlate(row.vector, row.prod_vec_semvec)), axis=1)
    dat['corr_vec_seg'] = dat.apply(lambda row: 1 - float(np.correlate(row.vector, row.seg_vec)), axis=1)

    dat['rank_vec_rec'] = dat.apply(lambda row: rank(row.label, row.rec_vec), axis=1)
    dat['rank_vec_inv'] = dat.apply(lambda row: rank(row.label, row.inv_vec), axis=1)
    dat['rank_vec_acoustic'] = dat.apply(lambda row: rank(row.label, row['prod_vec_acoustic']), axis=1)
    dat['rank_vec_acoustic_semvec'] = dat.apply(lambda row: rank(row.label, row['prod_vec_acoustic_semvec']), axis=1)
    dat['rank_vec_semvec'] = dat.apply(lambda row: rank(row.label, row['prod_vec_semvec']), axis=1)
    dat['rank_vec_seg'] = dat.apply(lambda row: rank(row.label, row.seg_vec), axis=1)

#dat.to_pickle(f'data/exp_seg_vs_record_metrics_{CONDITION}_{OBJECTIVE}.pickle')




import matplotlib.pyplot as plt
import ptitprince as pt
import numpy as np 
import pandas as pd

#dat = pd.read_pickle(f'data/exp_seg_vs_record_metrics_{CONDITION}_{OBJECTIVE}.pickle')

for objective, dat in dats.items():

    df = pd.DataFrame({
        'file': np.tile(dat['file'], 6),
        'group': np.repeat(['rec', 'inv', 'semvec', 'acoustic semvec', 'acoustic', 'segment'], dat.shape[0]),
        'rMSE loss between true and resynth semantic vectors': dat['rmse_vec_rec'].to_list() + dat['rmse_vec_inv'].to_list() + dat['rmse_vec_semvec'].to_list() + dat['rmse_vec_acoustic_semvec'].to_list() + dat['rmse_vec_acoustic'].to_list() + dat['rmse_vec_seg'].to_list(),
        '1 - corr between true and resynth semantic vector': dat['corr_vec_rec'].to_list() + dat['corr_vec_inv'].to_list() + dat['corr_vec_semvec'].to_list() + dat['corr_vec_acoustic_semvec'].to_list() + dat['corr_vec_acoustic'].to_list() + dat['corr_vec_seg'].to_list(),
        'rank of resynth semantic vector': dat['rank_vec_rec'].to_list() + dat['rank_vec_inv'].to_list() + dat['rank_vec_semvec'].to_list() + dat['rank_vec_acoustic_semvec'].to_list() + dat['rank_vec_acoustic'].to_list() + dat['rank_vec_seg'].to_list(),
        'rMSE loss between true and resynth acoustics': dat['rmse_mel_rec'].to_list() + dat['rmse_mel_inv'].to_list() + dat['rmse_mel_semvec'].to_list() + dat['rmse_mel_acoustic_semvec'].to_list() + dat['rmse_mel_acoustic'].to_list() + dat['rmse_mel_seg'].to_list(),
        'Wasserstein Distance between true and resynth acoustics': dat['wasser_mel_rec'].to_list() + dat['wasser_mel_inv'].to_list() + dat['wasser_mel_semvec'].to_list() + dat['wasser_mel_acoustic_semvec'].to_list() + dat['wasser_mel_acoustic'].to_list() + dat['wasser_mel_seg'].to_list()})

    # plots including all data points ----

    # both plots in one figure
    plt.clf()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ort = "h"; pal = "Set2"; sigma = .2
    pt.RainCloud(x='group', y='rMSE loss between true and resynth acoustics', data=df, palette=pal, bw=sigma, width_viol=.6, ax=ax1, orient=ort, box_showfliers=False)
    pt.RainCloud(x='group', y='rMSE loss between true and resynth semantic vectors', data=df, palette=pal, bw=sigma, width_viol=.6, ax=ax2, orient=ort, box_showfliers=False)
    #ax1.set_xlim((-0.0001, 0.4))
    #ax2.set_xlim((-0.0001, 0.07))
    fig.savefig(f'plots/masterthesis_paul_{objective}_rmse.pdf')


    plt.clf()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ort = "h"; pal = "Set2"; sigma = .2
    pt.RainCloud(x='group', y='1 - corr between true and resynth semantic vector', data=df, palette=pal, bw=sigma, width_viol=.6, ax=ax1, orient=ort, box_showfliers=False)
    pt.RainCloud(x='group', y='rank of resynth semantic vector', data=df, palette=pal, bw=sigma, width_viol=.6, ax=ax2, orient=ort, box_showfliers=False)
    #ax1.set_xlim((-0.0001, 1.0))
    fig.savefig(f'plots/masterthesis_paul_{objective}_corr-rank.pdf')

    plt.clf()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ort = "h"; pal = "Set2"; sigma = .2
    pt.RainCloud(x='group', y='Wasserstein Distance between true and resynth acoustics', data=df, palette=pal, bw=sigma, width_viol=.6, ax=ax1, orient=ort, box_showfliers=False)
    #pt.RainCloud(x='group', y='rMSE loss between true and resynth semantic vectors', data=df, palette=pal, bw=sigma, width_viol=.6, ax=ax2, orient=ort, box_showfliers=False)
    #ax1.set_xlim((-0.0001, 0.3))
    #ax2.set_xlim((-0.0001, 0.07))
    fig.savefig(f'plots/masterthesis_paul_{objective}_wasserstein.pdf')




# play all words

import sounddevice as sd
import time

print(f"condition: {CONDITION}")
for index, row in dat.iterrows():
    print(f"{row['label']} (inv, planned acoustic_semvec, rec, seg)")
    sd.play(row['inv_sig'] * 4, row['inv_sr'], blocking=True)
    time.sleep(0.1)
    sd.play(row['prod_sig_acoustic_semvec'] * 4, 44100, blocking=True)
    time.sleep(0.1)
    sd.play(row['rec_sig'], row['rec_sr'], blocking=True)
    time.sleep(0.1)
    sd.play(row['seg_sig'], row['seg_sr'], blocking=True)
    time.sleep(0.5)

