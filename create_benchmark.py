import pandas as pd
import soundfile as sf

GECO_CORPUS_WAV_DIR = '/home/tino/Documents/phd/projects/VTL/create_vtl_corpus/create_vtl_corpus/vtl_corpus1.0_with_GECO/wav_original/'

dat_raw = pd.read_pickle('/home/tino/ml_cloud_nextcloud/GECO_data/geco_embedding_preprocessed_balanced_vectors_checked_extrem_long_removed_valid_matched_prot4.pkl')

#(dat['original_label'] == 'Problem').sum()
#(dat['original_label'] == 'Lehrer').sum()
#(dat['original_label'] == 'Wissenschaft').sum()
#(dat['original_label'] == 'Liebe').sum()
#(dat['original_label'] == 'genau').sum()
#(dat['original_label'] == 'Klasse').sum()

def sample(dat):
    """sample some rows from the data frame"""
    return pd.concat((
        dat[dat['original_label'] == 'Problem'],       # 3
        dat[dat['original_label'] == 'Studium'][:6],    # 5
        dat[dat['original_label'] == 'Wissenschaft'],  # 1
        dat[dat['original_label'] == 'Liebe'],         # 2
        dat[dat['original_label'] == 'genau'][:20],    # 20
        dat[dat['original_label'] == 'Klasse'][:5],    # 5
        ))

dat = dat_raw
dat = dat[dat['synthesized'] == False]
benchmark_data = sample(dat)
# select columns
benchmark_data = benchmark_data[['original_label', 'label', 'file', 'vector', 'cp_norm', 'sampling_rate', 'melspec_norm', 'melspec_norm_min']]
# rename columns
benchmark_data.columns = ['original_label', 'label', 'file', 'vector', 'seg_cp', 'rec_sr', 'rec_mel', 'rec_mel_min']

dat = dat_raw
dat = dat[dat['synthesized'] == True]
rec_sigs = list()
seg_cps = list()
for file_ in benchmark_data.file:
    sig, sr = sf.read(f'{GECO_CORPUS_WAV_DIR}{file_}')
    seg_cps.append(dat[dat['file'].str.contains(file_[0:6])]['cp_norm'].iloc[0])
    rec_sigs.append(sig)
benchmark_data['rec_sig'] = rec_sigs
benchmark_data['seg_cp'] = seg_cps

# reorder columns
benchmark_data = benchmark_data[['original_label', 'label', 'file', 'vector', 'rec_sig', 'rec_sr', 'rec_mel', 'rec_mel_min', 'seg_cp']]

benchmark_data.to_pickle('data/human_recordings.pickle')


#####################################
#  GECO: ABER ALSO ODER (ESSV2021)  #
#####################################
import pandas as pd

import sounddevice as sd

from paule import util

from tqdm import tqdm

tqdm.pandas()


dat = pd.read_pickle('geco_aber_also_oder_180_matched.pkl')

dat.columns = ['label', 'file', 'seg_cp_notnorm', 'seg_cp', 'rec_sig',
        'rec_sr', 'rec_mel_notnorm', 'rec_mel', 'synthesized', 'vector',
        'rec_mel_min']

dat = dat[dat['synthesized'] == False]

dat = dat[['label', 'file', 'vector', 'rec_sig', 'rec_sr', 'rec_mel', 'rec_mel_min', 'seg_cp']]
# fix wrong type of rec_sig and convert to float64 [-1, 1]:
dat['rec_sig'] = dat['rec_sig'] / 2 ** 15

dat.to_pickle('geco_aber-also-oder_30.pickle')


sd.play(dat['rec_sig'].iloc[75])


dat['seg_sig'] = dat['seg_cp'].progress_apply(lambda cp: util.speak(util.inv_normalize_cp(cp))[0])
dat['seg_sr'] = 44100

