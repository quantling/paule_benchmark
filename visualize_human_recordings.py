import pickle
import os


from paule import visualize

with open('results_20210901_default/vector_acoustic-semvec_results.pickle', 'rb') as pfile:
    results = pickle.load(pfile)


with open('results_20210901_default/init-seg_acoustic-semvec_results.pickle', 'rb') as pfile:
    results = pickle.load(pfile)



with open('results_20210901_segment-embedder/acoustics_acoustic-semvec_results.pickle', 'rb') as pfile:
    results = pickle.load(pfile)

with open('results_20210901_segment-embedder/init-seg_acoustic-semvec_results.pickle', 'rb') as pfile:
    results = pickle.load(pfile)


LABEL = ['praktisch', 'Anglistik', 'tatsaechlich']

RESULT_DIR = 'results_20210906_default/'
with open(os.path.join(RESULT_DIR, 'acoustics_acoustic-semvec_results.pickle'), 'rb') as pfile:
    results = pickle.load(pfile)
for ii, result in enumerate(results):
    visualize.vis_result(result, condition=f'acoustics_acoustic-semvec_{LABEL[ii]}', folder='visualize')

with open(os.path.join(RESULT_DIR, 'vector_acoustic-semvec_results.pickle'), 'rb') as pfile:
    results = pickle.load(pfile)
for ii, result in enumerate(results):
    visualize.vis_result(result, condition=f'vector_acoustic-semvec_{LABEL[ii]}', folder='visualize')

