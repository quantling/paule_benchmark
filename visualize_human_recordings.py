import pickle


from paule import visualize

with open('results_20210901_default/vector_acoustic-semvec_results.pickle', 'rb') as pfile:
    results = pickle.load(pfile)


with open('results_20210901_default/init-seg_acoustic-semvec_results.pickle', 'rb') as pfile:
    results = pickle.load(pfile)



with open('results_20210901_segment-embedder/acoustics_acoustic-semvec_results.pickle', 'rb') as pfile:
    results = pickle.load(pfile)

with open('results_20210901_segment-embedder/init-seg_acoustic-semvec_results.pickle', 'rb') as pfile:
    results = pickle.load(pfile)


