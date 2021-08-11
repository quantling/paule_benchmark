======
README
======

Paule Benchmarks
================
This will grow to a collection of benchmarks to compare the different flavours
of the Predcitive Articulatory speech synthesis model Utelising Lexical
Embeddings (Paule) package and to show its limitations and strength.


Human recordings benchmark
==========================
The human recordings benchmark compares the resynthesis quality for a small set
of words (Lehrer, Wissenschaft, Liebe for now) between the recovering of the
segment based synthesis and the resynthesis of the human recording.


Run
---
To run the benchmark install the paule package and execute::

   python benchmark_human_recordings.py

The benchmark should finish in around ???? hours depending on your hardware
specifications.

Results
-------
Results of the benchmark can be found below
``results/benchmark_human_recordings/``.


TODO
====
- Benchmark: "minimal pair" / local contrast
- (Benchmark: aber, also, oder)
- Benchmark: babi, babu, baba

Questions
=========
- Predictive/Forward model comparison ("small" models -> bad gradients?)


Metrics
-------
- loss improvement production real (mse acoustics, mse semantics, cross corr / rank)
- loss improvement prediction imagined
- final production loss
- final prediction loss
- babi, babu, baba: formant transitions in first /a/; tongue raising

- number of trainable parameters
- execution time
- evtl. training time


"Experiments"
-------------
- segment based resynthesis vs. human recording
- formant transitions in babi, babu, baba
- semantically driven synthesis "Miete" -> "Miete", "Miete" -> "mitte", "mitte"
  -> "Miete" and, "mitte" -> "mitte"; Übergang visualisieren
- semantically driven synthesis "aber" -> "oder" and "oder" -> "aber"??

- cross-correlation loss vs. MSE loss


- initilize "mitte" -> target 0% "mitte"-"Miete"
- initilize "mitte" -> target 10% "mitte"-"Miete"
- initilize "mitte" -> target 50% "mitte"-"Miete"
- initilize "mitte" -> target 90% "mitte"-"Miete"
- initilize "mitte" -> target 100% "mitte"-"Miete"

Hypothesis: duration of /i/ increases monotonically.

Story-Telling objective
=======================

- possible to start from semvec (witout acoustics)
- enhancement through embedder

- https://github.com/lochenchou/MOSNet

