#!/usr/bin/env python
# CREATED:2013-08-22 12:20:01 by Brian McFee <brm2132@columbia.edu>
'''Music segmentation using timbre, pitch, repetition and time.

If run as a program, usage is:

    ./segmenter.py AUDIO.mp3 OUTPUT.lab

'''


import sys
import os
import argparse
import string

import numpy as np
import scipy.spatial
import scipy.signal
import scipy.linalg

import sklearn.cluster

# Requires librosa-develop 0.3 branch
import librosa

# Suppress neighbor links within REP_WIDTH beats of the current one
REP_WIDTH=7

# Only consider repetitions of at least (FILTER_WIDTH-1)/2
FILTER_WIDTH=15

# How much state to use?
N_STEPS = 2

# Which similarity metric to use?
METRIC='sqeuclidean'

# Sample rate for signal analysis
SR=22050

# Hop length for signal analysis
HOP_LENGTH=512

# Maximum number of structural components to consider
MAX_REP=12

# Minimum and maximum average segment duration
MIN_SEG=10.0
MAX_SEG=30.0

# Minimum tempo threshold; if we dip below this, double the bpm estimator and resample
MIN_TEMPO=70.0

# Minimum duration (in beats) of a "non-repeat" section
MIN_NON_REPEATING = (FILTER_WIDTH - 1) / 2

SEGMENT_NAMES = list(string.ascii_uppercase)
for x in string.ascii_uppercase:
    SEGMENT_NAMES.extend(['%s%s' % (x, y) for y in string.ascii_lowercase])

def hp_sep(y):
    D_h, D_p = librosa.decompose.hpss(librosa.stft(y))
    return librosa.istft(D_h), librosa.istft(D_p)

def get_beats(y, sr, hop_length):
    
    odf = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length, aggregate=np.median, n_mels=128)
    bpm, beats = librosa.beat.beat_track(onsets=odf, sr=sr, hop_length=hop_length)
    
    if bpm < MIN_TEMPO:
        bpm, beats = librosa.beat.beat_track(onsets=odf, sr=sr, hop_length=hop_length, bpm=2*bpm)
        
    return bpm, beats

def features(filename):
    print '\t[1/4] loading audio'
    y, sr = librosa.load(filename, sr=SR)
    
    print '\t[2/4] Separating harmonic and percussive signals'
    y_perc, y_harm = hp_sep(y)
    
    print '\t[3/4] detecting beats'
    bpm, beats = get_beats(y=y_perc, sr=sr, hop_length=HOP_LENGTH)
    
    print '\t[4/4] generating CQT'
    M1 = np.abs(librosa.cqt(y=y_harm, 
                            sr=sr, 
                            hop_length=HOP_LENGTH, 
                            bins_per_octave=12, 
                            fmin=librosa.midi_to_hz(24), 
                            n_bins=84))
    
    
    M1 = librosa.logamplitude(M1**2.0, ref_power=np.max)

    n = M1.shape[1]
    
    beats = beats[beats < n]
    
    beats = np.unique(np.concatenate([[0], beats]))
    
    times = librosa.frames_to_time(beats, sr=sr, hop_length=HOP_LENGTH)
    
    times = np.concatenate([times, [float(len(y)) / sr]])
    
    return librosa.feature.sync(M1, beats, aggregate=np.median), times

def save_segments(outfile, boundaries, beats, labels=None):

    if labels is None:
        labels = [('Seg#%03d' % idx) for idx in range(1, len(boundaries))]

    times = beats[boundaries]
    with open(outfile, 'w') as f:
        for idx, (start, end, lab) in enumerate(zip(times[:-1], times[1:], labels), 1):
            f.write('%.3f\t%.3f\t%s\n' % (start, end, lab))
    
    pass

def get_num_segs(duration):
    kmin = max(2, np.floor(duration / MAX_SEG).astype(int))
    kmax = max(3, np.ceil(duration / MIN_SEG).astype(int))

    return kmin, kmax

def clean_reps(S):
    # Median filter with reflected padding
    Sf = np.pad(S, [(0, 0), (FILTER_WIDTH, FILTER_WIDTH)], mode='reflect')
    Sf = scipy.signal.medfilt2d(Sf, kernel_size=(1, FILTER_WIDTH))
    Sf = Sf[:, FILTER_WIDTH:-FILTER_WIDTH]
    return Sf

def expand_transitionals(R, local=True):
    '''Sometimes, a frame does not repeat.  
    Sequences of non-repeating frames are bad news, so we'll link them up together as a transitional clique.
    
    input: 
    
      - filtered repetition matrix R
    '''
    
    n = len(R)
    R_out = R.copy()
    
    degree = np.sum(R, axis=0)
    
    start = None
    
    all_idx = []
    
    for i in range(n):
        if start is not None:
            # If we're starting a new repeating section,
            # or we're at the end
            if (i == n - 1) or (degree[i] > 0):
                
                # Fill in R_out[start:i, start:i]
                idx = slice(start, i)
                
                if i - start >= MIN_NON_REPEATING:
                    if local:
                        # Add links for all unique pairs
                        R_out[np.ix_(idx, idx)] = 1
                        R_out[idx, idx] = 0
                    else:
                        all_idx.extend(range(start, i))
                    
                # reset the counter
                start = None
                
        elif degree[i] == 0:
            start = i
    
    if not local and all_idx:
        # Add links for all unique pairs
        R_out[np.ix_(all_idx, all_idx)] = 1
        R_out[all_idx, all_idx] = 0
        
    return R_out

def rw_laplacian(A):
    Dinv = np.sum(A, axis=1)**-1.0
    Dinv[~np.isfinite(Dinv)] = 1.0
    L = np.eye(A.shape[0]) - (Dinv * A.T).T
    return L

def factorize(L, k=20):
    e_vals, e_vecs = scipy.linalg.eig(L)
    e_vals = e_vals.real
    e_vecs = e_vecs.real
    idx    = np.argsort(e_vals)
    
    e_vals = e_vals[idx]
    e_vecs = e_vecs[:, idx]
    
    return e_vecs[:, :k].T, e_vals[k] - e_vals[k-1]

def label_rep_sections(X, boundaries, n_types):
    # Classify each segment centroid
    Xs = librosa.feature.sync(X, boundaries)
    
    C = sklearn.cluster.KMeans(n_clusters=n_types, tol=1e-8)
    
    labels = C.fit_predict(Xs.T)
    
    return zip(boundaries[:-1], boundaries[1:]), labels

def label_clusterer(Lf, k_min, k_max):
    best_boundaries = None
    best_n_types    = None

    for n_types in range(2, MAX_REP+1):
        # Try to label the data with n_types 
        C = sklearn.cluster.KMeans(n_clusters=n_types, tol=1e-8)
        labels = C.fit_predict(Lf[:n_types].T)
        
        # Find the label change-points
        boundaries = 1 + np.asarray(np.where(labels[:-1] != labels[1:])).reshape((-1,))
        boundaries = np.unique(np.concatenate([[0], boundaries, [len(labels)]]))
        
        if best_boundaries is None or (k_min < len(boundaries) and len(boundaries) < k_max):
            best_boundaries = boundaries
            best_n_types    = n_types
    
    intervals, labels = label_rep_sections(Lf[:best_n_types], best_boundaries, best_n_types)
    
    return best_boundaries, labels

def estimate_bandwidth(D, k):
    D_sort = np.sort(D, axis=1)

    sigma = np.median(D_sort[:, 1+k])
    return sigma

def self_similarity(X, k):
    D = scipy.spatial.distance.cdist(X.T, X.T, metric=METRIC)
    sigma = estimate_bandwidth(D, k)
    A = np.exp(-0.5 * (D / sigma))
    return A

def do_segmentation(X, beats, parameters):

    # Find the segment boundaries
    print '\tpredicting segments...'
    k_min, k_max  = get_num_segs(beats[-1])

    # Get the raw recurrence plot
    Xs = librosa.segment.stack_memory(X, n_steps=N_STEPS)

    k_link = 1 + int(np.ceil(np.log2(X.shape[1])))
    R = librosa.segment.recurrence_matrix(  Xs, 
                                            k=k_link, 
                                            width=REP_WIDTH, 
                                            metric=METRIC,
                                            sym=True).astype(np.float32)
    A = self_similarity(Xs, k=k_link)

    # Mask the self-similarity matrix by recurrence
    S = librosa.segment.structure_feature(R)

    Sf = clean_reps(S)

    # De-skew
    Rf = librosa.segment.structure_feature(Sf, inverse=True)

    # Symmetrize by force
    Rf = np.maximum(Rf, Rf.T)

    # Expand non-repeating transitional sections
    Rf = expand_transitionals(Rf, local=False)

    # Suppress the diagonal
    Rf[np.diag_indices_from(Rf)] = 0

    # We can jump to a random neighbor, or +- 1 step in time
    # Call it the infinite jukebox matrix
    M = np.maximum(Rf, (np.eye(Rf.shape[0], k=1) + np.eye(Rf.shape[0], k=-1)))
    
    # Get the random walk graph laplacian
    L = rw_laplacian(M * A)

    # Get the bottom k eigenvectors of L
    Lf = factorize(L, k=1+MAX_REP)[0]

    boundaries, labels = label_clusterer(Lf, k_min, k_max)

    # Output lab file
    print '\tsaving output to ', parameters['output_file']
    save_segments(parameters['output_file'], boundaries, beats, labels)

def process_arguments():
    parser = argparse.ArgumentParser(description='Music segmentation')

    parser.add_argument(    '-v', '--verbose',
                            dest    =   'verbose',
                            action  =   'store_true',
                            default =   False,
                            help    =   'verbose output')

    parser.add_argument(    'input_song',
                            action  =   'store',
                            help    =   'path to input audio data')

    parser.add_argument(    'output_file',
                            action  =   'store',
                            help    =   'path to output segment file')

    return vars(parser.parse_args(sys.argv[1:]))

if __name__ == '__main__':

    parameters = process_arguments()

    # Load the features
    print '- ', os.path.basename(parameters['input_song'])
    X, beats    = features(parameters['input_song'])

    do_segmentation(X, beats, parameters)
