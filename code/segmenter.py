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

REP_WIDTH=3
FILTER_WIDTH=13
HOP_LENGTH=512
SR=22050
MAX_REP=12

GAP_THRESHOLD = 1e-7
EDGE_THRESHOLD = 1e-12

MIN_SEG=10.0
MAX_SEG=45.0

MIN_TEMPO=70.0
MIN_NON_REPEATING = 4
MAX_FILTER_WIDTH  = 5

DISTANCE_QUANTILE = 0.5

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
        print 'Doubling up BPM: ', bpm, len(beats), ' -> ',
        bpm, beats = librosa.beat.beat_track(onsets=odf, sr=sr, hop_length=hop_length, bpm=2*bpm)
        print bpm, len(beats)
        
    return bpm, beats

def features(filename):
    y, sr = librosa.load(filename, sr=SR)
    
    y_perc, y_harm = hp_sep(y)
    
    bpm, beats = get_beats(y=y_perc, sr=sr, hop_length=HOP_LENGTH)
    
    M1 = np.abs(librosa.cqt(y=y_harm, 
                            sr=sr, 
                            hop_length=HOP_LENGTH, 
                            bins_per_octave=12, 
                            fmin=librosa.midi_to_hz(24), 
                            n_bins=96))
    
    
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
    idx    = np.argsort(e_vals)
    
    e_vals = e_vals[idx]
    e_vecs = e_vecs[:, idx]
    
    return e_vecs[:, :k].T, e_vals[k] - e_vals[k-1]

def boundary_estimate_bandwidth(D):
    n = len(D)
    
    D = np.sort(D, axis=1)
    
    # Sigma[i] is some quantile distance from the ith point
    sigma = D[:, min(n-1, 1 + int(DISTANCE_QUANTILE * (n-1)))]**0.5
    
    return np.multiply.outer(sigma, sigma)

def graph_autogain(V, tau):
    '''Given a vector of scaled distances (V=D/sigma(i,j)) and a threshold, compute the scale factor `A` such that
    min_i exp(-A V[i]) <= tau
    
    =>
    
    A >= -log(tau)/V
    
    A <- min_i -log(tau) / V[i] = log(1/tau) / max(V)
    '''
    
    return -np.log(tau) / np.max(V)

def non_maximal(dists):
    '''Send distances to zero if they're not the maximum within some range'''
    
    max_dists = scipy.ndimage.maximum_filter1d(dists, MAX_FILTER_WIDTH)
    
    return dists * (dists >= max_dists)

def make_boundary_graph(X):
    
    n = X.shape[1]
    
    D = scipy.spatial.distance.cdist(X.T, X.T, metric='sqeuclidean')
    
    sigma = boundary_estimate_bandwidth(D)
    
    # Estimate the bandwidth based on how many links you expect to cut
    # Scale by so that the max distance gets pushed to below some maximum value
    dists = np.diag(D / sigma, k=1)

    alpha = graph_autogain(dists, EDGE_THRESHOLD)

    dists = non_maximal(dists)
    A = np.eye(n)
    A[range(n-1), range(1, n)] = np.exp(-alpha * dists)

    return np.maximum(A, A.T)


def factor_and_gap(L, k_max=None):
    e_vals, e_vecs = scipy.linalg.eig(L)
    e_vals = e_vals.real
    idx = np.argsort(e_vals)
    
    e_vals = e_vals[idx]
    e_vecs = e_vecs[:, idx]
    
    # Normalize the spectrum for comparison across graphs
    e_vals = e_vals / e_vals.sum()
    
    # Truncate to at most k_max segments
    if k_max is not None:
        e_vals = e_vals[:k_max]
        e_vecs = e_vecs[:, :k_max]
    else:
        k_max = len(e_vals)
        
    # Threshold the spectrum and find the first jump
    # Find the first jump above the threshold
    # gap[i] = e_vals[i+1] - e_vals[i]
    gap = np.diff(e_vals * (e_vals > GAP_THRESHOLD))
    
    k = k_max - 1
    if gap.any():
        k = min(k, 1 + np.flatnonzero(gap)[0])
    
    return e_vecs[:, :k].T, e_vals[k] - e_vals[k-1]

def detect_boundaries(Lbf, k):
    C = sklearn.cluster.KMeans(n_clusters=k, tol=1e-10, n_init=50)
    labels = C.fit_predict(Lbf.T)
        
    boundaries = 1 + np.asarray(np.where(labels[:-1] != labels[1:])).reshape((-1,))
    
    return np.unique(np.concatenate([[0], boundaries, [len(labels)]]))

def label_rep_sections(X, boundaries, n_types):
    # Classify each segment centroid
    Xs = librosa.feature.sync(X, boundaries)
    
    C = sklearn.cluster.KMeans(n_clusters=n_types, tol=1e-8)
    
    labels = C.fit_predict(Xs.T)
    
    return zip(boundaries[:-1], boundaries[1:]), labels

def do_segmentation(X, beats, parameters):


    # Find the segment boundaries
    print '\tpredicting segments...'
    k_min, k_max  = get_num_segs(beats[-1])

    # Get the raw recurrence plot
    R = librosa.segment.recurrence_matrix(librosa.segment.stack_memory(X), 
                                            k=2 * int(np.ceil(np.sqrt(X.shape[1]))), 
                                            width=REP_WIDTH, 
                                            metric='sqeuclidean',
                                            sym=True).astype(np.float32)

    S = librosa.segment.structure_feature(R)

    Sf = clean_reps(S)

    # De-skew
    Rf = librosa.segment.structure_feature(Sf, inverse=True)

    # Symmetrize by force
    Rf = np.maximum(Rf, Rf.T)

    # Expand transitionals
    Rf = expand_transitionals(Rf, local=False)

    # We can jump to a random neighbor, or +- 1 step in time
    # Call it the infinite jukebox matrix
    M = np.maximum(Rf, (np.eye(Rf.shape[0], k=1) + np.eye(Rf.shape[0], k=-1)))

    # Get the random walk graph laplacian
    L = rw_laplacian(M)

    # Get the bottom k eigenvectors of L
    Lf = factorize(L, k=1+MAX_REP)[0]


    best_gap        = -np.inf
    best_boundaries = None
    best_n_types    = None

    for n_types in range(2, MAX_REP+1):
        # Build the affinity matrix on the first n_types-1 repetition features
        A = make_boundary_graph(Lf[:n_types])
    
        # Compute its laplacian and factorization
        L_boundary = rw_laplacian(A)
    
        L_factor, gap = factor_and_gap(L_boundary, k_max)
    
        k = L_factor.shape[0]
        boundaries = detect_boundaries(L_factor, max(n_types, k))
    
        print '%s%.4e:\t%2d types:\t%3d segments' % (' ' * np.round(np.abs(np.log10(gap))), gap, n_types, k)
        
        if (gap > best_gap) and (k_min <= k):
            best_n_types   = n_types
            best_gap       = gap
        
            # Do boundary detection by clustering on L_factor
            best_boundaries = boundaries

    intervals, labels = label_rep_sections(Lf[:best_n_types], best_boundaries, best_n_types)

    # Output lab file
    print '\tsaving output to ', parameters['output_file']
    save_segments(parameters['output_file'], best_boundaries, beats, labels)

def process_arguments():
    parser = argparse.ArgumentParser(description='Music segmentation')

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
