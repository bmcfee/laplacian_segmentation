#!/usr/bin/env python
# CREATED:2013-08-22 12:20:01 by Brian McFee <brm2132@columbia.edu>
'''Music segmentation using timbre, pitch, repetition and time.

If run as a program, usage is:
'''


import os
import cPickle as pickle

import segmenter

def features(input_song):

    with open(input_song, 'r') as f:
        data = pickle.load(f)

    return data['features'], data['beats']

if __name__ == '__main__':

    parameters = segmenter.process_arguments()

    # Load the features
    print '- ', os.path.basename(parameters['input_song'])
    X, beats     = features(parameters['input_song'])

    segmenter.do_segmentation(X, beats, parameters)
