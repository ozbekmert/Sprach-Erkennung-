import math
import numpy as np
from praatio import tgio


def sec_to_samples(x, sampling_rate):
    """
    Wandelt einen skalaren Zeitwert in einen diskreten Abtastzeitpunkt um.
    
    Parameters:
        x: Zeitwert in Sekunden
        sampling_rate: Abtastfrequenz in Hz
        
    Returns:
        (int) diskreten Abtastzeitpunkt
    """
    
    # Abtastperiode
    sample = round(x * sampling_rate)
    
    # Abtastzeitpunkt ist abgerundet
    return int(sample)


def next_pow2(x):
    next_pow = 1
    counter = 0    
    if x == 0:
        return 0
    else:
        while next_pow < abs(x):
            next_pow = next_pow * 2
            counter += 1        
    return counter


def next_pow2_samples(x, sampling_rate):
    return next_pow2(sec_to_samples(x, sampling_rate))


def get_num_frames(signal_length_samples, window_size_samples, hop_size_samples):
    """        
    Returns:
        (int) Anzahl benötigter Rahmen (frames)
    """
    # Anzahl überlappender diskreter Abtastzeitpunkte von zwei Rahmen
    num_overlaps = window_size_samples - hop_size_samples
    return math.ceil((signal_length_samples - num_overlaps)/(window_size_samples - num_overlaps))


def normalize2d_array(x):
    # normalized between -1 and 1
    
    # avoid division by zero
    if np.max(np.abs(x)) != 0:
        return x/np.max(np.abs(x))
    else:
        return x


def hz2mel(hz):
    return 2595 * np.log10(1+hz/700.)


def mel2hz(mel):
    return 700*(10**(mel/2595.0)-1)


def array_vector_multiplication(data, window_type):
    m = np.array(data)
    c = np.array(window_type)
    return np.multiply(m, c)


def sec_to_frame(x, sampling_rate, hop_size_samples):
    """
    Converts time in seconds to frame index.

    :param x:  time in seconds
    :param sampling_rate:  sampling frequency in hz
    :param hop_size_samples:    hop length in samples
    :return: frame index
    """
    return int(np.floor(sec_to_samples(x, sampling_rate) / hop_size_samples))


def divide_interval(num, start, end):
    """
    Divides the number of states equally to the number of frames in the interval.

    :param num:  number of states.
    :param start: start frame index
    :param end: end frame index
    :return starts: start indexes
    :return end: end indexes
    """
    interval_size = end - start
    # gets remainder
    remainder = interval_size % num
    # init sate count per state with min value
    count = [int((interval_size - remainder)/num)] * num
    # the remainder is assigned to the first n states
    count[:remainder] = [x + 1 for x in count[:remainder]]
    # init starts with first start value
    starts = [start]
    ends = []
    # iterate over the states and sets start and end values
    for c in count[:-1]:
        ends.append(starts[-1] + c)
        starts.append(ends[-1])

    # set last end value
    ends.append(starts[-1] + count[-1])

    return starts, ends


def praat_file_to_target(praat_file, sampling_rate, window_size_samples, hop_size_samples, hmm):
    """
    Reads in praat file and calculates the word-based target matrix.

    :param praat_file: *.TextGrid file.
    :param sampling_rate: sampling frequency in hz
    :param window_size_samples: window length in samples
    :param hop_size_samples: hop length in samples
    :return: target matrix for DNN training
    """
    # gets list of intervals, start, end, and word/phone
    intervals, min_time, max_time = praat_to_interval(praat_file)

    # gets dimensions of target
    max_sample = sec_to_samples(max_time, sampling_rate)
    num_frames = get_num_frames(max_sample, window_size_samples, hop_size_samples)
    num_states = hmm.get_num_states()

    # init target with zeros
    target = np.zeros((num_frames, num_states))

    # parse intervals
    for interval in intervals:
        # get state index, start and end frame
        states = hmm.input_to_state(interval.label)
        start_frame = sec_to_frame(interval.start, sampling_rate, hop_size_samples)
        end_frame = sec_to_frame(interval.end, sampling_rate, hop_size_samples)

        # divide the interval equally to all states
        starts, ends = divide_interval(len(states), start_frame, end_frame)

        # assign one-hot-encoding to all segments of the interval
        for state, start, end in zip(states, starts, ends):
            # set state from start to end to 1
            target[start:end, state] = 1

    # find all columns with only zeros...
    zero_column_idxs = np.argwhere(np.amax(target, axis=1) == 0)
    # ...and set all as silent state
    target[zero_column_idxs, hmm.input_to_state('sil')] = 1

    return target


def praat_to_interval(praat_file):
    """
    Reads in one praat file and returns interval description.

    :param praat_file: *.TextGrid file path

    :return itervals: returns list of intervals,
                        containing start and end time and the corresponding word/phone.
    :return min_time: min timestamp of audio (should be 0)
    :return max_time: min timestamp of audio (should be audio file length)
    """
    # read in praat file (expects one *.TextGrid file path)
    tg = tgio.openTextgrid(praat_file)

    # read return values
    itervals = tg.tierDict['words'].entryList
    min_time = tg.minTimestamp
    max_time = tg.maxTimestamp

    # we will read in word-based
    return itervals, min_time, max_time


def viterbi(logLike, logPi, logA):  # loglike = b; logPi = pi, logA = a
    # Initialize
    T = logLike.shape[0]
    N = logA.shape[1]
    #print("T, N = " + str(T) + ", " + str(N))
    y = np.zeros((T, N))
    v = np.zeros((T, N))

    for i in range(N):
        y[0][i] = logPi[i] + logLike[0][i]
        v[0][i] = -1

    # Main loop
    for t in range(1, T):
        for j in range(N):
            temp = np.zeros(N)
            for i in range(N):
                temp[i] = y[t-1][i] + logA[i][j]

            y[t][j] = np.max(temp) + logLike[t][j]
            v[t][j] = np.argmax(temp)

    pstar = np.max(y[T-1])
    bestpathindex = np.argmax(y[T-1])
    best_path = np.zeros(T)
    best_path[-1] = bestpathindex

    for t in range(T-1):
        best_path[T-2-t] = v[T-1-t][int(best_path[T-1-t])]
        
    # Ergebnis
    return best_path, pstar


def limLog(x):
    """
    Log of c.

    :param x: numpy array.
    :return: log of x.
    """
    MINLOG = 1e-100
    return np.log(np.maximum(x, MINLOG))


def _needlemann_wunsch(reference, transcript):
    """
    Dynamic programming algorithm to align true transcription with output sequence in order to find the smallest distance.

    :param reference: Reference sequence (true transcription).
    :param transcript: output of viterbi
    
    :return ref_align: Alignment for reference.
    :return trans_align: Alignment for transcript.
    """
    gap_score=-1
    sim_func=lambda x, y: 0 if x == y else -1
    
    n_ref = len(reference)
    n_trans = len(transcript)
    d_mat = np.zeros(shape=[n_trans + 1, n_ref + 1,])
    # Initialize the dynamic programming calculation using base conditions
    for idr in range(n_ref + 1):
        d_mat[0, idr] = gap_score * idr
    for idt in range(n_trans + 1):
        d_mat[idt, 0] = gap_score * idt

    # Calculate all D[i,j]
    for i in range(1, n_trans + 1):
        for j in range(1, n_ref + 1):
            match = d_mat[i-1,j-1] + sim_func(reference[j-1], transcript[i-1])
            gaps = d_mat[i,j-1] + gap_score
            gapt = d_mat[i-1,j] + gap_score
            d_mat[i,j] = np.max([match, gaps, gapt])

    # Do the Traceback to create the alignment
    i = n_trans
    j = n_ref
    ref_align = []
    trans_align = []
    while (i > 0) and (j > 0):
        if d_mat[i, j] - sim_func(reference[j - 1], transcript[i - 1]) == d_mat[i - 1, j - 1]:
            ref_align.insert(0, reference[j - 1])
            trans_align.insert(0, transcript[i - 1])
            i = i - 1
            j = j - 1
        elif d_mat[i, j] - gap_score == d_mat[i, j - 1]:
            ref_align.insert(0, reference[j - 1])
            trans_align.insert(0, None)
            j = j - 1
        elif d_mat[i, j] - gap_score == d_mat[i - 1, j]:
            ref_align.insert(0, None)
            trans_align.insert(0, transcript[i - 1])
            i = i - 1
        else:
            raise('should not happen')

    while j > 0:
        ref_align.insert(0, reference[j - 1])
        trans_align.insert(0, None)
        j = j - 1

    while i > 0:
        ref_align.insert(0, None)
        trans_align.insert(0, transcript[i - 1])
        i = i - 1
    
    return(ref_align, trans_align)


def needlemann_wunsch(reference, transcript):
    """
    Counts number of errors.

    :param reference: Reference sequence (true transcription).
    :param transcript: output of viterbi

    :return N: Total number of words.
    :return D: Number of deleted words.
    :return I: Number of inserted words.
    :return S: Number of substituted words.
    """
    insertions = 0
    deletions = 0
    substitutions = 0
    ref_align, trans_align = _needlemann_wunsch(reference, transcript)
    for idr in range(len(ref_align)):
        if ref_align[idr] is None:
            insertions += 1
        elif trans_align[idr] is None:
            deletions += 1
        elif ref_align[idr] != trans_align[idr]:
            substitutions += 1

    return len(reference), deletions, insertions, substitutions
