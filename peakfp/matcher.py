"""PeakFP matcher.
"""
import argparse
import os
from collections import defaultdict

import intervaltree
import numpy as np

import constants
from utils import load_pickle


def window_peaks(peaks: np.array, window_size: int, hop_size: int):
    """Group peaks into time windows.
    Args:
        peaks (np.array): PeakFP fingerprint.
        window_size (int): Window size in samples.
        hop_size (int): Hop size in samples.
    Yields:
        peaks (list): Windowed peaks.
    """
    max_time = peaks[-1][1]
    n_windows = int(np.ceil(max(0, max_time - window_size + 1) /
                            hop_size)) + 1
    for window_idx in range(n_windows):
        window_start = window_idx * hop_size
        window_end = window_start + window_size
        yield peaks[(peaks[:, 1] >= window_start) &
                    (peaks[:, 1] < window_end)]


def get_matches(w_qpeaks: list, index: dict, common_peaks_threshold: int=5):
    """Find common, aligned peaks between query and references.
    Args:
        w_qpeaks (list): Window peaks.
        index (dict): FP index with keys: ["refs", "hashes"]
        common_peaks_threshold (int): Threshold that establishes the minimum
            number of common peaks to match.
    Returns:
        A dict with the windowed matches.
    """
    wmatches = defaultdict(
        lambda: {'score': [0],  # in later steps will add a value per window
                 'start': float('inf'),
                 'end': -float('inf')})
    for query_freq, query_time in w_qpeaks:
        query_hash = query_freq  # peak frequencies are the fingerprint hashes
        hits = index['hashes'].get(query_hash, [])
        for ref_id, ref_time in hits:
            diff_time = query_time - ref_time
            k = f"{ref_id}__{diff_time}"
            wmatches[k]['score'][0] = wmatches[k]['score'][0] + 1
            wmatches[k]['start'] = min(wmatches[k]['start'], query_time)
            wmatches[k]['end'] = max(wmatches[k]['end'], query_time)
    return {k: v for k, v in wmatches.items() if
            v['score'][0] >= common_peaks_threshold}


def merge_matches(raw_matches):
    """Merge overlapping matches
    Args:
        raw_matches (list): list of matches
    Returns:
        merged_matches(list): list of merged matches.

    TODO Test with different sets of matches"""
    for i in range(1, len(raw_matches)):
        m_1 = raw_matches[i - 1]
        m_2 = raw_matches[i]
        if 'refid' in m_1 and (  # refid and difft is available in raw_matches
                (m_1['refid'], m_1['difft']) != (m_1['refid'], m_2['difft'])):
            continue
        if m_1['end'] >= m_2['start']:
            m_2['start'] = m_1['start']
            m_2['score'] = m_1['score'] + m_2['score']
            m_1['score'] = None  # mark m_1 to be removed later
    merged_matches = []
    for match in raw_matches:
        if match['score']:
            merged_matches.append(match)
    return merged_matches


def select_longest(matches):
    """Select the longest match when a variety of them overlap.
    Args:
        matches (list): List of overlapping matches.
    Returns:
        match (dict): Longest match of the input list.

    """
    # in order to avoid choosing random ref_id in case there is no longest:
    matches = sorted(matches, key=lambda x: (x['refid'], x['difft']))
    max_match_length = 0
    longest_match = {}
    for match in matches:
        match_length = match['end'] - match['start']
        if match_length > max_match_length:
            max_match_length = match_length
            longest_match = match
    return longest_match.copy()


def postprocess_raw_matches(all_matches: dict):
    """Postprocess raw matches.
    Sorts all raw matches by start time and calls merge_matches to merge
    overlapping matches. Then, returns the merged matches with the correct
    format (processed_matches).

    Args:
        all_matches (dict): Matches (common peaks) between query and index.
    Returns:
        processed_matches (list): Processed matches.
    """
    processed_matches = []
    for refid_difft, matches in all_matches.items():
        merged_matches = merge_matches(sorted(matches,
                                             key=lambda x: x['start']))
        for match in merged_matches:
            ref_id, diff_time = [int(x) for x in refid_difft.split('__')]
            match.update({'refid': ref_id,
                          'difft': diff_time})
            processed_matches.append(match)
    return processed_matches


def select_best_matches(processed_matches):
    """Select best matches.
    Selects longest matches in case two different matches overlap. It also
    performs a merge after the select_longest selection is made.

    TODO Test with different sets of matches
    """
    itree = intervaltree.IntervalTree()
    for match in processed_matches:
        if match['start'] == match['end']: # this happens when matching against reference in index
            continue
        itree.addi(match['start'], match['end'], match)
    itree.split_overlaps()
    grouped_overlaps = defaultdict(list)
    for interval in itree:
        istart_iend = f"{interval.begin}__{interval.end}"
        grouped_overlaps[istart_iend].append(interval.data)

    best_matches = []
    for istart_iend, matches in grouped_overlaps.items():
        istart, iend = [int(x) for x in istart_iend.split('__')]
        selected_match = select_longest(matches)
        selected_match['start'] = istart
        selected_match['end'] = iend
        best_matches.append(selected_match)
    best_matches = merge_matches(sorted(best_matches,
                                           key=lambda x: x['start']))
    return best_matches


def format_matches(matches: list, refs_names: list):
    """Standarizes matches format.
    Args:
        matches (list): Matches to format.
        ref_names (list): References names in index.
    Returns:
        Formatted_matches (list of dicts)
    """
    frame_length = constants.FFT_HOP_SIZE / constants.SAMPLE_RATE  # in seconds
    formatted_matches = []
    for match in matches:
        formatted_matches.append(
            {'query_start': match['start'] * frame_length,
             'query_end': match['end'] * frame_length,
             'ref': refs_names[match['refid']],
             'ref_start': (match['start'] - match['difft']) * frame_length,
             'ref_end': (match['end'] - match['difft']) * frame_length,
             'score': np.round(np.mean(match['score']), 1)})
    return formatted_matches


def run_matching(argslist: list):
    """Matching main process.
    1. Takes query fp peaks stored in query_fp_path.
    2. Groups them into windows.
    3. Find common, aligned peaks between query and references (raw matches).
    4. Postprocesses raw matches
    5. Select best matches (i.e. longest when matches overlap)
    6. Format final matches

    Args:
        args (list): list of the following args:
            query_fp_path (str): Path of query peakfp fingerprint.
            index_path (str): Index path.
            common_peaks_th (int): Minimum score (min common peaks) to match.
            window_size (int): Matching window size.
            hop_size (int): Matching hop size.
    Returns:
        formatted_matches (list): List of matches.
    """
    query_fp_path, index_path, common_peaks_th, window_size, hop_size = argslist

    qpeaks = load_pickle(query_fp_path)
    index = load_pickle(index_path)
    windowed_qpeaks = window_peaks(qpeaks,
                                   window_size=window_size,
                                   hop_size=hop_size)
    all_matches = defaultdict(list)
    for w_qpeaks in windowed_qpeaks:
        wmatches = get_matches(w_qpeaks, index, common_peaks_th)
        for refid_difft, match in wmatches.items():
            all_matches[refid_difft].append(match)

    processed_matches = postprocess_raw_matches(all_matches)
    best_matches = select_best_matches(processed_matches)
    #---
    return format_matches(best_matches, index['refs'])



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "query_fp_path",
        help="Path of query peakfp fingerprint.")
    parser.add_argument(
        "index_path",
        help="Index path.")
    parser.add_argument(
        "--threshold",
        default=5,
        type=int,
        help="Minimum score. Equivalent to number of peaks in common.")
    parser.add_argument(
        "--window_size",
        default=600,
        type=int,
        help= "Matching window size. 600 is equivalent to 9.6sec (with FFT Hop "
              "size of 180.")
    parser.add_argument(
        "--hop_size",
        default=300,
        type=int,
        help="Matching hop size.")
    matcher_args = parser.parse_args()

    query_fp_path = vars(matcher_args).get('query_fp_path')
    index_path = vars(matcher_args).get('index_path')
    common_peaks_th = (vars(matcher_args).get('threshold'))
    window_size = int(vars(matcher_args).get('window_size'))
    hop_size = int(vars(matcher_args).get('hop_size'))
    argslist = (query_fp_path, index_path, common_peaks_th,
                window_size, hop_size)
    query_matches = run_matching(argslist)
    print('"Query","Query begin time","Query end time","Reference",'
          '"Reference begin time","Reference end time","Confidence"')
    print('"","","","","","",""')
    def_matches = []
    for qmatch in query_matches:
        qmatch.update({'query': os.path.basename(query_fp_path)})
        print('"{}","{}","{}","{}","{}","{}","{}"'.format(
                qmatch['query'], qmatch['query_start'], qmatch['query_end'],
                qmatch['ref'], qmatch['ref_start'],
                qmatch['ref_end'], qmatch['score']))
