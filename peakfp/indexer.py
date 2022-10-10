"""Creates PeakFP index.
"""
import argparse
import pickle

from utils import load_pickle

def create_index(peakfps_list_path, index_path):
    """Creates PeakFP index. The hash corresponds to the peak frequency
    and time.

    Args:
        peakfps_list_path (str): Path of \n separated list of
            fingerprints paths.
        index_path (str): Destination index path.
    Returns:
        None
    """
    with open(peakfps_list_path, 'r', encoding='utf-8') as fin:
        peakfps_list = fin.read().splitlines()
    index = {'refs': peakfps_list,
             'hashes': {}}
    for i_ref, peakfp_path in enumerate(peakfps_list):
        print(peakfp_path)
        peaks = load_pickle(peakfp_path)
        for freq, time in peaks:  # time given in frames
            if freq not in index['hashes']:
                hash_value = freq  #  peak frequency is the hash
                index['hashes'][hash_value] = []
            index['hashes'][freq].append((i_ref, time))
    with open(index_path, 'wb') as fout:
        pickle.dump(index, fout)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "peakfps_list_path",
        help="Path of \n separated list of fingerprints paths.")
    parser.add_argument(
        "index_path",
        help="Destination index path.")
    args = parser.parse_args()
    create_index(args.peakfps_list_path,
                 args.index_path)
