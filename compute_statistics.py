import sys
import argparse
import pprint
import pandas as pd
import intervaltree as itree


def create_tree(gts, results, threshold=None):
    """Creates intervaltree with Groundtruth and Tech results.
    Args:
        results (DataFrame): Tech results pandas DataFrame.
        gts (DataFrame): Groundtruth (annotations) pandas DataFrame.
    Returns:
        tree (intervaltree): tree.
    """
    def _add_to_tree(tree, segments, tag):
        """Add relevant segments to the tree.
        Args:
            tree (intervaltree): tree.
            segments (DataFrame): Segments desired to be added to the tree.
            tag (str): "GT" or "RS". Segment tag.
        Returns:
            tree (intervaltree): updated tree.
        """
        for idx, s in segments.iterrows():
            if (s['query_end'] - s['query_start']) <= 0:
                # intervaltree doesn't allow 0-length segments
                continue
            if threshold is not None and tag=='RS' and s['score'] < threshold:
                continue
            interval = itree.Interval(s['query_start'], s['query_end'],
                                      ["%s %s" % (tag, idx)])

            if s['query'] not in tree:
                tree[s['query']] = {}
            if s['reference'] not in tree[s['query']]:
                tree[s['query']][s['reference']] = itree.IntervalTree()
            tree[s['query']][s['reference']].add(interval)
        return tree

    tree = {}
    tree = _add_to_tree(tree, gts, "GT")
    tree = _add_to_tree(tree, results, "RS")

    return tree


def extract_ids_from_segments(df, status, tag):
    """Extract ids from segments.
    Filter segments by status and type (tag).
    Args:
        df (DataFrame):  DataFrame
        status (str): "TP" True Positive, "FP" False Positive, "TN" True Negative, "FN" False Negative.
        tag (str): "GT" or "RS". Segment tag.
    Returns:
        ids_unique (list): list with unique ids.
    """
    ids = []
    for idx, row in df.query('status == "%s"' % status).iterrows():
        ids += [int(v.split(" ")[1]) for v in row['debug'] if v.startswith(tag)]
    ids_unique = set(ids)

    return ids_unique


def sum_seconds_from_segments(df, status, tag):
    """Sum seconds from segments.
    Filter segments by status and type (tag).
    Args:
        df (DataFrame):  DataFrame
        status (str): "TP" True Positive, "FP" False Positive, "TN" True Negative, "FN" False Negative.
        tag (str): "GT" or "RS". Segment tag.
    Returns:
        seconds (float): Summed seconds from segments.
    """
    seconds = 0
    for idx, row in df.query('status == "%s"' % status).iterrows():
        number_segments = len([l for l in row['debug'] if l.startswith(tag)])
        seconds += number_segments * (row.end - row.begin)

    return seconds


def compute_statistics(results, gts, threshold=None):
    """Compute statistics.
    Args:
        results (DataFrame): Tech results pandas DataFrame.
        gts (DataFrame): Groundtruth (annotations) pandas DataFrame.
    Returns:
        out_stats (dict): Dictionary with stats values.
    """
    def _merge_segments_data(data1, data2):
        """Merge overlapping segments in intervaltree."""
        return data1 + data2

    out_stats = {}

    # MERGE RESULTS AND GROUNDTRUTH
    mixed_tree = create_tree(gts, results, threshold)
    for query, ref_trees in mixed_tree.items():
        for ref, tree in ref_trees.items():
            tree.split_overlaps()
            tree.merge_overlaps(_merge_segments_data)

    # CLASSIFY MERGED SEGMENTS INTO TP, FP, FN...
    mixed_segments = []
    for query, ref_trees in mixed_tree.items():
        for ref, tree in ref_trees.items():
            for interval in sorted(tree.items()):
                status = "UNK"
                types_found = sorted(list(set(
                    [l.split(" ")[0] for l in interval.data])))
                if types_found == ["GT"]:
                    status = "FN"
                elif types_found == ["RS"]:
                    status = "FP"
                elif types_found == ["GT", "RS"]:
                    status = "TP"
                mixed_segments.append({
                    'query': query,
                    'ref': ref,
                    'begin': interval.begin,
                    'end': interval.end,
                    'status': status,
                    'debug': interval.data
                })
    mixed_df = pd.DataFrame(mixed_segments)
    mixed_df.sort_values(by=['query', 'ref', 'begin', 'end'], inplace=True)

    # UNIQUES
    tps_rs_unique = extract_ids_from_segments(mixed_df, "TP", "RS")
    fps_rs_unique = extract_ids_from_segments(mixed_df, "FP", "RS")
    fps_rs_unique -= tps_rs_unique

    tps_gt_unique = extract_ids_from_segments(mixed_df, "TP", "GT")
    fns_gt_unique = extract_ids_from_segments(mixed_df, "FN", "GT")
    fns_gt_unique -= tps_gt_unique

    precision = len(tps_rs_unique) / (len(tps_rs_unique) + len(fps_rs_unique))
    recall = len(tps_rs_unique) / (len(tps_rs_unique) + len(fns_gt_unique))
    f1_score = 2 * precision * recall / (precision + recall)

    out_stats['uniques'] = {
        'TP_results': len(tps_rs_unique),
        'TP_groundtruth': len(tps_gt_unique),
        'FP_results': len(fps_rs_unique),
        'FN_groundtruth': len(fns_gt_unique),
        'TP_event_ratio': len(tps_rs_unique) / len(tps_gt_unique),
        'metrics': {
            'precision': precision,
            'recall': recall,
            'f1-score': f1_score,
        },
        'check': {
            'TP_plus_FP_results': len(tps_rs_unique) + len(fps_rs_unique),
            'TP_plus_FN_groundtruth': len(tps_gt_unique) + len(fns_gt_unique),
        },
    }

    # SECONDS
    tp_seconds_with_duplicates = sum_seconds_from_segments(mixed_df, "TP",
                                                           "RS")
    fp_seconds_with_duplicates = sum_seconds_from_segments(mixed_df, "FP",
                                                           "RS")
    fn_seconds_with_duplicates = sum_seconds_from_segments(mixed_df, "FN",
                                                           "GT")
    precision_sec = tp_seconds_with_duplicates / (
        tp_seconds_with_duplicates + fp_seconds_with_duplicates)
    recall_sec = tp_seconds_with_duplicates / (
        tp_seconds_with_duplicates + fn_seconds_with_duplicates)
    f1_score_sec = 2 * precision_sec * recall_sec / (precision_sec + recall_sec)
    out_stats['seconds'] = {
        'TPs': tp_seconds_with_duplicates,
        'FPs': fp_seconds_with_duplicates,
        'FNs': fn_seconds_with_duplicates,
        'metrics': {
            'precision': precision_sec,
            'recall': recall_sec,
            'f1-score': f1_score_sec,
        },
        'check': {
            'TP_plus_FP': (tp_seconds_with_duplicates +
                           fp_seconds_with_duplicates),
            'TP_plus_FN': (tp_seconds_with_duplicates +
                           fn_seconds_with_duplicates),
        }
    }

    tp_seconds = sum([
        row.end - row.begin
        for idx, row in mixed_df.query('status == "TP"').iterrows()])
    fp_seconds = sum([
        row.end - row.begin
        for idx, row in mixed_df.query('status == "FP"').iterrows()])
    fn_seconds = sum([
        row.end - row.begin
        for idx, row in mixed_df.query('status == "FN"').iterrows()])

    overlength_seconds = 0
    for idx, row in mixed_df.query('status == "FP"').iterrows():
        rs_ids = set([int(v.split(" ")[1]) for v in row['debug']
                      if v.startswith("RS")])
        rs_ids -= tps_rs_unique
        if len(rs_ids) > 0:
            overlength_seconds += row.end - row.begin

    precision_sec_uniq = tp_seconds / (tp_seconds + fp_seconds)
    recall_sec_uniq = tp_seconds / (tp_seconds + fn_seconds)
    f1_score_sec_uniq = 2 * precision_sec_uniq * recall_sec_uniq / (
        precision_sec_uniq + recall_sec_uniq)
    out_stats['seconds_without_duplicates'] = {
        'TPs': tp_seconds,
        'FPs': fp_seconds,
        'FPs_subtypes': {
            'FPs_overlength': overlength_seconds,
            'FPs_other': fp_seconds - overlength_seconds
        },
        'FNs': fn_seconds,
        'metrics': {
            'precision': precision_sec_uniq,
            'recall': recall_sec_uniq,
            'f1-score': f1_score_sec_uniq,
        },
        'check': {
            'TP_plus_FP': tp_seconds + fp_seconds,
            'TP_plus_FN': tp_seconds + fn_seconds,
        }
    }

    return out_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Compute statistics and metrics to measure the performance "
        "of an AFP algorithm.")
    )
    parser.add_argument(
        "results_file",
        help=("Results file path. It has to be as csv with at least the following "
        "fields in the header: {query, reference, query_start, query_end, ref_start, ref_end, score}"))
    parser.add_argument(
        "annotations_file",
        help="Annotations groundtruth file path.")
    parser.add_argument(
        "-t",
        "--xtag",
        default="unanimity",
        help="Cross-Annotations tag.")
    parser.add_argument(
        "-th",
        "--threshold",
        default=None,
        type=float,
        help="Custom score threshold.")
    args = parser.parse_args()

    out_stats = {}
    out_stats['input'] = {
        'results_file': args.results_file,
        'groundtruth_file': args.annotations_file,
        'cross_annotation_tag': args.xtag
    }

    # LOAD RESULTS
    results = pd.read_csv(args.results_file)
    out_stats['results'] = {
        'num': len(results),
        'seconds': sum([r.query_end - r.query_start
                        for idx, r in results.iterrows()])
    }

    # LOAD GROUNDTRUTH
    gts = pd.read_csv(args.annotations_file)
    if args.xtag == "unanimity":
        gts = gts[gts['x_tag'].isin(['unanimity'])]
    elif args.xtag == "majority":
        gts = gts[gts['x_tag'].isin(['unanimity', 'majority'])]
    elif args.xtag == "single":
        gts = gts[gts['x_tag'].isin(['unanimity', 'majority', 'single'])]
    gts.reset_index(inplace=True, drop=True)
    out_stats['groundtruth'] = {
        'num': len(gts),
        'seconds': sum([gt.query_end - gt.query_start
                        for idx, gt in gts.iterrows()])
    }

    # COMPUTE STATISTICS
    out_stats.update(compute_statistics(results, gts, args.threshold))
    pprint.pprint(out_stats)
