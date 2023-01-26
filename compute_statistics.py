""" Measure the performance of AFP algorithms.
Compute statistics and metrics to measure the performance of an AFP algorithm.
This code has been used in the publication 'BAF: An Audio Fingerprinting Dataset
for Broadcast Monitoring' by Guillem Cortès, Alex Ciurana, Emilio Molina, Marius
Miron, Owen Meyers, Joren Six and Xavier Serra.
"""
import argparse
import pprint
import pandas as pd
import intervaltree as itree
import math


class UnknownTagError(Exception):
    """Unknown Tag. It must be either 'GT' (GroundTruth) or 'RS' (Results)"""

    pass


def create_tree(groundtruths, results, threshold=None):
    """Creates intervaltree with Groundtruth and Tech results.
    Args:
        results (DataFrame): Tech results pandas DataFrame.
        groundtruths (DataFrame): Groundtruth (annotations) pandas DataFrame.
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
        for idx, seg in segments.iterrows():
            if (seg["query_end"] - seg["query_start"]) <= 0:
                # intervaltree doesn't allow 0-length segments
                continue
            if threshold is not None and tag == "RS" and seg["score"] < threshold:
                continue
            if tag == "GT":
                if math.isnan(seg["snr"]):
                    interval = itree.Interval(
                        seg["query_start"],
                        seg["query_end"],
                        [f"{tag} {idx} SNR={math.nan}"],
                    )
                else:
                    interval = itree.Interval(
                        seg["query_start"],
                        seg["query_end"],
                        [f"{tag} {idx} SNR={seg['snr']}"],
                    )
            elif tag == "RS":
                interval = itree.Interval(
                    seg["query_start"], seg["query_end"], [f"{tag} {idx}"]
                )
            else:
                raise UnknownTagError(f"Unkown tag {tag}")

            if seg["query"] not in tree:
                tree[seg["query"]] = {}
            if seg["reference"] not in tree[seg["query"]]:
                tree[seg["query"]][seg["reference"]] = itree.IntervalTree()
            tree[seg["query"]][seg["reference"]].add(interval)
        return tree

    tree = {}
    tree = _add_to_tree(tree, groundtruths, "GT")
    tree = _add_to_tree(tree, results, "RS")

    return tree


def extract_ids_from_segments(segments, status, tag):
    """Extract ids from segments with specific status and type (tag).
    Args:
        segments (DataFrame):  DataFrame
        status (str): "TP" True Positive, "FP" False Positive,
                      "TN" True Negative, "FN" False Negative.
        tag (str): "GT" or "RS". Segment tag.
    Returns:
        ids_unique (list): list with unique ids.
    """
    ids = []
    for _, row in segments.query(f'status == "{status}"').iterrows():
        ids += [int(v.split(" ")[1]) for v in row["debug"] if v.startswith(tag)]
    ids_unique = set(ids)

    return ids_unique


def sum_seconds_from_segments(segments, status, tag):
    """Sum seconds from segments.
    Filtering segments by status and type (tag). This method counts duplicate
    matches.
    Args:
        segments (DataFrame):  DataFrame
        status (str): "TP" True Positive, "FP" False Positive,
                      "TN" True Negative, "FN" False Negative.
        tag (str): "GT" or "RS". Segment tag.
    Returns:
        seconds (float): Summed seconds from segments.
    """
    seconds = 0
    for _, row in segments.query(f'status == "{status}"').iterrows():
        number_segments = len([l for l in row["debug"] if l.startswith(tag)])
        seconds += number_segments * (row.end - row.begin)

    return seconds


def compute_statistics(results, groundtruths, threshold=None):
    """Compute statistics.
    Args:
        results (DataFrame): Tech results pandas DataFrame.
        groundtruths (DataFrame): Groundtruth (annotations) pandas DataFrame.
    Returns:
        out_stats (dict): Dictionary with stats values.
    """

    def _merge_segments_data(data1, data2):
        """Merge overlapping segments in intervaltree."""
        return data1 + data2

    out_stats = {}

    # MERGE RESULTS AND GROUNDTRUTH
    mixed_tree = create_tree(groundtruths, results, threshold)
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
                # types_found = sorted(
                #     list(set([l.split(" ")[0] for l in interval.data]))
                # )
                types = []
                for l in interval.data:
                    line = l.split(" ")
                    types.append(line[0])
                    if line[0] == "GT" and len(line) == 3:
                        snr = line[2].split("SNR=")[1]
                types_found = sorted(list(set(types)))
                if types_found == ["GT"]:
                    status = "FN"
                elif types_found == ["RS"]:
                    status = "FP"
                    snr = math.nan
                elif types_found == ["GT", "RS"]:
                    status = "TP"
                mixed_segments.append(
                    {
                        "query": query,
                        "ref": ref,
                        "begin": interval.begin,
                        "end": interval.end,
                        "status": status,
                        "snr": snr,
                        "debug": interval.data,
                    }
                )
    mixed_df = pd.DataFrame(mixed_segments)
    mixed_df.sort_values(by=["query", "ref", "begin", "end"], inplace=True)

    # UNIQUES
    tps_rs_unique = extract_ids_from_segments(mixed_df, "TP", "RS")
    fps_rs_unique = extract_ids_from_segments(mixed_df, "FP", "RS")
    fps_rs_unique -= tps_rs_unique

    tps_gt_unique = extract_ids_from_segments(mixed_df, "TP", "GT")
    fns_gt_unique = extract_ids_from_segments(mixed_df, "FN", "GT")
    fns_gt_unique -= tps_gt_unique

    # Results segments precision
    precision = len(tps_rs_unique) / (len(tps_rs_unique) + len(fps_rs_unique))
    # GroundTruth segments recall
    recall_gt = len(tps_gt_unique) / (len(tps_gt_unique) + len(fns_gt_unique))

    out_stats["uniques"] = {
        "TP_results": len(tps_rs_unique),
        "TP_groundtruth": len(tps_gt_unique),
        "FP_results": len(fps_rs_unique),
        "FN_groundtruth": len(fns_gt_unique),
        "TP_match_ratio": len(tps_rs_unique) / len(tps_gt_unique),
        "metrics": {"precision": precision, "recall_GT": recall_gt},
        "check": {
            "TP_plus_FP_results": len(tps_rs_unique) + len(fps_rs_unique),  # results
            "TP_plus_FN_groundtruth": len(tps_gt_unique) + len(fns_gt_unique),  # GT
        },
    }

    # SECONDS
    tp_seconds_with_duplicates = sum_seconds_from_segments(mixed_df, "TP", "RS")
    fp_seconds_with_duplicates = sum_seconds_from_segments(mixed_df, "FP", "RS")
    fn_seconds_with_duplicates = sum_seconds_from_segments(mixed_df, "FN", "GT")
    precision_sec = tp_seconds_with_duplicates / (
        tp_seconds_with_duplicates + fp_seconds_with_duplicates
    )
    recall_sec = tp_seconds_with_duplicates / (
        tp_seconds_with_duplicates + fn_seconds_with_duplicates
    )
    f1_score_sec = 2 * precision_sec * recall_sec / (precision_sec + recall_sec)
    out_stats["seconds"] = {
        "TPs": tp_seconds_with_duplicates,
        "FPs": fp_seconds_with_duplicates,
        "FNs": fn_seconds_with_duplicates,
        "metrics": {
            "precision": precision_sec,
            "recall": recall_sec,
            "f1-score": f1_score_sec,
        },
        "check": {
            "TP_plus_FP": (tp_seconds_with_duplicates + fp_seconds_with_duplicates),
            "TP_plus_FN": (tp_seconds_with_duplicates + fn_seconds_with_duplicates),
        },
    }

    tp_seconds = sum(
        [row.end - row.begin for _, row in mixed_df.query('status == "TP"').iterrows()]
    )
    fp_seconds = sum(
        [row.end - row.begin for _, row in mixed_df.query('status == "FP"').iterrows()]
    )
    fn_seconds = sum(
        [row.end - row.begin for _, row in mixed_df.query('status == "FN"').iterrows()]
    )

    overlength_seconds = 0
    for _, row in mixed_df.query('status == "FP"').iterrows():
        rs_ids = set([int(v.split(" ")[1]) for v in row["debug"] if v.startswith("RS")])
        rs_ids -= tps_rs_unique
        if len(rs_ids) > 0:
            overlength_seconds += row.end - row.begin

    precision_sec_uniq = tp_seconds / (tp_seconds + fp_seconds)
    recall_sec_uniq = tp_seconds / (tp_seconds + fn_seconds)
    f1_score_sec_uniq = (
        2
        * precision_sec_uniq
        * recall_sec_uniq
        / (precision_sec_uniq + recall_sec_uniq)
    )
    out_stats["seconds_without_duplicates"] = {
        "TPs": tp_seconds,
        "FPs": fp_seconds,
        "FPs_subtypes": {
            "FPs_overlength": overlength_seconds,
            "FPs_other": fp_seconds - overlength_seconds,
        },
        "FNs": fn_seconds,
        "metrics": {
            "precision": precision_sec_uniq,
            "recall": recall_sec_uniq,
            "f1-score": f1_score_sec_uniq,
        },
        "check": {
            "TP_plus_FP": tp_seconds + fp_seconds,
            "TP_plus_FN": tp_seconds + fn_seconds,
        },
    }

    return out_stats, mixed_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Compute statistics and metrics to measure the performance "
            "of an AFP algorithm."
        )
    )
    parser.add_argument(
        "results_file",
        help=(
            "Results file path. It has to be as csv with at least the following "
            "fields in the header: {query, reference, query_start, query_end,"
            "ref_start, ref_end, score}"
        ),
    )
    parser.add_argument("annotations_file", help="Annotations groundtruth file path.")
    parser.add_argument(
        "-o", "--out_csv", default=None, help="Output csv file path with the stasts."
    )
    parser.add_argument(
        "-t", "--xtag", default="unanimity", help="Cross-Annotations tag."
    )
    parser.add_argument(
        "-th", "--threshold", default=None, type=float, help="Custom score threshold."
    )
    args = parser.parse_args()

    stats = {}
    stats["input"] = {
        "results_file": args.results_file,
        "groundtruth_file": args.annotations_file,
        "cross_annotation_tag": args.xtag,
    }

    # LOAD RESULTS
    res = pd.read_csv(args.results_file)
    stats["results"] = {
        "num": len(res),
        "seconds": sum([r.query_end - r.query_start for _, r in res.iterrows()]),
    }

    # LOAD GROUNDTRUTH
    gts = pd.read_csv(args.annotations_file)
    if args.xtag == "unanimity":
        gts = gts[gts["x_tag"].isin(["unanimity"])]
    elif args.xtag == "majority":
        gts = gts[gts["x_tag"].isin(["unanimity", "majority"])]
    elif args.xtag == "single":
        gts = gts[gts["x_tag"].isin(["unanimity", "majority", "single"])]
    gts.reset_index(inplace=True, drop=True)
    stats["groundtruth"] = {
        "num": len(gts),
        "seconds": sum([gt.query_end - gt.query_start for _, gt in gts.iterrows()]),
    }

    # COMPUTE STATISTICS
    out_stats, stats_df = compute_statistics(res, gts, args.threshold)
    stats.update(out_stats)
    pprint.pprint(stats)

    if args.out_csv is not None:
        stats_df.to_csv(args.out_csv, index=False)
