import numpy as np
import math
import itertools
from collections import namedtuple
from .coco_format import CocoPairs, CocoPairsNetwork
from .human import Human
from scipy.ndimage import maximum_filter, gaussian_filter


heatmap_supress = False
heatmap_gaussian = True
adaptive_threshold = False

NMS_Threshold = 0.15
Local_PAF_Threshold = 0.2
PAF_Count_Threshold = 5
Part_Count_Threshold = 4
Part_Score_Threshold = 4.5

PartPair = namedtuple('PartPair', ['score', 'part_idx1', 'part_idx2', 'idx1', 'idx2',
                                   'coord1', 'coord2', 'score1', 'score2'], verbose=False)


def non_max_suppression(plain, window_size=3, threshold=NMS_Threshold):
    under_threshold_indices = plain < threshold
    plain[under_threshold_indices] = 0
    return plain * (plain == maximum_filter(plain, footprint=np.ones((window_size, window_size))))


def estimate(heat_mat, paf_mat):
    if heat_mat.shape[2] == 19:
        heat_mat = np.rollaxis(heat_mat, 2, 0)
    if paf_mat.shape[2] == 38:
        paf_mat = np.rollaxis(paf_mat, 2, 0)

    if heatmap_supress:
        heat_mat = heat_mat - heat_mat.min(axis=1).min(axis=1).reshape(19, 1, 1)
        heat_mat = heat_mat - heat_mat.min(axis=2).reshape(19, heat_mat.shape[1], 1)

    if heatmap_gaussian:
        heat_mat = gaussian_filter(heat_mat, sigma=0.5)

    if adaptive_threshold:
        _NMS_Threshold = max(np.average(heat_mat) * 4.0, NMS_Threshold)
        _NMS_Threshold = min(_NMS_Threshold, 0.3)
    else:
        _NMS_Threshold = NMS_Threshold

    # extract interesting coordinates using NMS.
    coords = []  # [[coords in plane1], [....], ...]
    for plain in heat_mat[:-1]:
        nms = non_max_suppression(plain, 5, _NMS_Threshold)
        coords.append(np.where(nms >= _NMS_Threshold))

    # score pairs
    pairs_by_conn = list()
    for (part_idx1, part_idx2), (paf_x_idx, paf_y_idx) in zip(CocoPairs, CocoPairsNetwork):
        pairs = score_pairs(
            part_idx1, part_idx2,
            coords[part_idx1], coords[part_idx2],
            paf_mat[paf_x_idx], paf_mat[paf_y_idx],
            heatmap=heat_mat,
            rescale=(1.0 / heat_mat.shape[2], 1.0 / heat_mat.shape[1])
        )
        pairs_by_conn.extend(pairs)

    # merge pairs to human
    # pairs_by_conn is sorted by CocoPairs(part importance) and Score between Parts.
    humans = [Human([pair]) for pair in pairs_by_conn]
    while True:
        merge_items = None
        for k1, k2 in itertools.combinations(humans, 2):
            if k1 == k2:
                continue
            if k1.is_connected(k2):
                merge_items = (k1, k2)
                break

        if merge_items is not None:
            merge_items[0].merge(merge_items[1])
            humans.remove(merge_items[1])
        else:
            break

    # reject by subset count
    humans = [human for human in humans if human.part_count() >= PAF_Count_Threshold]
    # reject by subset max score
    humans = [human for human in humans if human.get_max_score() >= Part_Score_Threshold]
    return humans


def score_pairs(part_idx1, part_idx2, coord_list1, coord_list2, paf_mat_x, paf_mat_y, heatmap, rescale=(1.0, 1.0)):
    connection_temp = []
    cnt = 0
    for idx1, (y1, x1) in enumerate(zip(coord_list1[0], coord_list1[1])):
        for idx2, (y2, x2) in enumerate(zip(coord_list2[0], coord_list2[1])):
            score, count = get_score(x1, y1, x2, y2, paf_mat_x, paf_mat_y)
            cnt += 1
            if count < PAF_Count_Threshold or score <= 0.0:
                continue
            connection_temp.append(PartPair(
                score=score,
                part_idx1=part_idx1, part_idx2=part_idx2,
                idx1=idx1, idx2=idx2,
                coord1=(x1 * rescale[0], y1 * rescale[1]),
                coord2=(x2 * rescale[0], y2 * rescale[1]),
                score1=heatmap[part_idx1][y1][x1],
                score2=heatmap[part_idx2][y2][x2],
            ))

    connection = []
    used_idx1, used_idx2 = set(), set()
    for candidate in sorted(connection_temp, key=lambda x: x.score, reverse=True):
        # check not connected
        if candidate.idx1 in used_idx1 or candidate.idx2 in used_idx2:
            continue
        connection.append(candidate)
        used_idx1.add(candidate.idx1)
        used_idx2.add(candidate.idx2)
    return connection


def get_score(x1, y1, x2, y2, paf_mat_x, paf_mat_y):
    __num_inter = 10
    __num_inter_f = float(__num_inter)
    dx, dy = x2 - x1, y2 - y1
    norm_vec = math.sqrt(dx ** 2 + dy ** 2)

    if norm_vec < 1e-4:
        return 0.0, 0
    vx, vy = dx / norm_vec, dy / norm_vec

    xs = np.arange(x1, x2, dx / __num_inter_f) if x1 != x2 else np.full((__num_inter,), x1)
    ys = np.arange(y1, y2, dy / __num_inter_f) if y1 != y2 else np.full((__num_inter,), y1)
    xs = (xs + 0.5).astype(np.int8)
    ys = (ys + 0.5).astype(np.int8)

    # without vectorization
    paf_xs = np.zeros(__num_inter)
    paf_ys = np.zeros(__num_inter)
    for idx, (mx, my) in enumerate(zip(xs, ys)):
        paf_xs[idx] = paf_mat_x[my][mx]
        paf_ys[idx] = paf_mat_y[my][mx]

    local_scores = paf_xs * vx + paf_ys * vy
    thidxs = local_scores > Local_PAF_Threshold

    return sum(local_scores * thidxs), sum(thidxs)