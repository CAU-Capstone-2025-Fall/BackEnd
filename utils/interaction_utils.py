# utils/interaction_utils.py
import itertools

"""
pairwise Shapley–Taylor interaction 계산 (2-way)
함수 이름은 기존 compute_interaction_sets 그대로 유지
"""


def compute_interaction_sets(x: dict, f, feature_groups: dict, top_k=3):
    """2-way Shapley-Taylor interaction 계산"""

    # --------------------------------
    # 1) feature group 목록 (독립 feature 그룹)
    # --------------------------------
    group_names = list(feature_groups.keys())
    G = len(group_names)

    # --------------------------------
    # 2) Helper: 특정 그룹 마스킹
    # --------------------------------
    def mask_groups(original: dict, mask_set: set):
        x_new = dict(original)
        for g in mask_set:
            for col in feature_groups[g]:
                x_new[col] = 0
        return x_new

    # baseline f(x)
    f_empty = f(x)

    # --------------------------------
    # 3) Shapley-Taylor 2-way interaction
    #
    # ST_ij = sum_{S subset (All \ {i,j})}
    #          [ f(S ∪ {i,j}) - f(S ∪ {i}) - f(S ∪ {j}) + f(S) ]
    #          / 2^(|All|-2)
    #
    # --------------------------------
    interaction_scores = []
    all_groups_set = set(group_names)

    # 모든 pair 조합
    for (g1, g2) in itertools.combinations(group_names, 2):

        remainder = list(all_groups_set - {g1, g2})
        R = len(remainder)

        total = 0.0
        normalizer = (2 ** R)

        # 모든 subset S ⊆ remainder
        for bits in range(2 ** R):
            S = set()
            for idx in range(R):
                if bits & (1 << idx):
                    S.add(remainder[idx])

            # f(S)
            f_s = f(mask_groups(x, S))

            # f(S ∪ {i})
            f_si = f(mask_groups(x, S | {g1}))

            # f(S ∪ {j})
            f_sj = f(mask_groups(x, S | {g2}))

            # f(S ∪ {i,j})
            f_sij = f(mask_groups(x, S | {g1, g2}))

            # local term
            val = f_sij - f_si - f_sj + f_s
            total += val

        score = total / normalizer
        interaction_scores.append(((g1, g2), score))

    # --------------------------------
    # 4) 정렬 후 top-K 반환
    # --------------------------------
    interaction_scores.sort(key=lambda x: abs(x[1]), reverse=True)
    return interaction_scores[:top_k]
