"""
fusionART.py — Reconstructed Fusion ART engine for ARTEM/STEM.

NOTE: The original `fusionART` module was NOT released with the ARTEM repository
("STEM cannot be fully shared"). This is a clean-room re-implementation built to
match (a) the public Fusion ART / STEM equations in Chang & Tan (IJCAI-17,
`0206.pdf`) and Tan et al. (2007/2019), and (b) the exact API surface that
`eventRetriever.py` calls:

    fa = FusionART(numspace, lengths, beta, alpha, gamma, rho)
    fa.setActivityF1(vector)        # vector = [[t], [..384..], [..384..], [..384..]]
    J = fa.resSearch()              # winner-take-all + vigilance search (+ recruit)
    fa.autoLearn(J)                 # template learning on node J
    fa.codes                        # list of nodes; node['weights'][k]
    fa.uncommitted(j)               # bool
    fa.matchValField[k](in_k, w_k)  # per-channel match  m = |x^k ^ w^k| / |x^k|

Encoding uses fuzzy ART with complement coding per channel (eq. 1-4 of 0206.pdf):
    choice : T_j = sum_k gamma_k * |x^k ^ w_j^k| / (alpha_k + |w_j^k|)        (1)
    select : J = argmax_j T_j                                                  (2)
    match  : m_j^k = |x^k ^ w_j^k| / |x^k|  >= rho_k  (resonance)              (3)
    learn  : w_j^k(new) = (1-beta_k) w_j^k + beta_k (x^k ^ w_j^k)              (4)
where ^ is fuzzy-AND (element-wise min) and |.| is the L1 norm.
"""

import numpy as np


def complement_code(vec):
    """x -> [x, 1-x].  Assumes values already normalized to [0, 1]."""
    v = np.asarray(vec, dtype=float)
    return np.concatenate([v, 1.0 - v])


def _fuzzy_and(a, b):
    return np.minimum(a, b)


class FusionART:
    def __init__(self, numspace, lengths, beta, alpha, gamma, rho):
        self.numspace = numspace
        self.lengths = list(lengths)
        self.beta = list(beta)
        self.alpha = list(alpha)
        self.gamma = list(gamma)
        self.rho = list(rho)

        # F1 activity per channel (raw, complement-coded on demand)
        self.activityF1 = [None] * numspace

        # committed + (trailing) uncommitted category nodes
        self.codes = []

        # per-channel match / choice functions (overridable, see ARTxtralib)
        self.matchValField = [self._make_match(k) for k in range(numspace)]
        self.choiceActField = [self._make_choice(k) for k in range(numspace)]

        # ART always keeps one uncommitted node available for recruitment
        self._add_uncommitted()

    # ---- node bookkeeping -------------------------------------------------
    def _cc_len(self, k):
        return 2 * self.lengths[k]

    def _add_uncommitted(self):
        weights = [np.ones(self._cc_len(k)) for k in range(self.numspace)]
        self.codes.append({"weights": weights, "committed": False})

    def uncommitted(self, j):
        return not self.codes[j].get("committed", False)

    # ---- per-channel functions (fuzzy AC model) --------------------------
    def _make_match(self, k):
        def match(input_k, weight_k):
            x = complement_code(input_k)
            w = np.asarray(weight_k, dtype=float)
            xn = x.sum()
            if xn == 0:
                return 0.0
            return float(_fuzzy_and(x, w).sum() / xn)
        return match

    def _make_choice(self, k):
        def choice(input_k, weight_k):
            x = complement_code(input_k)
            w = np.asarray(weight_k, dtype=float)
            return float(_fuzzy_and(x, w).sum() / (self.alpha[k] + w.sum()))
        return choice

    # ---- F1 / search / learn ---------------------------------------------
    def setActivityF1(self, vector):
        self.activityF1 = [list(v) for v in vector]

    def _choice_value(self, j):
        T = 0.0
        for k in range(self.numspace):
            if self.gamma[k] <= 0:
                continue
            T += self.gamma[k] * self.choiceActField[k](
                self.activityF1[k], self.codes[j]["weights"][k]
            )
        return T

    def _resonates(self, j):
        for k in range(self.numspace):
            if self.gamma[k] <= 0:
                continue
            m = self.matchValField[k](self.activityF1[k], self.codes[j]["weights"][k])
            if m < self.rho[k]:
                return False
        return True

    def resSearch(self):
        """Winner-take-all by choice; vigilance test; reset until resonance.
        Returns the index J of the resonating node (recruits one if needed)."""
        order = sorted(range(len(self.codes)),
                       key=lambda j: self._choice_value(j), reverse=True)
        for j in order:
            if self.uncommitted(j):
                return j                 # uncommitted node always resonates
            if self._resonates(j):
                return j
        self._add_uncommitted()
        return len(self.codes) - 1

    def autoLearn(self, J):
        """Template learning (eq. 4). Commits node J and keeps a fresh
        uncommitted node available."""
        for k in range(self.numspace):
            x = complement_code(self.activityF1[k])
            w = np.asarray(self.codes[J]["weights"][k], dtype=float)
            self.codes[J]["weights"][k] = (1.0 - self.beta[k]) * w + \
                self.beta[k] * _fuzzy_and(x, w)
        if self.uncommitted(J):
            self.codes[J]["committed"] = True
            self._add_uncommitted()
        return J
