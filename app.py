"""
Change Point Detection Dashboard - v6.1 (single-file Streamlit app)
Corrected collapsed RJMCMC vs official changeforest.
Fixes the ps_single/ps_full session_state key collision (v6 bug).
"""
from __future__ import annotations

import hashlib
import io
import math
import os
import time
from dataclasses import dataclass, replace, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.special import betaln, gammaln
from sklearn.ensemble import RandomForestClassifier

try:
    from changeforest import changeforest as _changeforest, Control as _Control
    CHANGEFOREST_AVAILABLE, CHANGEFOREST_ERROR = True, ""
except Exception as exc:
    CHANGEFOREST_AVAILABLE, CHANGEFOREST_ERROR = False, str(exc)
    _changeforest = _Control = None

st.set_page_config(page_title="Change Point Detection Dashboard",
                   layout="wide", page_icon="\U0001F4C8")

COL_RJ, COL_CF, COL_BL, COL_TRUE, COL_SERIES = "#2C7FB8", "#d62728", "#7f7f7f", "#2ca02c", "#9a9890"


@dataclass(frozen=True)
class PriorConfig:
    a_p: float = 1.0
    b_p: float = 1.0
    p_geom: float = 1.0 / 70.0
    r_dispersion: Optional[float] = None
    min_segment: int = 7
    max_changepoints: Optional[int] = None

    def validate(self, n: int) -> None:
        if self.a_p <= 0 or self.b_p <= 0:
            raise ValueError("a_p and b_p must be positive.")
        if not 0 < self.p_geom < 1:
            raise ValueError("p_geom must lie strictly between 0 and 1.")
        if self.r_dispersion is not None and self.r_dispersion <= 0:
            raise ValueError("r_dispersion must be positive or None.")
        if self.min_segment < 1 or 2 * self.min_segment > n:
            raise ValueError("min_segment is incompatible with the series length.")
        if self.max_changepoints is not None and self.max_changepoints < 0:
            raise ValueError("max_changepoints must be non-negative or None.")


@dataclass(frozen=True)
class SamplerConfig:
    n_warmup: int = 4_000
    n_samples: int = 8_000
    thin: int = 1
    seed: int = 0
    move_weights: Tuple[float, float, float] = (0.35, 0.35, 0.30)
    target_accept: float = 0.30
    adapt_rate: float = 0.7

    def validate(self) -> None:
        if self.n_warmup < 0 or self.n_samples < 1 or self.thin < 1:
            raise ValueError("Invalid warm-up, sample, or thinning value.")
        if len(self.move_weights) != 3 or any(w <= 0 for w in self.move_weights):
            raise ValueError("move_weights must contain three positive values.")
        if not 0 < self.target_accept < 1 or self.adapt_rate <= 0:
            raise ValueError("Invalid adaptation settings.")


def validate_counts(y: Sequence[float]) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or len(y) < 4 or not np.all(np.isfinite(y)):
        raise ValueError("Series must be finite, one-dimensional and length >= 4.")
    if np.any(y < 0):
        raise ValueError("Negative-Binomial observations must be non-negative.")
    if not np.allclose(y, np.round(y)):
        raise ValueError("Counts must be integers. Aggregate before analysis.")
    return np.round(y).astype(np.int64)


def estimate_global_dispersion(y: np.ndarray) -> float:
    mu = float(np.mean(y))
    var = float(np.var(y, ddof=1)) if len(y) > 1 else mu
    if mu <= 0 or var <= mu:
        return 1.0e6
    return max(mu * mu / (var - mu), 0.05)


def prefix_statistics(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return np.concatenate(([0.0], np.cumsum(y, dtype=float))), y


def segment_log_marginal(start, stop, sum_y, y, r, prior) -> float:
    m = stop - start
    if m < prior.min_segment:
        return -np.inf
    s = float(sum_y[stop] - sum_y[start])
    obs = y[start:stop]
    combinatorial = float(np.sum(gammaln(obs + r) - gammaln(r) - gammaln(obs + 1.0)))
    integrated_p = betaln(prior.a_p + m * r, prior.b_p + s) - betaln(prior.a_p, prior.b_p)
    return combinatorial + float(integrated_p)


def segmentation_log_prior(cps: Tuple[int, ...], n: int, prior: PriorConfig) -> float:
    bounds = (0,) + cps + (n,)
    if np.any(np.diff(bounds) < prior.min_segment):
        return -np.inf
    k = len(cps)
    if prior.max_changepoints is not None and k > prior.max_changepoints:
        return -np.inf
    return k * math.log(prior.p_geom) + (n - k - 1) * math.log1p(-prior.p_geom)


def collapsed_log_posterior(cps, n, sum_y, y, r, prior) -> float:
    lp = segmentation_log_prior(cps, n, prior)
    if not np.isfinite(lp):
        return -np.inf
    bounds = (0,) + cps + (n,)
    for start, stop in zip(bounds[:-1], bounds[1:]):
        lp += segment_log_marginal(start, stop, sum_y, y, r, prior)
    return float(lp)


def birth_candidates(cps, n, min_segment) -> Tuple[int, ...]:
    bounds = (0,) + cps + (n,)
    out: List[int] = []
    for left, right in zip(bounds[:-1], bounds[1:]):
        out.extend(range(left + min_segment, right - min_segment + 1))
    return tuple(out)


def shift_pairs(cps, n, min_segment) -> Tuple[Tuple[int, int], ...]:
    pairs: List[Tuple[int, int]] = []
    for i, old in enumerate(cps):
        left = cps[i - 1] if i > 0 else 0
        right = cps[i + 1] if i + 1 < len(cps) else n
        for new in range(left + min_segment, right - min_segment + 1):
            if new != old:
                pairs.append((old, new))
    return tuple(pairs)


def feasible_moves(cps, n, prior) -> Dict[str, object]:
    births = birth_candidates(cps, n, prior.min_segment)
    if prior.max_changepoints is not None and len(cps) >= prior.max_changepoints:
        births = tuple()
    return {"birth": births, "death": cps, "shift": shift_pairs(cps, n, prior.min_segment)}


def normalized_move_probabilities(feasible, log_w) -> Dict[str, float]:
    names = [nm for nm in ("birth", "death", "shift") if len(feasible[nm]) > 0]
    if not names:
        raise RuntimeError("No feasible proposal from the current state.")
    v = np.array([log_w[nm] for nm in names], dtype=float)
    v -= v.max()
    w = np.exp(v)
    return dict(zip(names, w / w.sum()))


def proposal_log_probability(old, new, move, n, prior, log_w) -> float:
    feasible = feasible_moves(old, n, prior)
    probs = normalized_move_probabilities(feasible, log_w)
    if move not in probs:
        return -np.inf
    if move == "birth":
        added, legal = set(new) - set(old), feasible["birth"]
        if len(added) != 1 or next(iter(added)) not in legal:
            return -np.inf
        return math.log(probs[move]) - math.log(len(legal))
    if move == "death":
        removed, legal = set(old) - set(new), feasible["death"]
        if len(removed) != 1 or next(iter(removed)) not in legal:
            return -np.inf
        return math.log(probs[move]) - math.log(len(legal))
    removed, added = set(old) - set(new), set(new) - set(old)
    pair = (next(iter(removed)), next(iter(added))) if len(removed) == len(added) == 1 else None
    legal = feasible["shift"]
    if pair not in legal:
        return -np.inf
    return math.log(probs[move]) - math.log(len(legal))


def propose(cps, n, prior, log_w, rng):
    feasible = feasible_moves(cps, n, prior)
    probs = normalized_move_probabilities(feasible, log_w)
    names = tuple(probs)
    move = str(rng.choice(names, p=[probs[nm] for nm in names]))
    if move == "birth":
        loc = int(rng.choice(feasible[move]))
        proposed, reverse = tuple(sorted(cps + (loc,))), "death"
    elif move == "death":
        loc = int(rng.choice(feasible[move]))
        proposed, reverse = tuple(cp for cp in cps if cp != loc), "birth"
    else:
        old, new = feasible[move][int(rng.integers(len(feasible[move])))]
        proposed, reverse = tuple(sorted(new if cp == old else cp for cp in cps)), "shift"
    lqf = proposal_log_probability(cps, proposed, move, n, prior, log_w)
    lqr = proposal_log_probability(proposed, cps, reverse, n, prior, log_w)
    return move, reverse, proposed, lqf, lqr


def run_collapsed_rjmcmc(y, prior=None, sampler=None) -> Dict[str, object]:
    y = validate_counts(y)
    n = len(y)
    prior = prior or PriorConfig()
    sampler = sampler or SamplerConfig()
    prior.validate(n)
    sampler.validate()
    rng = np.random.default_rng(sampler.seed)
    r = prior.r_dispersion if prior.r_dispersion is not None else estimate_global_dispersion(y)
    sum_y, cache = prefix_statistics(y)

    names = ("birth", "death", "shift")
    log_w = {nm: math.log(w) for nm, w in zip(names, sampler.move_weights)}
    current: Tuple[int, ...] = tuple()
    current_lp = collapsed_log_posterior(current, n, sum_y, cache, r, prior)
    accepted = {nm: 0 for nm in names}
    attempted = {nm: 0 for nm in names}
    retained: List[Tuple[int, ...]] = []
    k_trace: List[int] = []
    lp_trace: List[float] = []

    total = sampler.n_warmup + sampler.n_samples * sampler.thin
    for it in range(total):
        move, reverse, proposed, lqf, lqr = propose(current, n, prior, log_w, rng)
        proposed_lp = collapsed_log_posterior(proposed, n, sum_y, cache, r, prior)
        log_alpha = proposed_lp - current_lp + lqr - lqf
        did_accept = math.log(rng.random()) < min(0.0, log_alpha)
        if did_accept:
            current, current_lp = proposed, proposed_lp
        if it < sampler.n_warmup:
            eta = sampler.adapt_rate / math.sqrt(it + 1.0)
            log_w[move] += eta * (float(did_accept) - sampler.target_accept)
            centre = sum(log_w.values()) / len(log_w)
            log_w = {nm: v - centre for nm, v in log_w.items()}
        else:
            attempted[move] += 1
            accepted[move] += int(did_accept)
            if (it - sampler.n_warmup) % sampler.thin == 0:
                retained.append(current)
                k_trace.append(len(current))
                lp_trace.append(current_lp)

    cp_counts = np.zeros(n - 1, dtype=float)
    for cps in retained:
        for cp in cps:
            cp_counts[cp - 1] += 1.0
    return {
        "posterior_probability": cp_counts / len(retained),
        "samples": retained,
        "k_trace": np.asarray(k_trace, dtype=int),
        "logpost_trace": np.asarray(lp_trace, dtype=float),
        "r_used": float(r), "prior": prior, "sampler": sampler,
        "acceptance_rate": {nm: accepted[nm] / attempted[nm] if attempted[nm] else np.nan
                            for nm in names},
    }


def posterior_changepoints(result, threshold=0.20, cluster_gap=3):
    if not 0 < threshold < 1:
        raise ValueError("threshold must lie between 0 and 1.")
    probs = np.asarray(result["posterior_probability"])
    raw = (np.where(probs >= threshold)[0] + 1).tolist()
    if not raw:
        return [], []
    clusters: List[List[int]] = [[raw[0]]]
    for cp in raw[1:]:
        if cp - clusters[-1][-1] <= cluster_gap:
            clusters[-1].append(cp)
        else:
            clusters.append([cp])
    points = [max(c, key=lambda cp: probs[cp - 1]) for c in clusters]
    return points, [float(probs[cp - 1]) for cp in points]


def split_rhat(chains) -> float:
    arrays = [np.asarray(c, dtype=float) for c in chains]
    length = min(map(len, arrays))
    half = length // 2
    if half < 2:
        return float("nan")
    split = np.vstack([p for c in arrays for p in (c[:half], c[-half:])])
    m, nn = split.shape
    between = nn * np.var(np.mean(split, axis=1), ddof=1)
    within = np.mean(np.var(split, axis=1, ddof=1))
    if within == 0:
        return 1.0 if between == 0 else float("inf")
    variance = ((nn - 1) / nn) * within + between / nn
    return float(np.sqrt(variance / within))


def effective_sample_size(x) -> float:
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 10 or np.var(x) == 0:
        return float(n)
    x = x - x.mean()
    f = np.fft.rfft(x, n=2 * n)
    acov = np.fft.irfft(f * np.conjugate(f))[:n] / n
    rho = acov / acov[0]
    total, t = 0.0, 1
    while t + 1 < n:
        pair = rho[t] + rho[t + 1]
        if pair <= 0:
            break
        total += pair
        t += 2
    return float(n / (1.0 + 2.0 * total))


def run_multiple_chains(y, prior=None, sampler=None, n_chains=4):
    sampler = sampler or SamplerConfig()
    results = [run_collapsed_rjmcmc(y, prior, replace(sampler, seed=sampler.seed + 10_007 * c))
               for c in range(n_chains)]
    return {
        "chains": results,
        "posterior_probability": np.mean([r["posterior_probability"] for r in results], axis=0),
        "k_rhat": split_rhat([r["k_trace"] for r in results]),
        "k_ess": float(np.sum([effective_sample_size(r["k_trace"]) for r in results])),
        "r_used": results[0]["r_used"],
        "acceptance_rate": results[0]["acceptance_rate"],
    }


def enumerate_valid_segmentations(n, prior) -> List[Tuple[int, ...]]:
    states: List[Tuple[int, ...]] = []
    def visit(last, cps):
        if n - last >= prior.min_segment:
            states.append(cps)
        for cp in range(last + prior.min_segment, n - prior.min_segment + 1):
            if prior.max_changepoints is None or len(cps) < prior.max_changepoints:
                visit(cp, cps + (cp,))
    visit(0, tuple())
    return sorted(set(states), key=lambda x: (len(x), x))


def verify_detailed_balance() -> Dict[str, float]:
    y = np.array([0, 1, 0, 2, 1, 0, 3, 1, 0, 1, 0, 2], dtype=int)
    n = len(y)
    prior = PriorConfig(p_geom=1 / 5, min_segment=2, r_dispersion=2.0)
    sum_y, cache = prefix_statistics(y)
    w = {"birth": math.log(0.35), "death": math.log(0.35), "shift": math.log(0.30)}
    states = enumerate_valid_segmentations(n, prior)
    log_pi = {s: collapsed_log_posterior(s, n, sum_y, cache, 2.0, prior) for s in states}
    max_error, checked = 0.0, 0
    for old in states:
        for new in states:
            d = len(new) - len(old)
            if d == 1 and set(old).issubset(new):
                fwd, rev = "birth", "death"
            elif d == -1 and set(new).issubset(old):
                fwd, rev = "death", "birth"
            elif d == 0 and len(set(old) - set(new)) == len(set(new) - set(old)) == 1:
                fwd = rev = "shift"
            else:
                continue
            lqf = proposal_log_probability(old, new, fwd, n, prior, w)
            lqr = proposal_log_probability(new, old, rev, n, prior, w)
            if not (np.isfinite(lqf) and np.isfinite(lqr)):
                continue
            lr = log_pi[new] - log_pi[old] + lqr - lqf
            ff = log_pi[old] + lqf + min(0.0, lr)
            fr = log_pi[new] + lqr + min(0.0, -lr)
            max_error = max(max_error, abs(ff - fr))
            checked += 1
    return {"pairs_checked": checked, "max_log_flux_error": max_error,
            "passed": checked > 0 and max_error <= 1e-10}


@dataclass(frozen=True)
class ChangeforestConfig:
    method: str = "random_forest"
    segmentation: str = "bs"
    model_selection_alpha: float = 0.01
    model_selection_n_permutations: int = 199
    minimal_relative_segment_length: float = 0.05
    random_forest_n_estimators: int = 200
    random_forest_max_depth: Optional[int] = None
    seed: int = 0

    def validate(self) -> None:
        if self.method not in {"random_forest", "knn", "change_in_mean"}:
            raise ValueError(f"Unknown method: {self.method}")
        if self.segmentation not in {"bs", "sbs", "wbs"}:
            raise ValueError(f"Unknown segmentation: {self.segmentation}")
        if not 0 < self.model_selection_alpha < 1:
            raise ValueError("model_selection_alpha must lie in (0, 1).")
        if not 0 < self.minimal_relative_segment_length < 0.5:
            raise ValueError("minimal_relative_segment_length must lie in (0, 0.5).")
        if self.model_selection_n_permutations < 19:
            raise ValueError("Too few permutations for a usable p-value.")

    def to_control(self):
        kwargs = dict(model_selection_alpha=self.model_selection_alpha,
                      model_selection_n_permutations=self.model_selection_n_permutations,
                      minimal_relative_segment_length=self.minimal_relative_segment_length,
                      random_forest_n_estimators=self.random_forest_n_estimators,
                      seed=self.seed)
        if self.random_forest_max_depth is not None:
            kwargs["random_forest_max_depth"] = self.random_forest_max_depth
        return _Control(**kwargs)


def prepare_design_matrix(y: Sequence[float], transform: str = "none") -> np.ndarray:
    arr = np.asarray(y, dtype=float)
    if arr.ndim != 1:
        raise ValueError("Series must be one-dimensional.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Series contains non-finite values.")
    if transform == "anscombe":
        if np.any(arr < 0):
            raise ValueError("Anscombe transform requires non-negative counts.")
        arr = 2.0 * np.sqrt(arr + 0.375)
    elif transform == "log1p":
        if np.any(arr < -1):
            raise ValueError("log1p transform requires values above -1.")
        arr = np.log1p(arr)
    elif transform != "none":
        raise ValueError(f"Unknown transform: {transform}")
    return arr.reshape(-1, 1)


def _collect_significant_nodes(node) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    def visit(cur):
        if cur is None:
            return
        if getattr(cur, "is_significant", False):
            out.append({"start": cur.start, "stop": cur.stop, "split": cur.best_split,
                        "gain": float(cur.max_gain), "p_value": float(cur.p_value)})
        visit(getattr(cur, "left", None))
        visit(getattr(cur, "right", None))
    visit(node)
    return out


def detect_changeforest(y, config: Optional[ChangeforestConfig] = None,
                        transform: str = "none") -> Dict[str, object]:
    if not CHANGEFOREST_AVAILABLE:
        raise ImportError("changeforest is not installed. Run: pip install changeforest")
    config = config or ChangeforestConfig()
    config.validate()
    X = prepare_design_matrix(y, transform=transform)
    if len(X) < 20:
        return {"points": [], "p_values": [], "gains": [], "runtime_s": 0.0,
                "config": asdict(config), "n": len(X), "tree": None}
    t0 = time.time()
    result = _changeforest(X, config.method, config.segmentation, config.to_control())
    runtime = time.time() - t0
    points = list(result.split_points())
    lookup = {d["split"]: d for d in _collect_significant_nodes(result)}
    return {"points": points,
            "p_values": [lookup.get(p, {}).get("p_value", np.nan) for p in points],
            "gains": [lookup.get(p, {}).get("gain", np.nan) for p in points],
            "runtime_s": runtime, "tree": result, "config": asdict(config), "n": len(X)}


def changeforest_segmentation_table(result: Dict[str, object]) -> pd.DataFrame:
    if not result.get("tree"):
        return pd.DataFrame(columns=["start", "stop", "split", "gain", "p_value"])
    return pd.DataFrame(_collect_significant_nodes(result["tree"]))


def detect_rolling_rf_baseline(y, window: int = 7, n_estimators: int = 200,
                               z_threshold: float = 1.5,
                               probability_threshold: float = 0.5,
                               cluster_gap: int = 3, seed: int = 0) -> Dict[str, object]:
    arr = np.asarray(y, dtype=float)
    n = len(arr)
    if n < 10:
        return {"points": [], "confidences": [], "runtime_s": 0.0}
    t0 = time.time()
    features = np.zeros((n, 6))
    for i in range(n):
        past = arr[max(0, i - window):i] if i > 0 else arr[i:i + 1]
        future = arr[i:min(n, i + window)]
        features[i] = [past.mean(), past.std() if len(past) > 1 else 0.0,
                       future.mean(), future.std() if len(future) > 1 else 0.0,
                       future.mean() - past.mean(), arr[i]]
    shifts = features[:, 4]
    scale = shifts.std() or 1.0
    labels = (np.abs(shifts) > z_threshold * scale).astype(int)
    if labels.sum() in (0, n):
        return {"points": [], "confidences": [], "runtime_s": time.time() - t0}
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=5,
                                 random_state=seed, class_weight="balanced")
    clf.fit(features, labels)
    probs = clf.predict_proba(features)[:, 1]
    flagged = np.where(probs > probability_threshold)[0].tolist()
    if not flagged:
        return {"points": [], "confidences": [], "runtime_s": time.time() - t0}
    clusters: List[List[int]] = [[flagged[0]]]
    for idx in flagged[1:]:
        if idx - clusters[-1][-1] <= cluster_gap:
            clusters[-1].append(idx)
        else:
            clusters.append([idx])
    points = [max(c, key=lambda i: probs[i]) for c in clusters]
    return {"points": points, "confidences": [float(probs[i]) for i in points],
            "runtime_s": time.time() - t0}


def match_detections(detected, truth, tolerance: int = 7) -> Dict[str, object]:
    remaining = list(truth)
    matched: List[Tuple[int, int]] = []
    false_positives: List[int] = []
    for d in sorted(detected):
        cands = [t for t in remaining if abs(d - t) <= tolerance]
        if cands:
            best = min(cands, key=lambda t: abs(d - t))
            remaining.remove(best)
            matched.append((d, best))
        else:
            false_positives.append(d)
    tp, fp, fn = len(matched), len(false_positives), len(remaining)
    precision = tp / (tp + fp) if (tp + fp) else (1.0 if fn == 0 else 0.0)
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall,
            "f1": f1, "matched": matched, "false_positives": false_positives,
            "missed": remaining}


N_DAYS, SEG = 365, 365 // 3

SCENARIOS: Dict[str, str] = {
    "A_large_mean_shift":  "Large mean shift (mu 0.3 to 2.5 to 0.4, r=2)",
    "B_small_mean_shift":  "Small mean shift (mu 0.5 to 0.9 to 0.55, r=2)",
    "C_dispersion_only":   "Dispersion-only change (mu=1.0; r 3.0 to 0.4 to 3.0)",
    "D_zero_inflation":    "Zero-inflation change (structural zeros 0.10 to 0.85 to 0.10)",
    "E_short_burst":       "Short 10-day burst at days 180-190",
    "F_closely_spaced":    "Four changepoints spaced 40 days apart",
    "G_null_control":      "Null control: no changepoint",
    "H_very_sparse":       "Very sparse counts (mu 0.03 to 0.25 to 0.04, r=1)",
}


def _nb(rng, mu, r, size):
    return rng.negative_binomial(r, r / (r + mu), size).astype(float)


def simulate(scenario: str, seed: int) -> Tuple[np.ndarray, List[int]]:
    rng = np.random.default_rng(seed)
    if scenario == "A_large_mean_shift":
        return np.concatenate([_nb(rng, m, 2.0, SEG) for m in (0.3, 2.5, 0.4)]), [SEG, 2 * SEG]
    if scenario == "B_small_mean_shift":
        return np.concatenate([_nb(rng, m, 2.0, SEG) for m in (0.5, 0.9, 0.55)]), [SEG, 2 * SEG]
    if scenario == "C_dispersion_only":
        return np.concatenate([_nb(rng, 1.0, r, SEG) for r in (3.0, 0.4, 3.0)]), [SEG, 2 * SEG]
    if scenario == "D_zero_inflation":
        parts = []
        for zi in (0.10, 0.85, 0.10):
            base = _nb(rng, 1.0, 2.0, SEG)
            parts.append(np.where(rng.random(SEG) < zi, 0.0, base))
        return np.concatenate(parts), [SEG, 2 * SEG]
    if scenario == "E_short_burst":
        y = _nb(rng, 0.4, 2.0, N_DAYS)
        y[180:190] = _nb(rng, 3.0, 2.0, 10)
        return y, [180, 190]
    if scenario == "F_closely_spaced":
        return (np.concatenate([_nb(rng, m, 2.0, 40) for m in (0.4, 1.8, 0.4, 1.8, 0.4)]),
                [40, 80, 120, 160])
    if scenario == "G_null_control":
        return _nb(rng, 0.8, 2.0, N_DAYS), []
    if scenario == "H_very_sparse":
        return np.concatenate([_nb(rng, m, 1.0, SEG) for m in (0.03, 0.25, 0.04)]), [SEG, 2 * SEG]
    raise ValueError(f"Unknown scenario: {scenario}")


def series_summary(y) -> Dict[str, float]:
    y = np.asarray(y, dtype=float)
    mean = float(y.mean())
    var = float(y.var(ddof=1)) if len(y) > 1 else 0.0
    return {"n": len(y), "mean": mean, "variance": var,
            "var_mean_ratio": var / mean if mean > 0 else float("nan"),
            "percent_zero": 100.0 * float((y == 0).mean())}


P_GEOM_GRID = (1 / 30, 1 / 70, 1 / 150)


def sensitivity_single_series(y, truth=None, p_geom_values=P_GEOM_GRID,
                              base_prior=None, sampler=None, threshold=0.20,
                              tolerance=7, n_chains=2) -> pd.DataFrame:
    base_prior = base_prior or PriorConfig()
    sampler = sampler or SamplerConfig(n_warmup=2_000, n_samples=4_000)
    rows = []
    for i, value in enumerate(p_geom_values):
        prior = replace(base_prior, p_geom=float(value))
        cfg = replace(sampler, seed=sampler.seed + 97 * i)
        pooled = run_multiple_chains(y, prior, cfg, n_chains=n_chains)
        points, conf = posterior_changepoints(pooled, threshold=threshold)
        k_all = np.concatenate([c["k_trace"] for c in pooled["chains"]])
        row = {"p_geom": value,
               "prior_expected_segment_length": round(1.0 / value, 1),
               "posterior_mean_k": float(k_all.mean()),
               "posterior_mode_k": int(np.bincount(k_all).argmax()),
               "posterior_k_q05": float(np.quantile(k_all, 0.05)),
               "posterior_k_q95": float(np.quantile(k_all, 0.95)),
               "n_detected": len(points), "detected": points,
               "mean_detection_probability": float(np.mean(conf)) if conf else np.nan,
               "k_rhat": pooled["k_rhat"], "k_ess": pooled["k_ess"],
               "r_used": pooled["r_used"]}
        if truth is not None:
            m = match_detections(points, truth, tolerance)
            row.update({k: m[k] for k in ("tp", "fp", "fn", "precision", "recall", "f1")})
        rows.append(row)
    return pd.DataFrame(rows)


def sensitivity_across_scenarios(scenarios, n_repeats=5, p_geom_values=P_GEOM_GRID,
                                 min_segment=7, threshold=0.20, tolerance=7,
                                 n_warmup=1_500, n_samples=3_000, progress=None) -> pd.DataFrame:
    rows = []
    p_list = list(p_geom_values)
    total = len(scenarios) * n_repeats * len(p_list)
    done = 0
    for si, scenario in enumerate(scenarios):
        for rep in range(n_repeats):
            seed = 1000 * si + rep
            y, truth = simulate(scenario, seed)
            for value in p_list:
                prior = PriorConfig(p_geom=float(value), min_segment=min_segment)
                cfg = SamplerConfig(n_warmup=n_warmup, n_samples=n_samples, seed=seed)
                res = run_collapsed_rjmcmc(y, prior, cfg)
                points, conf = posterior_changepoints(res, threshold=threshold)
                m = match_detections(points, truth, tolerance)
                rows.append({"scenario": scenario, "rep": rep, "p_geom": value,
                             "prior_expected_segment_length": round(1.0 / value, 1),
                             "posterior_mean_k": float(res["k_trace"].mean()),
                             "n_detected": len(points),
                             "mean_detection_probability": float(np.mean(conf)) if conf else np.nan,
                             **{k: m[k] for k in ("tp", "fp", "fn", "precision", "recall", "f1")}})
                done += 1
                if progress:
                    progress(done / total)
    return pd.DataFrame(rows)


def summarise_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby(["scenario", "p_geom"])
              .agg(prior_expected_segment_length=("prior_expected_segment_length", "first"),
                   posterior_mean_k=("posterior_mean_k", "mean"),
                   n_detected=("n_detected", "mean"),
                   precision=("precision", "mean"), recall=("recall", "mean"),
                   f1=("f1", "mean"), fp=("fp", "mean"), fn=("fn", "mean"))
              .round(3).reset_index())


def stability_of_locations(locations: Dict[float, List[int]], tolerance: int = 7):
    keys = list(locations)
    if len(keys) < 2:
        return {"n_common": 0, "common": [], "jaccard": np.nan}
    common = [cp for cp in locations[keys[0]]
              if all(any(abs(cp - o) <= tolerance for o in locations[k]) for k in keys[1:])]
    union: List[int] = []
    for k in keys:
        for cp in locations[k]:
            if not any(abs(cp - u) <= tolerance for u in union):
                union.append(cp)
    return {"n_common": len(common), "common": common,
            "jaccard": len(common) / len(union) if union else np.nan}


def _hash_bytes(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def _try_melt_wide_format(df):
    if {"variable", "value"}.issubset(set(df.columns)):
        date_col = next((c for c in df.columns if c in ("date", "day", "timestamp", "time")), None)
        if date_col is None:
            df = df.copy()
            df["__date__"] = pd.RangeIndex(len(df))
            date_col = "__date__"
        out = df[[date_col, "variable", "value"]].copy()
        out.columns = ["date", "keyword", "count"]
        return out
    return None


def _detect_long_format(df):
    date_col = next((c for c in df.columns if c in ("date", "day", "timestamp", "time")), None)
    kw_col = next((c for c in df.columns if c in ("keyword", "word", "term", "variable")), None)
    cnt_col = next((c for c in df.columns if c in ("count", "frequency", "freq", "value", "n")), None)
    if None in (date_col, kw_col, cnt_col):
        return None
    out = df[[date_col, kw_col, cnt_col]].copy()
    out.columns = ["date", "keyword", "count"]
    return out


def _coerce_to_long_dataframe(raw_df, filename):
    raw_df = raw_df.copy()
    raw_df.columns = [str(c).strip().lower() for c in raw_df.columns]

    out = _detect_long_format(raw_df)
    if out is not None:
        return out, None

    out = _try_melt_wide_format(raw_df)
    if out is not None:
        return out, None

    date_col = next((c for c in raw_df.columns if c in ("date", "day", "timestamp", "time")), None)
    if date_col is not None and raw_df.shape[1] > 2:
        value_cols = [c for c in raw_df.columns if c != date_col]
        try:
            melted = raw_df.melt(id_vars=[date_col], value_vars=value_cols,
                                 var_name="keyword", value_name="count")
            melted.columns = ["date", "keyword", "count"]
            return melted, None
        except Exception:
            pass

    if raw_df.shape[1] >= 3:
        candidate = raw_df.iloc[:, :3].copy()
        candidate.columns = ["date", "keyword", "count"]
        parsed = pd.to_datetime(candidate["date"], errors="coerce")
        if parsed.notna().mean() > 0.5:
            candidate["date"] = parsed
            return candidate, None

    return None, (f"'{filename}' could not be interpreted as date/keyword/count data "
                  f"(columns found: {list(raw_df.columns)}). The file was skipped, "
                  f"but other uploaded files were still processed.")


@st.cache_data(show_spinner=False, max_entries=32)
def _parse_bytes_cached(file_bytes: bytes, filename: str, filehash: str):
    name = filename.lower()
    bio = io.BytesIO(file_bytes)
    try:
        if name.endswith((".csv", ".txt")):
            sep = "," if name.endswith(".csv") else None
            chunks = []
            try:
                reader = pd.read_csv(bio, sep=sep, engine="python", header=0,
                                     chunksize=50_000, on_bad_lines="skip")
                for chunk in reader:
                    chunks.append(chunk)
                raw_df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
            except Exception:
                bio.seek(0)
                raw_df = pd.read_csv(bio, sep=sep, engine="python", header=None,
                                     index_col=False, on_bad_lines="skip")
                raw_df.columns = [f"col_{i}" for i in range(raw_df.shape[1])]
            if raw_df.empty:
                bio.seek(0)
                raw_df = pd.read_csv(bio, sep=sep, engine="python", header=None,
                                     index_col=False, on_bad_lines="skip")
                raw_df.columns = [f"col_{i}" for i in range(raw_df.shape[1])]
        elif name.endswith((".rds", ".rdata", ".rda")):
            import pyreadr, tempfile
            suffix = "." + name.rsplit(".", 1)[-1]
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name
            try:
                result = pyreadr.read_r(tmp_path)
                if not result:
                    return None, f"'{filename}' contained no readable R objects."
                raw_df = list(result.values())[0]
                if raw_df is None:
                    return None, f"'{filename}' held an R object pyreadr cannot convert."
            finally:
                os.unlink(tmp_path)
        else:
            return None, f"Unsupported file type: {filename}"
    except Exception as exc:
        try:
            bio.seek(0)
            raw_df = pd.read_csv(bio, header=None, index_col=False,
                                 engine="python", on_bad_lines="skip")
            raw_df.columns = [f"col_{i}" for i in range(raw_df.shape[1])]
        except Exception:
            return None, f"Failed to parse '{filename}': {exc}"

    long_df, msg = _coerce_to_long_dataframe(raw_df, filename)
    if long_df is None:
        return None, msg

    long_df["date"] = pd.to_datetime(long_df["date"], errors="coerce")
    long_df["count"] = pd.to_numeric(long_df["count"], errors="coerce").fillna(0)
    long_df = long_df.dropna(subset=["date"])
    long_df["keyword"] = long_df["keyword"].astype(str).str.strip()
    if isinstance(long_df["date"].dtype, pd.DatetimeTZDtype):
        long_df["date"] = long_df["date"].dt.tz_localize(None)
    if long_df.empty:
        return None, f"'{filename}' parsed but held no valid date/count rows after cleaning."
    return long_df, None


def load_single_file(uploaded_file, progress_cb=None):
    file_bytes = uploaded_file.getvalue()
    if progress_cb:
        progress_cb(0.3)
    df, msg = _parse_bytes_cached(file_bytes, uploaded_file.name, _hash_bytes(file_bytes))
    if progress_cb:
        progress_cb(1.0)
    if df is None:
        st.warning(msg)
        return None
    return df


def build_master_dataframe(uploaded_files):
    frames, manifest = [], []
    progress = st.progress(0.0, text="Ingesting files...")
    n = len(uploaded_files)
    for i, f in enumerate(uploaded_files, start=1):
        def _cb(frac, i=i, n=n, name=f.name):
            progress.progress(min(1.0, (i - 1 + frac) / n),
                              text=f"Ingesting {name} ({i}/{n})...")
        df = load_single_file(f, progress_cb=_cb)
        if df is None or df.empty:
            continue
        dataset_id = f"dataset_{i}"
        df["dataset_id"] = dataset_id
        df["source_filename"] = f.name
        frames.append(df)
        manifest.append({"dataset_id": dataset_id, "filename": f.name, "rows": len(df),
                         "keywords": df["keyword"].nunique()})
    progress.progress(1.0, text="Done.")
    time.sleep(0.2)
    progress.empty()
    if not frames:
        return None, []
    master = pd.concat(frames, ignore_index=True)
    master["date"] = pd.to_datetime(master["date"], errors="coerce")
    return master.dropna(subset=["date"]), manifest


def load_demo_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(2024)
    dates = pd.date_range("2024-01-01", periods=730, freq="D")
    frames = []
    specs = [("keyword1", [0.4, 2.6, 0.6], 2.0), ("keyword2", [1.2, 0.3, 1.0], 1.5),
             ("keyword3", [0.05, 0.05, 0.35], 1.0)]
    for name, mus, r in specs:
        segments = [rng.negative_binomial(r, r / (r + mu), 244)[:244] for mu in mus]
        counts = np.concatenate(segments)[:730]
        if len(counts) < 730:
            counts = np.pad(counts, (0, 730 - len(counts)))
        frames.append(pd.DataFrame({"date": dates, "keyword": name,
                                    "count": counts.astype(float)}))
    demo = pd.concat(frames, ignore_index=True)
    demo["dataset_id"] = "demo"
    demo["source_filename"] = "demo_dataset"
    return demo


def profile_dataset(df: pd.DataFrame) -> Dict[str, object]:
    counts = pd.to_numeric(df["count"], errors="coerce").fillna(0)
    return {"n_rows": len(df), "n_keywords": df["keyword"].nunique(),
            "date_min": df["date"].min(), "date_max": df["date"].max(),
            "n_datasets": df["dataset_id"].nunique(),
            "percent_zero": 100.0 * float((counts == 0).mean()),
            "mean": float(counts.mean()), "variance": float(counts.var(ddof=1)),
            "var_mean_ratio": float(counts.var(ddof=1) / counts.mean())
                              if counts.mean() > 0 else float("nan")}


def build_anonymisation_map(df: pd.DataFrame) -> Dict[str, str]:
    return {kw: f"keyword{i+1}" for i, kw in enumerate(sorted(df["keyword"].unique()))}


def assign_frequency_tiers(df: pd.DataFrame) -> pd.DataFrame:
    means = df.groupby("keyword")["count"].mean().sort_values(ascending=False)
    low, high = means.quantile(1 / 3), means.quantile(2 / 3)
    tier = means.apply(lambda m: "A (High)" if m >= high
                       else ("B (Medium)" if m >= low else "C (Low)"))
    return pd.DataFrame({"keyword": means.index, "mean_frequency": means.values,
                         "tier": tier.values})


FREQ_MAP = {"Daily": "D", "Weekly": "W", "Monthly": "MS", "Yearly": "YS"}


@st.cache_data(show_spinner=False)
def aggregate_series(df, keywords, dataset_ids, freq_label):
    sub = df[df["keyword"].isin(keywords)]
    if dataset_ids is not None:
        sub = sub[sub["dataset_id"].isin(dataset_ids)]
    if sub.empty:
        return pd.Series(dtype=float)
    dates = pd.to_datetime(sub["date"], errors="coerce")
    counts = pd.to_numeric(sub["count"], errors="coerce").fillna(0)
    valid = dates.notna()
    dates, counts = dates[valid], counts[valid]
    if dates.empty:
        return pd.Series(dtype=float)
    s = pd.Series(counts.values, index=pd.DatetimeIndex(dates.values)).sort_index()
    return s.resample(FREQ_MAP[freq_label]).sum().fillna(0)


def overlay_figure(x, y, rj_points=None, cf_points=None, bl_points=None,
                   true_points=None, title="", as_bars=True):
    fig = go.Figure()
    if as_bars:
        fig.add_trace(go.Bar(x=x, y=y, name="Counts", marker_color="#c9c6bd",
                             marker_line_width=0, opacity=0.85))
    else:
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Counts",
                                 line=dict(color=COL_SERIES, width=1.6)))
    lo, hi = float(np.min(y)), float(np.max(y))
    pad = (hi - lo) * 0.08 if hi > lo else 1.0

    def rules(points, colour, dash, width):
        for p in points or []:
            idx = int(min(max(p, 0), len(x) - 1))
            fig.add_shape(type="line", x0=x[idx], x1=x[idx], y0=lo - pad, y1=hi + pad,
                          line=dict(color=colour, width=width, dash=dash))

    rules(true_points, COL_TRUE, "solid", 2.5)
    rules(rj_points, COL_RJ, "dash", 2)
    rules(cf_points, COL_CF, "dot", 2)
    rules(bl_points, COL_BL, "dashdot", 1.2)

    for name, colour, dash, pts in [("True changepoint", COL_TRUE, "solid", true_points),
                                    ("RJMCMC (corrected)", COL_RJ, "dash", rj_points),
                                    ("changeforest (official)", COL_CF, "dot", cf_points),
                                    ("Rolling-RF baseline", COL_BL, "dashdot", bl_points)]:
        if pts:
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name=name,
                                     line=dict(color=colour, width=2, dash=dash)))
    fig.update_layout(title=title, template="plotly_white", height=440,
                      xaxis_title="Time", yaxis_title="Aggregated frequency",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                      margin=dict(t=70, l=50, r=30, b=40), bargap=0.05)
    return fig


def posterior_figure(x, probs, threshold):
    fig = go.Figure()
    upper = min(len(probs), max(len(x) - 1, 0))
    fig.add_trace(go.Scatter(x=x[1:upper + 1], y=probs[:upper], mode="lines",
                             line=dict(color=COL_RJ, width=1.6), name="P(changepoint)"))
    fig.add_hline(y=threshold, line=dict(color="#888", width=1, dash="dot"),
                  annotation_text=f"threshold {threshold}")
    fig.update_layout(template="plotly_white", height=270,
                      title="RJMCMC posterior inclusion probability",
                      xaxis_title="Time", yaxis_title="Probability", yaxis_range=[0, 1],
                      margin=dict(t=60, l=50, r=30, b=40))
    return fig


def trace_figure(chains):
    fig = go.Figure()
    for i, c in enumerate(chains):
        fig.add_trace(go.Scatter(y=c["k_trace"], mode="lines", name=f"chain {i+1}",
                                 line=dict(width=1)))
    fig.update_layout(template="plotly_white", height=270,
                      title="Trace of k (number of changepoints)",
                      xaxis_title="Retained iteration", yaxis_title="k",
                      margin=dict(t=60, l=50, r=30, b=40))
    return fig


def confidence_figure(rj_conf, cf_conf, bl_conf=None):
    fig = go.Figure()
    fig.add_trace(go.Bar(name="RJMCMC posterior probability",
                         x=[f"CP {i+1}" for i in range(len(rj_conf))], y=rj_conf,
                         marker_color=COL_RJ))
    fig.add_trace(go.Bar(name="changeforest 1 - p", marker_color=COL_CF,
                         x=[f"CP {i+1}" for i in range(len(cf_conf))], y=cf_conf))
    if bl_conf:
        fig.add_trace(go.Bar(name="Rolling-RF score", marker_color=COL_BL,
                             x=[f"CP {i+1}" for i in range(len(bl_conf))], y=bl_conf))
    fig.update_layout(barmode="group", template="plotly_white", height=330,
                      title="Detection confidence by method",
                      yaxis_title="Score", margin=dict(t=60, l=50, r=30, b=40))
    return fig


def build_html_report(context: Dict[str, object]) -> str:
    rows = "".join(
        f"<tr><td>{r['Method']}</td><td>{r['Detected']}</td>"
        f"<td>{'' if pd.isna(r['Mean confidence']) else round(r['Mean confidence'], 3)}</td>"
        f"<td>{r['Runtime (s)']}</td></tr>"
        for r in context["metrics"].to_dict("records"))
    cps = "".join(f"<li>{c}</li>" for c in context.get("changepoint_lines", []))
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Change Point Detection Report</title>
<style>body{{font-family:Georgia,serif;max-width:60em;margin:3em auto;line-height:1.6;color:#222}}
h1{{border-bottom:2px solid #2C7FB8;padding-bottom:.3em}}
table{{border-collapse:collapse;width:100%;margin:1em 0}}
th,td{{border:1px solid #ccc;padding:.5em .8em;text-align:left}}
th{{background:#f2f2f2}} .note{{background:#fff8e1;padding:1em;border-left:4px solid #f0ad4e}}</style>
</head><body>
<h1>Change Point Detection Report</h1>
<p><strong>Series:</strong> {context['series_label']}<br>
<strong>Aggregation:</strong> {context['freq_label']}<br>
<strong>Observations:</strong> {context['n_obs']}<br>
<strong>Generated:</strong> {time.strftime('%Y-%m-%d %H:%M')}</p>
<h2>Prior and sampler</h2>
<p>Expected segment length 1/p_geom = {context['p_geom_label']}; minimum segment
{context['min_segment']}; Beta({context['a_p']}, {context['b_p']}); {context['n_chains']}
chains, {context['n_warmup']} warm-up and {context['n_samples']} retained iterations;
posterior threshold {context['threshold']}.</p>
<h2>Detection summary</h2>
<table><tr><th>Method</th><th>Detected</th><th>Mean confidence</th><th>Runtime (s)</th></tr>
{rows}</table>
<h2>Convergence</h2><p>split-Rhat for k = {context['k_rhat']}; ESS = {context['k_ess']};
estimated dispersion r = {context['r_used']}.</p>
<h2>Detected changepoints</h2><ul>{cps or '<li>None above threshold.</li>'}</ul>
<div class="note"><strong>Interpretation note.</strong> The RJMCMC score is a posterior
inclusion probability. The changeforest score is one minus a permutation p-value. These
are different quantities and should not be read on a common scale. The number of
changepoints is prior-sensitive; locations are considerably more stable than counts.</div>
</body></html>"""


st.sidebar.title("Configuration")
mode = st.sidebar.radio("Mode", ["Real data", "Simulation lab", "Prior sensitivity", "Deployment"],
                        index=0)

st.sidebar.subheader("Bayesian prior")
p_geom_choice = st.sidebar.select_slider(
    "Expected segment length (1/p_geom)", options=[30, 70, 150], value=70,
    help="Prior belief about how often discourse shifts. Studied formally in the "
         "Prior sensitivity mode.")
p_geom = 1.0 / p_geom_choice
min_segment = st.sidebar.slider("Minimum segment length", 3, 40, 7,
                                help="Applied identically to both tracks for fairness.")
a_p = st.sidebar.number_input("Beta prior a_p", 0.1, 10.0, 1.0, 0.1)
b_p = st.sidebar.number_input("Beta prior b_p", 0.1, 10.0, 1.0, 0.1)
threshold = st.sidebar.slider("Posterior inclusion threshold", 0.05, 0.90, 0.20, 0.05)

st.sidebar.subheader("Sampler")
n_warmup = st.sidebar.select_slider("Warm-up iterations", [500, 1000, 2000, 4000], 2000)
n_samples = st.sidebar.select_slider("Retained samples", [1000, 2000, 4000, 8000], 4000)
n_chains = st.sidebar.slider("Chains", 1, 4, 2)
seed = st.sidebar.number_input("Random seed", 0, 10_000, 0, 1)

st.sidebar.subheader("changeforest")
cf_alpha = st.sidebar.select_slider("Permutation alpha", [0.10, 0.05, 0.02, 0.01, 0.005], 0.01)
cf_segmentation = st.sidebar.selectbox(
    "Segmentation", ["bs", "sbs", "wbs"], index=0,
    help="bs is recommended: sbs and wbs over-segment sparse count series.")
cf_perms = st.sidebar.select_slider("Permutations", [99, 199, 499], 199)
cf_transform = st.sidebar.selectbox("Variance-stabilising transform",
                                    ["none", "anscombe", "log1p"], index=0)

st.sidebar.subheader("Evaluation")
tolerance = st.sidebar.slider("Matching tolerance (steps)", 1, 21, 7)
show_baseline = st.sidebar.checkbox("Show rolling-RF baseline (not changeforest)", False,
                                    help="The v5 heuristic, retained as a cautionary control.")

if not CHANGEFOREST_AVAILABLE:
    st.sidebar.error(f"changeforest unavailable: {CHANGEFOREST_ERROR}\n\n"
                     "Install with: pip install changeforest")

with st.sidebar.expander("Sampler correctness self-test"):
    if st.button("Run detailed-balance test", key="btn_balance_test"):
        out = verify_detailed_balance()
        if out["passed"]:
            st.success(f"Passed on {out['pairs_checked']} directed pairs "
                       f"(max log-flux error {out['max_log_flux_error']:.2e}).")
        else:
            st.error(f"Failed: {out}")
    st.caption("Verifies the implemented proposal and acceptance calculations satisfy "
               "detailed balance exactly on a small enumerable state space.")


def make_prior():
    return PriorConfig(a_p=a_p, b_p=b_p, p_geom=p_geom, min_segment=int(min_segment))


def make_sampler(s=None):
    return SamplerConfig(n_warmup=int(n_warmup), n_samples=int(n_samples),
                         seed=int(seed if s is None else s))


def make_cf_config(n, s=None):
    return ChangeforestConfig(
        segmentation=cf_segmentation, model_selection_alpha=float(cf_alpha),
        model_selection_n_permutations=int(cf_perms),
        minimal_relative_segment_length=float(np.clip(min_segment / max(n, 1), 1e-3, 0.49)),
        seed=int(seed if s is None else s))


def run_both_tracks(y, truth=None):
    t0 = time.time()
    pooled = run_multiple_chains(y, make_prior(), make_sampler(), n_chains=int(n_chains))
    rj_points, rj_conf = posterior_changepoints(pooled, threshold=threshold)
    t_rj = time.time() - t0

    cf = {"points": [], "p_values": [], "gains": [], "runtime_s": 0.0, "tree": None}
    cf_error = None
    if CHANGEFOREST_AVAILABLE:
        try:
            cf = detect_changeforest(y, make_cf_config(len(y)), transform=cf_transform)
        except Exception as exc:
            cf_error = str(exc)

    bl = detect_rolling_rf_baseline(y, seed=int(seed)) if show_baseline else None
    cf_conf = [1.0 - p for p in cf["p_values"] if np.isfinite(p)]

    rows = [{"Method": "RJMCMC (corrected)", "Detected": len(rj_points),
             "Mean confidence": float(np.mean(rj_conf)) if rj_conf else np.nan,
             "Runtime (s)": round(t_rj, 3)},
            {"Method": "changeforest (official)", "Detected": len(cf["points"]),
             "Mean confidence": float(np.mean(cf_conf)) if cf_conf else np.nan,
             "Runtime (s)": round(cf["runtime_s"], 3)}]
    if bl is not None:
        rows.append({"Method": "Rolling-RF baseline", "Detected": len(bl["points"]),
                     "Mean confidence": float(np.mean(bl["confidences"]))
                                        if bl["confidences"] else np.nan,
                     "Runtime (s)": round(bl["runtime_s"], 3)})
    metrics = pd.DataFrame(rows)

    evaluation = None
    if truth is not None:
        keys = ("tp", "fp", "fn", "precision", "recall", "f1")
        ev = [{"Method": "RJMCMC (corrected)",
               **{k: v for k, v in match_detections(rj_points, truth, tolerance).items() if k in keys}},
              {"Method": "changeforest (official)",
               **{k: v for k, v in match_detections(cf["points"], truth, tolerance).items() if k in keys}}]
        if bl is not None:
            ev.append({"Method": "Rolling-RF baseline",
                       **{k: v for k, v in match_detections(bl["points"], truth, tolerance).items()
                          if k in keys}})
        evaluation = pd.DataFrame(ev).round(3)

    return dict(pooled=pooled, rj_points=rj_points, rj_conf=rj_conf, cf=cf,
                cf_conf=cf_conf, cf_error=cf_error, bl=bl,
                metrics=metrics, evaluation=evaluation)


def render_diagnostics(pooled):
    c1, c2, c3, c4 = st.columns(4)
    rhat = pooled["k_rhat"]
    c1.metric("split-Rhat (k)", f"{rhat:.3f}" if np.isfinite(rhat) else "n/a",
              help="Values below about 1.01 indicate agreement between chains.")
    c2.metric("ESS (k)", f"{pooled['k_ess']:.0f}")
    c3.metric("Dispersion r", f"{pooled['r_used']:.3f}")
    rates = pooled["acceptance_rate"]
    c4.metric("Accept birth / death", f"{rates['birth']:.2f} / {rates['death']:.2f}")
    if np.isfinite(rhat) and rhat > 1.05:
        st.warning("split-Rhat exceeds 1.05. Increase warm-up or retained samples "
                   "before reporting these results.")
    st.plotly_chart(trace_figure(pooled["chains"]), use_container_width=True)


if mode == "Real data":
    st.title("Change Point Detection in Online News Word Frequencies")
    st.caption("Corrected collapsed RJMCMC compared against the official changeforest "
               "package. Upload CSV, TXT, RDS or RData in date/keyword/count form; wide "
               "layouts are reshaped automatically.")

    tab_data, tab_explore, tab_analyse, tab_diag = st.tabs(
        ["1. Data", "2. Explore", "3. Analyse", "4. Diagnostics"])

    with tab_data:
        st.subheader("Upload data")
        files = st.file_uploader(
            "CSV, TXT, RDS or RData. Multiple files are merged with provenance retained.",
            type=["csv", "txt", "rds", "rdata", "rda"], accept_multiple_files=True)
        c1, c2 = st.columns([1, 3])
        if c1.button("Load demo dataset", key="btn_load_demo"):
            st.session_state["master"] = load_demo_dataset()
            st.session_state["manifest"] = [{"dataset_id": "demo", "filename": "demo_dataset",
                                             "rows": 2190, "keywords": 3}]
            st.success("Demo dataset loaded: 3 keywords over 730 days.")
        if c2.button("Clear all data", key="btn_clear_data"):
            for key in ("master", "manifest", "real_result", "anonymisation_map",
                       "working", "ps_single_result", "ps_full_result", "ps_real_result"):
                st.session_state.pop(key, None)
            st.info("Cleared.")

        if files:
            master, manifest = build_master_dataframe(files)
            if master is not None:
                st.session_state["master"] = master
                st.session_state["manifest"] = manifest
                st.success(f"Loaded {len(manifest)} file(s), {len(master):,} rows.")
            else:
                st.error("No file could be interpreted. Expected columns date, keyword, count.")

        if "manifest" in st.session_state and st.session_state["manifest"]:
            st.subheader("File manifest")
            st.dataframe(pd.DataFrame(st.session_state["manifest"]),
                         hide_index=True, use_container_width=True)

        with st.expander("Expected data formats"):
            st.markdown(
                "- **Long**: columns `date`, `keyword`, `count` (aliases accepted: "
                "`day`/`timestamp`, `word`/`term`/`variable`, `frequency`/`freq`/`value`/`n`).\n"
                "- **Wide**: a `date` column plus one column per keyword; reshaped automatically.\n"
                "- **R files**: `.rds` and `.RData` are read via pyreadr; the first data frame is used.\n"
                "- Files that cannot be interpreted are skipped with a warning; the remaining "
                "files are still processed.")

    with tab_explore:
        if "master" not in st.session_state:
            st.info("Load data on the Data tab first.")
        else:
            master = st.session_state["master"]
            prof = profile_dataset(master)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Records", f"{prof['n_rows']:,}")
            c2.metric("Keywords", prof["n_keywords"])
            c3.metric("Datasets", prof["n_datasets"])
            c4.metric("Date range",
                      f"{prof['date_min']:%Y-%m-%d} to {prof['date_max']:%Y-%m-%d}")
            c5, c6, c7 = st.columns(3)
            c5.metric("Zero observations", f"{prof['percent_zero']:.1f}%")
            c6.metric("Mean count", f"{prof['mean']:.4f}")
            c7.metric("Variance / mean", f"{prof['var_mean_ratio']:.2f}",
                      help="Values above 1 indicate overdispersion, which justifies the "
                           "Negative-Binomial likelihood over a Poisson.")

            st.subheader("Anonymisation")
            anonymise = st.checkbox("Replace keywords with generic identifiers", True,
                                    key="chk_anonymise",
                                    help="Keeps evaluation purely statistical and avoids "
                                         "topic bias when judging detections.")
            working = master.copy()
            if anonymise:
                mapping = build_anonymisation_map(master)
                st.session_state["anonymisation_map"] = mapping
                working["keyword"] = working["keyword"].map(mapping)
            st.session_state["working"] = working

            st.subheader("Frequency tiers")
            tiers = assign_frequency_tiers(working)
            st.dataframe(tiers.round(5), hide_index=True, use_container_width=True,
                         height=240)
            st.caption("Tiers are data-driven 33rd and 67th percentile splits on mean "
                       "occurrence rate, so results can be stratified by how common a "
                       "keyword is.")

            st.subheader("Preview")
            search = st.text_input("Filter keywords", "", key="txt_filter_kw")
            preview = working if not search else working[
                working["keyword"].str.contains(search, case=False, na=False)]
            page_size = st.select_slider("Rows per page", [25, 50, 100, 250], 50,
                                         key="sld_page_size")
            total_pages = max(1, math.ceil(len(preview) / page_size))
            page = st.number_input("Page", 1, total_pages, 1, 1, key="num_page")
            start = (page - 1) * page_size
            st.dataframe(preview.iloc[start:start + page_size], use_container_width=True,
                         hide_index=True)
            st.caption(f"Showing {min(page_size, max(len(preview) - start, 0))} of "
                       f"{len(preview):,} rows (page {page} of {total_pages}).")

    with tab_analyse:
        if "working" not in st.session_state:
            st.info("Load and explore data first.")
        else:
            working = st.session_state["working"]
            tiers = assign_frequency_tiers(working)
            all_keywords = sorted(working["keyword"].unique())

            c1, c2 = st.columns(2)
            tier_filter = c1.multiselect("Restrict to tiers",
                                         ["A (High)", "B (Medium)", "C (Low)"], [],
                                         key="ms_tier_filter")
            pool = (all_keywords if not tier_filter
                    else sorted(tiers[tiers["tier"].isin(tier_filter)]["keyword"]))
            chosen = c2.multiselect("Keywords", pool, default=pool[:1], key="ms_keywords")

            c3, c4 = st.columns(2)
            dataset_ids = c3.multiselect("Datasets", sorted(working["dataset_id"].unique()),
                                         default=sorted(working["dataset_id"].unique()),
                                         key="ms_datasets")
            freq_label = c4.selectbox("Temporal aggregation", list(FREQ_MAP), index=0,
                                      key="sel_freq")

            if chosen and st.button("Run both detectors", type="primary", key="btn_run_real"):
                series = aggregate_series(working, chosen, dataset_ids or None, freq_label)
                if len(series) < 20:
                    st.error("Fewer than 20 points after aggregation. Choose a coarser "
                             "frequency or add keywords.")
                elif 2 * min_segment > len(series):
                    st.error(f"Minimum segment {min_segment} is too large for a series of "
                             f"length {len(series)}. Reduce it in the sidebar.")
                else:
                    with st.spinner("Sampling and segmenting..."):
                        res = run_both_tracks(series.values.astype(float), truth=None)
                    st.session_state["real_result"] = (res, series, chosen, freq_label)

            if "real_result" in st.session_state:
                res, series, chosen, freq_label = st.session_state["real_result"]
                if res["cf_error"]:
                    st.warning(f"changeforest failed on this series: {res['cf_error']}")
                x = series.index
                st.plotly_chart(
                    overlay_figure(x, series.values, res["rj_points"], res["cf"]["points"],
                                   res["bl"]["points"] if res["bl"] else None, None,
                                   title=f"{', '.join(chosen)} ({freq_label})"),
                    use_container_width=True)
                st.plotly_chart(
                    posterior_figure(x, res["pooled"]["posterior_probability"], threshold),
                    use_container_width=True)
                st.dataframe(res["metrics"], hide_index=True, use_container_width=True)
                if res["rj_conf"] or res["cf_conf"]:
                    st.plotly_chart(
                        confidence_figure(res["rj_conf"], res["cf_conf"],
                                          res["bl"]["confidences"] if res["bl"] else None),
                        use_container_width=True)
                    st.caption("These bars are not on a common scale: the RJMCMC score is a "
                               "posterior probability, the changeforest score is one minus a "
                               "permutation p-value.")

                lines: List[str] = []
                records = []
                for label, pts, conf in [("RJMCMC", res["rj_points"], res["rj_conf"]),
                                         ("changeforest", res["cf"]["points"], res["cf_conf"])]:
                    for j, p in enumerate(pts):
                        stamp = x[min(p, len(x) - 1)]
                        score = conf[j] if j < len(conf) else np.nan
                        records.append({"method": label, "index": p, "date": stamp,
                                        "score": round(float(score), 4)
                                                 if np.isfinite(score) else np.nan})
                        lines.append(f"{label}: {stamp:%Y-%m-%d} (index {p}, score "
                                     f"{score:.3f})" if np.isfinite(score)
                                     else f"{label}: {stamp:%Y-%m-%d} (index {p})")
                if records:
                    dated = pd.DataFrame(records)
                    st.subheader("Detected changepoints with dates")
                    st.dataframe(dated, hide_index=True, use_container_width=True)
                    d1, d2 = st.columns(2)
                    d1.download_button("Download changepoints (CSV)",
                                       dated.to_csv(index=False).encode(),
                                       "detected_changepoints.csv", "text/csv",
                                       key="dl_real_csv")
                    report = build_html_report({
                        "metrics": res["metrics"], "series_label": ", ".join(chosen),
                        "freq_label": freq_label, "n_obs": len(series),
                        "p_geom_label": p_geom_choice, "min_segment": min_segment,
                        "a_p": a_p, "b_p": b_p, "n_chains": n_chains,
                        "n_warmup": n_warmup, "n_samples": n_samples,
                        "threshold": threshold,
                        "k_rhat": round(res["pooled"]["k_rhat"], 4),
                        "k_ess": round(res["pooled"]["k_ess"]),
                        "r_used": round(res["pooled"]["r_used"], 4),
                        "changepoint_lines": lines})
                    d2.download_button("Download HTML report", report.encode(),
                                       "changepoint_report.html", "text/html",
                                       key="dl_real_html")
                else:
                    st.info("No changepoint exceeded the current threshold.")

    with tab_diag:
        if "real_result" not in st.session_state:
            st.info("Run an analysis first.")
        else:
            res = st.session_state["real_result"][0]
            render_diagnostics(res["pooled"])
            st.subheader("changeforest segmentation tree")
            table = changeforest_segmentation_table(res["cf"])
            if table.empty:
                st.info("No significant split was retained at this alpha.")
            else:
                st.dataframe(table, hide_index=True, use_container_width=True)
                st.caption("gain is the maximised classifier log-likelihood ratio; p_value "
                           "comes from the permutation test.")


elif mode == "Simulation lab":
    st.title("Simulation and Evaluation Lab")
    st.caption("Eight Negative-Binomial stress-test scenarios with known ground truth. "
               "No detector sees the true changepoints; truth is used only at evaluation.")

    scenario = st.selectbox("Scenario", list(SCENARIOS), key="sim_scenario",
                            format_func=lambda k: f"{k.split('_')[0]} — {SCENARIOS[k]}")
    rep_seed = st.number_input("Replicate seed", 0, 9999, 0, 1, key="sim_seed")
    y, truth = simulate(scenario, int(rep_seed))
    x = np.arange(len(y))

    s = series_summary(y)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Observations", s["n"])
    c2.metric("Mean", f"{s['mean']:.3f}")
    c3.metric("Variance / mean", f"{s['var_mean_ratio']:.2f}")
    c4.metric("Zeros", f"{s['percent_zero']:.1f}%")

    if st.button("Run detection", type="primary", key="btn_run_sim"):
        with st.spinner("Sampling and segmenting..."):
            st.session_state["sim_result"] = (run_both_tracks(y, truth), scenario, y, truth)

    if "sim_result" in st.session_state:
        res, scen, y, truth = st.session_state["sim_result"]
        x = np.arange(len(y))
        st.plotly_chart(overlay_figure(x, y, res["rj_points"], res["cf"]["points"],
                                       res["bl"]["points"] if res["bl"] else None, truth,
                                       title=SCENARIOS[scen]), use_container_width=True)
        st.plotly_chart(posterior_figure(x, res["pooled"]["posterior_probability"], threshold),
                        use_container_width=True)
        left, right = st.columns(2)
        left.subheader("Detection summary")
        left.dataframe(res["metrics"], hide_index=True, use_container_width=True)
        if res["evaluation"] is not None:
            right.subheader(f"Evaluation (tolerance {tolerance})")
            right.dataframe(res["evaluation"], hide_index=True, use_container_width=True)
        with st.expander("MCMC diagnostics"):
            render_diagnostics(res["pooled"])
        with st.expander("changeforest segmentation tree"):
            table = changeforest_segmentation_table(res["cf"])
            if table.empty:
                st.info("No significant split at this alpha.")
            else:
                st.dataframe(table, hide_index=True, use_container_width=True)


elif mode == "Prior sensitivity":
    st.title("Prior Sensitivity Study")
    st.caption("How far do posterior conclusions depend on the geometric segment-length "
               "prior rather than on the data? Grid: expected segment length 30, 70, 150.")

    tab_single, tab_full, tab_real = st.tabs(
        ["Single simulated series", "Full study across scenarios", "Real uploaded series"])

    def render_sensitivity(df, y=None, truth=None, key=""):
        display = df.drop(columns=["detected"]).round(3)
        st.dataframe(display, hide_index=True, use_container_width=True)
        locations = {row["p_geom"]: row["detected"] for _, row in df.iterrows()}
        stab = stability_of_locations(locations, tolerance)
        c1, c2 = st.columns(2)
        c1.metric("Detections common to all priors", stab["n_common"])
        c2.metric("Jaccard stability",
                  f"{stab['jaccard']:.2f}" if np.isfinite(stab["jaccard"]) else "n/a")
        st.caption("Changepoints found under all three priors are prior-robust and should "
                   "be reported as primary findings. Those appearing only under the tightest "
                   "prior are prior-driven and belong in a secondary list.")
        if y is not None:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=np.arange(len(y)), y=y, marker_color="#c9c6bd",
                                 name="Counts", opacity=0.8))
            palette = ["#1f77b4", "#ff7f0e", "#9467bd"]
            lo, hi = float(np.min(y)), float(np.max(y))
            for i, (_, row) in enumerate(df.iterrows()):
                for p in row["detected"]:
                    fig.add_shape(type="line", x0=p, x1=p, y0=lo, y1=hi,
                                  line=dict(color=palette[i % 3], width=2, dash="dash"))
                fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                                         line=dict(color=palette[i % 3], width=2, dash="dash"),
                                         name=f"1/{int(row['prior_expected_segment_length'])}"))
            for p in (truth or []):
                fig.add_shape(type="line", x0=p, x1=p, y0=lo, y1=hi,
                              line=dict(color=COL_TRUE, width=2.5))
            fig.update_layout(template="plotly_white", height=390,
                              title="Detections under each prior setting",
                              margin=dict(t=60, l=50, r=30, b=40))
            st.plotly_chart(fig, use_container_width=True)
        st.download_button("Download sensitivity table (CSV)",
                           display.to_csv(index=False).encode(),
                           f"prior_sensitivity_{key}.csv", "text/csv", key=f"dl_ps_{key}")

    # --- Tab 1: single simulated series ---
    # FIX (v6.1): storage key "ps_single_result" is now distinct from the button
    # key "btn_ps_single". In v6 both used the string "ps_single", so the button
    # widget (which owns that session_state slot as a bool) silently overwrote
    # the stored results tuple, causing "TypeError: cannot unpack non-iterable
    # bool object" on the next rerun. Every button/storage pair below uses two
    # different names for the same reason.
    with tab_single:
        scenario = st.selectbox("Scenario", list(SCENARIOS), key="ps_scen",
                                format_func=lambda k: f"{k.split('_')[0]} — {SCENARIOS[k]}")
        rep_seed = st.number_input("Replicate seed", 0, 9999, 0, 1, key="ps_seed")
        y, truth = simulate(scenario, int(rep_seed))
        if st.button("Run sensitivity", type="primary", key="btn_ps_single"):
            with st.spinner("Running three prior settings..."):
                st.session_state["ps_single_result"] = (
                    sensitivity_single_series(y, truth, P_GEOM_GRID, make_prior(),
                                              make_sampler(), threshold, tolerance,
                                              int(n_chains)), y, truth)
        stored = st.session_state.get("ps_single_result")
        if isinstance(stored, tuple) and len(stored) == 3:
            df, y, truth = stored
            render_sensitivity(df, y, truth, key="single")

    with tab_full:
        chosen = st.multiselect("Scenarios", list(SCENARIOS), default=list(SCENARIOS)[:4],
                                format_func=lambda k: k.split("_")[0], key="ps_full_scenarios")
        repeats = st.slider("Replicates per scenario", 1, 10, 3, key="ps_full_repeats")
        st.caption(f"This runs {len(chosen) * repeats * 3} MCMC fits and may take several minutes.")
        if st.button("Run full study", type="primary", key="btn_ps_full") and chosen:
            bar = st.progress(0.0, text="Running...")
            st.session_state["ps_full_result"] = sensitivity_across_scenarios(
                chosen, repeats, P_GEOM_GRID, int(min_segment), threshold, tolerance,
                int(n_warmup) // 2, int(n_samples) // 2,
                progress=lambda f: bar.progress(min(f, 1.0), text=f"Running... {f:.0%}"))
            bar.empty()
        raw = st.session_state.get("ps_full_result")
        if isinstance(raw, pd.DataFrame) and not raw.empty:
            summary = summarise_sensitivity(raw)
            for label, field in [("Mean F1 by prior", "f1"),
                                 ("Posterior mean number of changepoints", "posterior_mean_k"),
                                 ("Mean false positives", "fp")]:
                st.subheader(label)
                st.dataframe(summary.pivot(index="scenario",
                                           columns="prior_expected_segment_length",
                                           values=field), use_container_width=True)
            c1, c2 = st.columns(2)
            c1.download_button("Download summary (CSV)", summary.to_csv(index=False).encode(),
                               "prior_sensitivity_summary.csv", "text/csv", key="dl_ps_full_sum")
            c2.download_button("Download raw results (CSV)", raw.to_csv(index=False).encode(),
                               "prior_sensitivity_raw.csv", "text/csv", key="dl_ps_full_raw")

    with tab_real:
        if "working" not in st.session_state:
            st.info("Upload data in Real data mode first, then return here.")
        else:
            working = st.session_state["working"]
            c1, c2 = st.columns(2)
            kws = c1.multiselect("Keywords", sorted(working["keyword"].unique()),
                                 default=sorted(working["keyword"].unique())[:1],
                                 key="ps_real_kw")
            freq_label = c2.selectbox("Aggregation", list(FREQ_MAP), index=0, key="ps_real_freq")
            if kws and st.button("Run sensitivity on real series", type="primary",
                                 key="btn_ps_real"):
                series = aggregate_series(working, kws, None, freq_label)
                if len(series) < 20:
                    st.error("Series too short after aggregation.")
                else:
                    with st.spinner("Running three prior settings..."):
                        st.session_state["ps_real_result"] = (
                            sensitivity_single_series(series.values.astype(float), None,
                                                      P_GEOM_GRID, make_prior(), make_sampler(),
                                                      threshold, tolerance, int(n_chains)),
                            series)
            stored_real = st.session_state.get("ps_real_result")
            if isinstance(stored_real, tuple) and len(stored_real) == 2:
                df, series = stored_real
                render_sensitivity(df, series.values.astype(float), None, key="real")
                locations = {row["p_geom"]: row["detected"] for _, row in df.iterrows()}
                stab = stability_of_locations(locations, tolerance)
                if stab["common"]:
                    dates = [series.index[min(p, len(series) - 1)] for p in stab["common"]]
                    st.subheader("Prior-robust changepoints (report these as primary)")
                    st.dataframe(pd.DataFrame({"index": stab["common"], "date": dates}),
                                 hide_index=True, use_container_width=True)


else:
    st.title("Deployment: localhost and ngrok")
    st.markdown("""
### Local run (Windows)

```
pip install -r requirements.txt
streamlit run app.py
```

Streamlit serves on `http://localhost:8501`.

### Exposing the app with ngrok

Run Streamlit and ngrok in **two separate terminals**.

Terminal 1:

```
streamlit run app.py --server.port 8501 --server.headless true
```

Terminal 2:

```
ngrok http 8501
```

ngrok prints a public forwarding URL such as `https://xxxx-xx-xx.ngrok-free.app`.
Open that URL to reach the dashboard from any machine.

### Notebook or Colab alternative

```python
from pyngrok import ngrok
ngrok.set_auth_token("YOUR_TOKEN")
tunnel = ngrok.connect(8501, "http")
print(tunnel.public_url)
```

### Settings worth knowing

Add a `.streamlit/config.toml` beside `app.py`:

```toml
[server]
port = 8501
headless = true
maxUploadSize = 2000
enableXsrfProtection = true
enableCORS = false

[browser]
gatherUsageStats = false
```

`maxUploadSize` is raised to 2000 MB because the Yle corpus is large. Setting
`enableCORS = false` with XSRF protection left on is the combination that keeps
file uploads working through an ngrok tunnel.

### Practical notes

- ngrok free tier issues a new URL on every restart; the paid tier supports a
  reserved domain.
- Long MCMC runs continue server-side, but a browser tab left idle behind a
  tunnel may disconnect. Prefer fewer chains and shorter runs when demonstrating
  remotely, then reproduce final numbers locally.
- Never expose a tunnel carrying unpublished corpus data on a public URL without
  the anonymisation checkbox enabled.
""")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Environment")
        st.dataframe(pd.DataFrame([
            {"Component": "changeforest",
             "Status": "available" if CHANGEFOREST_AVAILABLE else f"missing ({CHANGEFOREST_ERROR})"},
            {"Component": "pandas", "Status": pd.__version__},
            {"Component": "numpy", "Status": np.__version__},
            {"Component": "streamlit", "Status": st.__version__},
        ]), hide_index=True, use_container_width=True)
    with c2:
        st.subheader("Sampler self-test")
        if st.button("Verify detailed balance", key="btn_deploy_test"):
            out = verify_detailed_balance()
            if out["passed"]:
                st.success(f"Passed on {out['pairs_checked']} directed pairs "
                          f"(max log-flux error {out['max_log_flux_error']:.2e})")
            else:
                st.error(f"Failed: {out}")
