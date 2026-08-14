"""Synthetic world generators — Floor Certification Law suite (SYNTHETIC MECHANISM TESTS).

Families per EXPERIMENT_PLAN v2 §B1/§B2/§B4:
  (i)   continuum margin family — regular left-neighborhood frontier margin (kappa, s0)
  (ii)  T2' per-(n,m) family — additive-lower witnesses (slack set relative to (n,m))
  (iii) B'-class K-cell stratified worlds — regime-edge / K-sweep instances
  (iv)  separation instance — DRO-box blocked where B' certifies
  plus: matched-pair worlds (B2, two mechanism families), non-theorem-aligned family
  (R044, irregular margin), null worlds (R003 calibration).

A World is stratified at micro-bin granularity: arrays (p, q, eta) over n_bins bins in
DECLARED score order (prefix lattices accept bins left-to-right; the declared order is
world structure available identically to every arm — R039 §1). w = q/p exactly.

RNG discipline (R039 §3): all sampling via streams from
  np.random.SeedSequence(20260611, spawn_key=(cell_index, repeat_index))
through `rng_for`; world draws use spawn_key=("world", family_id, draw_index) streams.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

MASTER_SEED = 20260611


def _spawn_key(parts) -> tuple:
    """SeedSequence spawn keys must be uint32 ints; flatten nested keys and hash
    non-integers with a PROCESS-STABLE hash (zlib.crc32 — Python's hash() is salted
    per process and would break the registered reproducibility across workers)."""
    import zlib
    out = []
    stack = list(parts if isinstance(parts, (tuple, list)) else [parts])
    for p in stack:
        if isinstance(p, (tuple, list)):
            out.extend(_spawn_key(p))
        elif isinstance(p, (int, np.integer)):
            out.append(int(p) % (2 ** 31))
        else:
            out.append(zlib.crc32(repr(p).encode()) % (2 ** 31))
    return tuple(out)


def rng_for(cell_index, repeat_index) -> np.random.Generator:
    """Registered per-(cell, repeat) PCG64 stream."""
    ss = np.random.SeedSequence(MASTER_SEED, spawn_key=_spawn_key((cell_index, repeat_index)))
    return np.random.Generator(np.random.PCG64(ss))


def rng_world(family_id, draw_index) -> np.random.Generator:
    ss = np.random.SeedSequence(MASTER_SEED, spawn_key=_spawn_key(("world", family_id, draw_index)))
    return np.random.Generator(np.random.PCG64(ss))


# ----------------------------------------------------------------------------- world


def equal_blocks(n_bins: int, K: int) -> np.ndarray:
    """Contiguous equal-width score blocks (declared world structure)."""
    return (np.arange(n_bins) * K) // n_bins


@dataclass
class World:
    """Stratified world at micro-bin granularity, declared score order.

    `cells` is the DECLARED stratified-shift structure (the B′ class 𝒦): on-class
    worlds have w = q/p exactly piecewise-constant on these cells (enforced by the
    family constructors via `_project_w_onclass`); off-class variation exists only in
    the B4.4 misspecification worlds (perturb_offclass).
    """
    p: np.ndarray            # source bin masses (sum 1)
    q: np.ndarray            # target bin masses (sum 1)
    eta: np.ndarray          # per-bin conditional risk P(L=1 | bin)
    name: str = "world"
    B_class: float = 5.0     # declared bounded-ratio class constant
    meta: dict = field(default_factory=dict)   # kappa, s0, beta_star, ... (oracle side)
    cells: np.ndarray | None = None            # declared cell_of_bin (K-cell class)

    def __post_init__(self):
        if self.cells is None:
            self.cells = equal_blocks(len(self.p), min(16, len(self.p)))

    @property
    def w(self) -> np.ndarray:
        return self.q / self.p

    @property
    def n_bins(self) -> int:
        return len(self.p)

    # ---- oracle/population quantities (diagnostics + validity harness ONLY; the
    #      certificate never receives a World object — R039 §7/§8)
    def coverage(self) -> np.ndarray:
        return np.cumsum(self.q)

    def risk(self) -> np.ndarray:
        cov = np.cumsum(self.q)
        num = np.cumsum(self.q * self.eta)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(cov > 0, num / cov, 0.0)

    def frontier(self, alpha: float) -> tuple[float, int]:
        """beta*(alpha) over the prefix lattice and the oracle-frontier prefix index."""
        cov, risk = self.coverage(), self.risk()
        feas = risk <= alpha
        if not feas.any():
            return 0.0, -1
        j = int(np.flatnonzero(feas)[np.argmax(cov[feas])])
        return float(cov[feas].max()), j

    def localized(self, j: int) -> float:
        """E_P[w^2 S]/(E_P[w S])^2 for prefix j (accepted bins 0..j)."""
        sel = slice(0, j + 1)
        ews = float((self.p[sel] * self.w[sel]).sum())
        ew2s = float((self.p[sel] * self.w[sel] ** 2).sum())
        return ew2s / ews ** 2 if ews > 0 else np.inf

    def global_ess_frac(self) -> float:
        return float(1.0 / (self.p * self.w ** 2).sum())

    # ---- sampling (sufficient statistics; per-(cell, repeat) streams)
    def sample_counts(self, rng, n_w: int, n_r: int, m_w: int, m_f: int):
        cnt_wP = rng.multinomial(n_w, self.p)
        cnt_r = rng.multinomial(n_r, self.p)
        loss_r = rng.binomial(cnt_r, self.eta)
        cnt_wQ = rng.multinomial(m_w, self.q)
        cnt_f = rng.multinomial(m_f, self.q)
        return cnt_wP, cnt_r, loss_r, cnt_wQ, cnt_f

    def sample_source(self, rng, n: int):
        cnt = rng.multinomial(n, self.p)
        loss = rng.binomial(cnt, self.eta)
        return cnt, loss

    def sample_target(self, rng, m: int):
        return rng.multinomial(m, self.q)


def cells_from_source_quantiles(world: World, K: int) -> np.ndarray:
    """Label-blind macro-cells: equal SOURCE-mass contiguous bins (R039 §2 analogue).
    Deterministic given world.p — population-level registration for synthetic suites."""
    cum = np.cumsum(world.p)
    edges = np.searchsorted(cum, np.arange(1, K) / K, side="left")
    cell = np.zeros(world.n_bins, int)
    for e in edges:
        cell[e + 1:] += 1
    return np.minimum(cell, K - 1)


def lattice_prefix(world: World, n_lat: int) -> np.ndarray:
    """Prefix lattice endpoints: n_lat equally spaced SOURCE-quantile prefixes."""
    cum = np.cumsum(world.p)
    bins = np.searchsorted(cum, np.arange(1, n_lat + 1) / (n_lat + 1), side="left")
    return np.unique(np.minimum(bins, world.n_bins - 1))


# ------------------------------------------------------------------- (i) continuum


def _project_w_onclass(q: np.ndarray, w_raw: np.ndarray, cells: np.ndarray):
    """Project a raw weight profile onto the declared K-cell class: w constant per
    cell (target-mass-weighted cell average), then p = q/w renormalized so that
    w = q/p is EXACTLY piecewise constant. Keeps on-class worlds inside 𝒦."""
    w_cell = np.empty_like(w_raw)
    for k in np.unique(cells):
        idx = cells == k
        w_cell[idx] = (q[idx] @ w_raw[idx]) / q[idx].sum()
    p = q / w_cell
    p = p / p.sum()
    return p


def family_continuum(intensity: int, nbins: int = 400, alpha: float = 0.2,
                     eta_lo: float = 0.05, eta_hi: float = 0.55, g: float = 0.55,
                     beta_star_target: float | None = None, K_class: int = 4) -> World:
    """Two-block regular-margin continuum family with bounded-ratio tilt, ON-CLASS:
    w is piecewise constant on K_class declared equal-width cells (the B′ stratified
    class), so the registered ρ_λ fully prices the weight-estimation error.

    eta(x) = eta_lo for x < g, eta_hi for x >= g; target q uniform => frontier
    beta* ≈ g·(1 + (alpha-eta_lo)/(eta_hi-alpha)) on the lattice; kappa = eta_hi - alpha
    (linear budget law phi(beta*-s) = kappa*s). beta_star_target (used by the
    beta-ladder runs) repositions g so the frontier sits at the requested value.
    Shift intensity L1..L4 = exponential source tilt, clipped to the class bound.
    """
    levels = {1: (0.7, 2.0), 2: (1.2, 3.0), 3: (1.8, 4.0), 4: (2.4, 5.0)}
    r, B_cls = levels[int(intensity)]
    c_front = 1.0 + (alpha - eta_lo) / (eta_hi - alpha)
    # finer bins for small frontier targets so the actual lattice frontier can hit
    # the request (round-2 review fix: discretized beta* drift in the beta->0 limb)
    if beta_star_target is not None and beta_star_target < 0.12:
        nbins = max(nbins, int(np.ceil(40.0 / beta_star_target / K_class) * K_class))

    def build(g_val: float) -> World:
        x = (np.arange(nbins) + 0.5) / nbins
        q = np.full(nbins, 1.0 / nbins)
        eta = np.where(x < g_val, eta_lo, eta_hi)
        cells = equal_blocks(nbins, K_class)
        w_raw = np.clip(np.exp(r * (x - 0.5)), 1.0 / B_cls, B_cls)
        p = _project_w_onclass(q, w_raw, cells)
        return World(p=p, q=q, eta=eta, name=f"continuum_L{intensity}",
                     B_class=B_cls, cells=cells)

    if beta_star_target is not None:
        g = float(np.clip(beta_star_target / c_front, 2.0 / nbins, 0.95))
        lo, hi = max(g / 2, 1.0 / nbins), min(g * 2, 0.97)
        for _ in range(24):                    # bisection: frontier is monotone in g
            mid = (lo + hi) / 2
            if build(mid).frontier(alpha)[0] < beta_star_target:
                lo = mid
            else:
                hi = mid
        g = hi
    world = build(g)
    beta_star, j_star = world.frontier(alpha)
    kappa = eta_hi - alpha
    world.meta = {"alpha": alpha, "kappa": kappa, "s0": beta_star - g,
                  "beta_star": beta_star, "j_star": j_star, "intensity": intensity,
                  "tilt": r, "w_max": float(world.w.max()), "g": g,
                  "beta_star_target": beta_star_target}
    return world


def family_nontheorem(nbins: int = 400, alpha: float = 0.2, intensity: int = 2,
                      beta_star_target: float | None = None, K_class: int = 4) -> World:
    """Non-theorem-aligned family (R044): irregular margin — eta has plateaus and a
    sub-linear approach to the frontier (phi flat spots break the regular
    left-neighborhood margin). NOT engineered to satisfy THEOREM 1 assumptions.
    Still ON-CLASS for the shift (w piecewise constant on declared cells) — the
    non-alignment is in the margin geometry, not in 𝒦."""
    r, B_cls = {1: (0.7, 2.0), 2: (1.2, 3.0), 3: (1.8, 4.0), 4: (2.4, 5.0)}[intensity]
    if beta_star_target is not None and beta_star_target < 0.12:
        nbins = max(nbins, int(np.ceil(40.0 / beta_star_target / K_class) * K_class))

    def build(t_scale: float) -> World:
        x = (np.arange(nbins) + 0.5) / nbins
        q = np.full(nbins, 1.0 / nbins)
        # irregular margin anchored at the frontier zone x ~ t_scale: square-root
        # kink + widening plateaus in u = x/t - 1 (shape is scale-portable, so the
        # frontier can be bisected over the FULL (0,1) range — round-2 review fix)
        u = np.maximum(x / t_scale - 1.0, 0.0)
        eta = np.clip(0.06 + 0.45 * np.sqrt(u) + 0.04 * np.floor(5 * u ** 1.5),
                      0.02, 0.9)
        cells = equal_blocks(nbins, K_class)
        w_raw = np.clip(np.exp(r * (x - 0.5)), 1.0 / B_cls, B_cls)
        p = _project_w_onclass(q, w_raw, cells)
        return World(p=p, q=q, eta=eta, name="nontheorem_irregular", B_class=B_cls,
                     cells=cells)

    shift0 = 0.45
    if beta_star_target is not None:
        lo, hi = 1.0 / nbins, 0.97         # frontier monotone increasing in t_scale
        for _ in range(30):
            mid = (lo + hi) / 2
            if build(mid).frontier(alpha)[0] < beta_star_target:
                lo = mid
            else:
                hi = mid
        shift0 = hi
    world = build(shift0)
    beta_star, j_star = world.frontier(alpha)
    world.meta = {"alpha": alpha, "beta_star": beta_star, "j_star": j_star,
                  "kappa": None, "s0": None, "irregular_margin": True,
                  "shift0": shift0, "beta_star_target": beta_star_target}
    return world


# ------------------------------------------------------------------ (iii) K-cell B'


def family_kcell(K_world: int = 4, loc_level: float = 2.0, B_cls: float = 5.0,
                 alpha: float = 0.2, nbins: int = 256) -> World:
    """B'-class stratified world: w piecewise constant on K_world=4 declared equal
    blocks, so EVERY refining partition K ∈ {4,8,16,32,64} (256 divisible) keeps the
    world inside 𝒦 — the K-sweep (B4.2) measures the PRICE of finer partitions, never
    silent misspecification. Accepted-region heterogeneity (localized ratio) is
    controlled by loc_level via the two accepted blocks' weight split."""
    assert nbins % 64 == 0, "nbins must be divisible by every sweep K"
    x = (np.arange(nbins) + 0.5) / nbins
    q = np.full(nbins, 1.0 / nbins)
    eta = 0.04 + 0.6 * x ** 2
    cells = equal_blocks(nbins, K_world)
    # block weights: accepted half = blocks 0,1 at levels (a/loc_level, a); rejected
    # half balances mean 1 under q-uniform; clipped to the class bound
    a = 1.6
    w_blocks = np.array([a / loc_level, a, 0.9, 0.0])
    w_blocks[3] = 4.0 - w_blocks[:3].sum()           # E_Q-uniform mean over 4 blocks
    w_blocks = np.clip(w_blocks, 1.0 / B_cls, B_cls)
    w_raw = w_blocks[cells].astype(float)
    p = _project_w_onclass(q, w_raw, cells)
    world = World(p=p, q=q, eta=eta, name=f"kcell_loc{loc_level}", B_class=B_cls,
                  cells=cells)
    beta_star, j_star = world.frontier(alpha)
    world.meta = {"alpha": alpha, "beta_star": beta_star, "j_star": j_star,
                  "K_world": K_world, "loc_level": loc_level}
    return world


# --------------------------------------------------------------- (iv) separation


def family_separation(alpha: float = 0.2) -> World:
    """Separation instance: actual w benign (=1) on the accepted low-risk half, heavy
    only on rejected bins; the worst case over the whole box {w <= B} concentrates
    hypothetical mass on accepted errors => DRO-box certifies nothing while the
    floor-aware estimated-w certificate certifies at moderate n."""
    K = 8
    eta = np.array([0.06, 0.06, 0.08, 0.08, 0.55, 0.6, 0.7, 0.8])
    p = np.array([0.125] * 8)
    w = np.array([1.0, 1.0, 1.0, 1.0, 0.6, 0.8, 1.2, 1.4])
    q = p * w
    q = q / q.sum()
    world = World(p=p, q=q, eta=eta, name="separation", B_class=5.0,
                  cells=np.arange(8))          # cells = strata (exactly on-class)
    beta_star, j_star = world.frontier(alpha)
    world.meta = {"alpha": alpha, "beta_star": beta_star, "j_star": j_star,
                  "working_beta": 0.45}
    return world


# ----------------------------------------------------------------- null worlds (R003)


def family_null(kind: int, alpha: float = 0.2) -> World:
    """Null-calibration worlds: the (alpha, beta) pair used by the harness is
    INFEASIBLE (beta set above beta*), so any issued certificate is a violation.
    kind indexes structurally different nulls."""
    if kind == 0:
        w = family_continuum(2)
    elif kind == 1:
        w = family_kcell(loc_level=2.5)
    elif kind == 2:
        w = family_separation()
    else:
        w = family_nontheorem()
    w.name = f"null_{w.name}"
    w.meta["null"] = True
    return w


def null_beta(world: World, margin: float = 0.02) -> float:
    """Infeasible floor: beta = beta* + margin (capped below 1)."""
    return min(world.meta["beta_star"] + margin, 0.99)


# ------------------------------------------------------- (B2) matched-pair families


def _solve_pair_member(a: float, G: float, q: np.ndarray):
    """4-stratum member: accepted w = a (uniform on strata 0,1), rejected (y_lo, y_hi)
    solving sum q w = G and sum q / w = 1 (valid p)."""
    qa = q[0] + q[1]
    S = (G - qa * a) / (q[2] + q[3]) * 2.0                 # y_lo + y_hi if q2=q3
    Hs = (1.0 - qa / a) / q[2]                             # 1/y_lo + 1/y_hi if q2=q3
    if S <= 0 or Hs <= 0:
        return None
    P = S / Hs
    disc = S * S - 4.0 * P
    if disc < 0:
        return None
    y_lo = (S - np.sqrt(disc)) / 2.0
    y_hi = (S + np.sqrt(disc)) / 2.0
    if y_lo <= 0:
        return None
    return np.array([a, a, y_lo, y_hi])


def family_matched_pair(ratio: float, draw: int, fam: int = 1, alpha: float = 0.2):
    """Matched-pair worlds: identical (q, eta) across members => identical frontier and
    working point; sum q w = G matched => identical global ESS; accepted-level a
    differs => localized functional ratio = a_HI / a_LO = `ratio`.

    fam=1: 4 strata (primary). fam=2: 8 strata, heterogeneous q/eta (structurally
    different family, reduced density). Returns (world_LO, world_HI, audit_dict) or
    None when the construction is infeasible at this draw (caller regenerates).
    """
    rng = rng_world(f"pair_f{fam}", draw)
    if fam == 1:
        q = np.array([0.25, 0.25, 0.25, 0.25])
        q = q + rng.uniform(-0.02, 0.02, 4)
        q[2:] = q[2:].mean()                    # keep rejected strata symmetric
        q = q / q.sum()
        eta_a = rng.uniform(0.04, 0.08)
        eta_r = rng.uniform(0.62, 0.78)
        eta = np.array([eta_a, eta_a, eta_r, eta_r])
        G = rng.uniform(1.35, 1.65)
        # feasible accepted levels: between the two roots of a^2 - 2Ga + G = 0 (sym case)
        a_min = G - np.sqrt(G * G - G)
        a_max = G + np.sqrt(G * G - G)
        # choose pair with localized ratio = `ratio`, centered geometrically
        a_lo = max(a_min * 1.02, np.sqrt(a_min * a_max / ratio))
        a_hi = a_lo * ratio
        if a_hi > a_max * 0.98:
            a_hi = a_max * 0.98
            a_lo = a_hi / ratio
            if a_lo < a_min * 1.02:
                return None
        members = []
        for a in (a_lo, a_hi):
            wv = _solve_pair_member(a, G, q)
            if wv is None or wv.max() > 5.0 or wv.min() < 0.18:
                return None
            members.append(wv)
    else:
        # fam2 v2 (structurally different, 6 strata): heterogeneous (q, eta), 3-level
        # accepted weights. v1 (8 strata) is preserved in git history — its K=8
        # nuisance pushed B' required-n to ~1e9, censoring 2/3 of worlds at any
        # practical bracket cap (an instrument power failure, documented in results;
        # not evidence about the localized functional).
        qa = rng.dirichlet(np.ones(3) * 30) * 0.5
        qr = rng.dirichlet(np.ones(3) * 30) * 0.5
        q = np.concatenate([qa, qr])
        eta = np.concatenate([rng.uniform(0.03, 0.10, 3), rng.uniform(0.55, 0.8, 3)])
        G = rng.uniform(1.35, 1.65)
        a_min = G - np.sqrt(G * G - G)
        a_max = G + np.sqrt(G * G - G)
        a_lo = max(a_min * 1.05, np.sqrt(a_min * a_max / ratio))
        a_hi = a_lo * ratio
        if a_hi > a_max * 0.95:
            return None
        members = []
        for a in (a_lo, a_hi):
            # accepted: three-level a*(0.9, 1.0, 1.1), renormed to mean a under qa
            wa = a * np.array([0.9, 1.0, 1.1])
            wa = wa * (a * qa.sum() / (qa @ wa))
            S_target = G - float(qa @ wa)                   # required sum qr*wr
            H_target = 1.0 - float((qa / wa).sum())         # required sum qr/wr
            # rejected two-level (y1, y1, y2): solve qr-weighted sum & harmonic
            q1 = qr[:2].sum(); q2 = qr[2]
            sol = None
            for y1 in np.linspace(0.2, 5.0, 2000):
                y2 = (S_target - q1 * y1) / q2
                if y2 <= 0.18 or y2 > 5.0:
                    continue
                if abs(q1 / y1 + q2 / y2 - H_target) < 2e-3:
                    sol = (y1, y2)
                    break
            if sol is None:
                return None
            wv = np.concatenate([wa, [sol[0], sol[0], sol[1]]])
            if wv.max() > 5.0 or wv.min() < 0.15:
                return None
            members.append(wv)
    worlds = []
    for tag, wv in zip(("LO", "HI"), members):
        p = q / wv
        p = p / p.sum()
        wld = World(p=p, q=q, eta=eta, name=f"pair_f{fam}_{tag}_r{ratio:.2f}_d{draw}",
                    B_class=5.0, cells=np.arange(len(q)))   # cells = strata (on-class)
        beta_star, j_star = wld.frontier(alpha)
        wld.meta = {"alpha": alpha, "beta_star": beta_star, "j_star": j_star,
                    "accepted_prefix": (len(q) // 2) - 1}
        worlds.append(wld)
    lo, hi = worlds
    jacc = (len(q) // 2) - 1
    audit = {
        "global_ess_lo": lo.global_ess_frac(), "global_ess_hi": hi.global_ess_frac(),
        "ess_rel_dev": abs(lo.global_ess_frac() - hi.global_ess_frac())
                       / lo.global_ess_frac(),
        "beta_star_lo": lo.meta["beta_star"], "beta_star_hi": hi.meta["beta_star"],
        "frontier_dev": abs(lo.meta["beta_star"] - hi.meta["beta_star"]),
        "localized_lo": lo.localized(jacc), "localized_hi": hi.localized(jacc),
        "localized_ratio": hi.localized(jacc) / lo.localized(jacc),
    }
    return lo, hi, audit


AUDIT_TOL = {"ess_rel_dev": 0.01, "frontier_dev": 0.005, "ratio_rel_dev": 0.10}


def audit_pair(audit: dict, target_ratio: float) -> bool:
    """Published audit rule (plan §B2): ESS match <=1%, frontier match within tolerance,
    achieved localized ratio within 10% of target. Failing pairs are REGENERATED at the
    next draw index — never silently kept."""
    return (audit["ess_rel_dev"] <= AUDIT_TOL["ess_rel_dev"]
            and audit["frontier_dev"] <= AUDIT_TOL["frontier_dev"]
            and abs(audit["localized_ratio"] / target_ratio - 1.0) <= AUDIT_TOL["ratio_rel_dev"])


# --------------------------------------------------- (B4.4) misspecification worlds


def perturb_offclass(world: World, cells: np.ndarray, eps: float, mode: str) -> World:
    """Perturb the TRUE shift off the K-cell stratified class (B4.4 epsilon grids).
    mode in {"tilt", "gradient", "tv"}; eps units registered per family:
      tilt     — relative amplitude of a smooth within-cell tilt of q
      gradient — relative within-cell linear gradient of q
      tv       — total-variation mass moved from the lowest-eta accepted cell to the
                 highest-eta cell (adversarial direction)
    """
    q = world.q.copy()
    nb = world.n_bins
    x = (np.arange(nb) + 0.5) / nb
    if mode == "tilt":
        q = q * (1.0 + eps * np.sin(2 * np.pi * 5 * x))
    elif mode == "gradient":
        K = cells.max() + 1
        for k in range(K):
            idx = np.flatnonzero(cells == k)
            if len(idx) < 2:
                continue
            t = np.linspace(-1, 1, len(idx))
            q[idx] = q[idx] * (1.0 + eps * t)
    elif mode == "tv":
        order = np.argsort(world.eta)
        lo_idx, hi_idx = order[: nb // 8], order[-nb // 8:]
        move = min(eps, q[lo_idx].sum() * 0.95)
        q[lo_idx] -= move * q[lo_idx] / q[lo_idx].sum()
        q[hi_idx] += move * q[hi_idx] / q[hi_idx].sum()
    q = np.maximum(q, 1e-12)
    q = q / q.sum()
    out = World(p=world.p.copy(), q=q, eta=world.eta.copy(),
                name=f"{world.name}_off_{mode}{eps}", B_class=world.B_class,
                cells=world.cells.copy())   # certificate keeps the DECLARED cells;
                                            # the q-perturbation puts TRUTH off-class

    beta_star, j_star = out.frontier(world.meta.get("alpha", 0.2))
    out.meta = dict(world.meta, beta_star=beta_star, j_star=j_star,
                    offclass=(mode, eps))
    return out
