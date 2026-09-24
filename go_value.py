#!/usr/bin/env python3
"""Compare OpenCode Go models by Artificial Analysis quality vs Go usage limits.

Combines the OpenCode Go model list / usage limits (parsed fresh from
https://opencode.ai/docs/go/#usage-limits on every run, and cross-checked
against the live endpoint https://opencode.ai/zen/go/v1/models) with
benchmark data from the Artificial Analysis free API
(https://artificialanalysis.ai/api/v2/data/llms/models, key required).

Data source attribution (required by AA ToS): https://artificialanalysis.ai/

Usage:
    export AA_API_KEY=***        # from artificialanalysis.ai account
    python3 go_value.py                        # live AA + live Go docs fetch
    python3 go_value.py --demo                 # live Go catalog, synthetic AA data
    python3 go_value.py --csv out.csv --show

Every run writes a self-contained HTML report (charts embedded as base64)
to <prefix>_report.html, plus charts, CSV (optional) and JSON data.
Nothing is cached and no model catalog is stored locally: the Go catalog
(limits/pricing) is re-parsed from the live docs MDX and AA data is
re-fetched on every run, so the report always reflects current data.
Network access is required; a failed live fetch is a fatal error.
"""

import argparse
import base64
import csv
import difflib
import hashlib
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, asdict
from html import escape as esc
from string import Template

# --------------------------------------------------------------------------
# OpenCode Go catalog: no local snapshot is stored. The catalog is always
# parsed fresh from the live docs MDX (see GO_DOCS_MDX_URL) on every run.
# Prices are USD per 1M tokens, lowest tier / off-peak where tiered.
# Token profiles are the docs' "typical request".
# --------------------------------------------------------------------------


@dataclass
class GoModel:
    id: str                 # model id on opencode.ai/zen/go (config: opencode-go/<id>)
    name: str               # display name in docs
    req_5h: int             # allowed requests per 5 hours
    req_week: int           # allowed requests per week
    req_month: int          # allowed requests per month
    price_in: float         # $/1M uncached input tokens
    price_out: float        # $/1M output tokens
    price_cache: float      # $/1M cached-read tokens
    tok_in: int             # typical input tokens per request
    tok_cache: int          # typical cached tokens per request
    tok_out: int            # typical output tokens per request
    usage_usd: float        # included usage value per month (per-model $ tier)
    notes: str = ""
    aa_hint: str = ""       # optional Artificial Analysis slug/name override
    trains: bool = False    # docs ## Privacy table: model trains on your prompts
    retention: str = ""     # docs ## Privacy table: data retention (e.g. "0 days")

ZEN_MODELS_URL = "https://opencode.ai/zen/go/v1/models"
# Live docs source (upstream OpenCode docs). Override with --go-docs-url
# only if upstream moves. No fallback snapshot exists: a failed fetch
# is a fatal error so reports never reflect stale data.
GO_DOCS_MDX_URL = ("https://raw.githubusercontent.com/anomalyco/opencode/dev/"
                   "packages/web/src/content/docs/go.mdx")
AA_URL = "https://artificialanalysis.ai/api/v2/data/llms/models"
AA_ATTRIBUTION = "Benchmarks: artificialanalysis.ai"
DEFAULT_OVERRIDES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "aa_match_overrides.json")

STOP_TOKENS = {
    "api", "compatible", "endpoint", "online", "latest", "high", "medium",
    "low", "think", "thinking", "reason", "reasoning", "instruct", "chat",
    "base", "beta", "nova", "version", "tier", "batch", "contributor",
}

# Modality/maturity markers that must agree on both sides. "exp" is deliberately
# NOT a stop token: "Vision Exp" must not fuzzy-match the base "Vision" model.
VARIANT_TOKENS = frozenset({"vision", "exp", "preview", "image", "video", "audio"})


# --------------------------------------------------------------------------
# HTTP helpers
# --------------------------------------------------------------------------

def http_get_json(url, headers=None, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": "go-value-analyzer/1.0", **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_zen_model_ids():
    """Live list of model ids currently served by OpenCode Go (public)."""
    data = http_get_json(ZEN_MODELS_URL)
    return sorted(m["id"] for m in data.get("data", []))


def fetch_aa_models(api_key):
    """Artificial Analysis LLM endpoint, fetched live on every run
    (rate limit 1000/day — one request per run)."""
    if not api_key:
        sys.exit("ERROR: no Artificial Analysis API key.\n"
                 "  Create one at https://artificialanalysis.ai (free account), then:\n"
                 "    export AA_API_KEY=***   # or use --api-key *** or run with --demo")
    try:
        payload = http_get_json(AA_URL, headers={"x-api-key": api_key})
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:300]
        hint = {401: " (invalid/missing key)", 429: " (rate limit, 1000 req/day)"}.get(e.code, "")
        sys.exit(f"ERROR: Artificial Analysis API returned {e.code}{hint}: {body}")
    except urllib.error.URLError as e:
        sys.exit(f"ERROR: could not reach artificialanalysis.ai: {e.reason}")
    return payload["data"]


def demo_aa_models(catalog):
    """Deterministic SYNTHETIC data so the pipeline can be inspected without an AA key.
    The Go catalog itself is still fetched live; only the AA benchmarks are synthetic."""
    # Omit quota extremes so the unmatched path is always exercised,
    # without hardcoding any model id (catalog is live and rotates).
    skip_ids = set()
    by_quota = sorted(catalog, key=lambda g: g.req_month)
    for gm in ([by_quota[0]] + by_quota[-1:] if by_quota else []):
        skip_ids.add(gm.id)
        if len(skip_ids) >= 2:
            break
    out = []
    for gm in catalog:
        if gm.id in skip_ids:
            continue  # intentionally omitted to exercise the unmatched path
        h = int(hashlib.sha256(gm.id.encode()).hexdigest(), 16)
        rng = lambda i, lo, hi: lo + ((h >> i) & 0xFFFF) / 0xFFFF * (hi - lo)
        out.append({
            "name": gm.name, "slug": gm.id,
            "model_creator": {"name": "?"},
            "evaluations": {
                "artificial_analysis_intelligence_index": round(rng(0, 25, 68), 1),
            },
            "pricing": {"price_1m_blended_3_to_1": round(rng(32, 0.1, 6), 2)},
            "median_output_tokens_per_second": round(rng(48, 35, 220), 1),
            "median_time_to_first_token_seconds": round(rng(64, 0.3, 9), 2),
        })
    return out


# --------------------------------------------------------------------------
# OpenCode Go catalog. Always parsed fresh from the live docs MDX on every
# run. No snapshot, no cache.
# --------------------------------------------------------------------------

def _clean_cell(s):
    """Flatten markdown/HTML inside a docs table cell. Promo rows strike out
    superseded values and attach notes as inline HTML, e.g.
    "~~6,500~~<br />**26,000**" or "~~$15~~ **$60**<br /><small>4x · Ends
    Sep 27</small>". Keep only the current (non-struck) value; drop notes."""
    s = re.sub(r"<small>.*?</small>", " ", s, flags=re.S | re.I)
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"~~.*?~~", " ", s, flags=re.S)
    s = s.replace("**", "").replace("`", "")
    return re.sub(r"\s+", " ", s).strip()


def _md_cells(line):
    return [_clean_cell(c) for c in line.strip().strip("|").split("|")]


def _parse_int(s):
    s = s.replace(",", "").strip()
    return int(s) if re.fullmatch(r"\d+", s) else None


def _parse_money(s):
    # Docs wrap the monthly limit as **$60** (bold) — strip markdown first.
    s = s.replace("$", "").replace(",", "").strip().strip("*_`").strip()
    return float(s) if re.fullmatch(r"[\d.]+", s) else None


def _md_tables(text):
    tables, lines, i = [], text.splitlines(), 0
    while i < len(lines):
        sep = lines[i + 1].strip() if i + 1 < len(lines) else ""
        if (lines[i].strip().startswith("|") and "-" in sep
                and set(sep) <= set("|-: ")):
            hdr, rows, j = _md_cells(lines[i]), [], i + 2
            while j < len(lines) and lines[j].strip().startswith("|"):
                rows.append(_md_cells(lines[j]))
                j += 1
            tables.append((hdr, rows))
            i = j
        else:
            i += 1
    return tables


def expand_shared_names(name):
    """'GLM-5.3/5.2/5.1' -> three names; 'Kimi K2.7/K2.6' -> two names."""
    if "/" not in name:
        return [name.strip()]
    parts = [p.strip() for p in name.split("/")]
    first, out = parts[0], [parts[0]]
    for p in parts[1:]:
        if re.fullmatch(r"[\d.]+", p):  # version-only segment shares the prefix
            out.append(re.sub(r"\d+(?:\.\d+)*$", "", first) + p)
        else:
            out.append(first.rsplit(" ", 1)[0] + " " + p)
    return [o for o in out if o]


def parse_go_catalog(mdx):
    """Build [GoModel] from the docs MDX. Returns (models, skipped_names)."""
    reqs, prices, profiles, ids, notes, privacy = {}, {}, {}, {}, {}, {}
    for hdr, rows in _md_tables(mdx):
        h = [x.lower() for x in hdr]

        def col(sub):
            # Prefer an exact header hit; fall back to substring for
            # docs renames ("requests per week" vs "per week").
            for k, x in enumerate(h):
                if x.strip() == sub:
                    return k
            return next((k for k, x in enumerate(h) if sub in x), None)

        c5, cm = col("requests per 5 hour"), col("requests per month")
        ctr, cret = col("model training"), col("data retention")
        # Docs renamed the "Usage" column to "Monthly limit" (Sep 2026) —
        # accept either header so the price table keeps parsing.
        c_usage = col("usage") or col("monthly limit") or col("limit")
        if ctr is not None and cret is not None:
            for r in rows:
                name = re.sub(r"\s*\([^()]*\)\s*$", "", r[0]).strip()
                if not name or not re.search(r"[a-z0-9]", name, re.I):
                    continue
                cell = (r[ctr] if ctr < len(r) else "").strip().lower()
                privacy[name] = (cell.startswith("yes"),
                                 (r[cret] if cret < len(r) else "").strip())
        elif c5 is not None and cm is not None:
            for r in rows:
                name = re.sub(r"\s*\([^()]*\)\s*$", "", r[0]).strip()
                vals = [_parse_int(r[c] if c is not None and c < len(r) else "") for c in (c5, col("per week"), cm)]
                if all(v is not None for v in vals):
                    reqs[name] = tuple(vals)
        elif col("input") is not None and col("cached read") is not None and c_usage is not None:
            for r in rows:
                m = re.match(r"(.+?)\s*(?:\(([^()]*)\)\s*)?$", r[0])
                base, suffix = m.group(1).strip(), (m.group(2) or "").strip()
                if suffix.startswith(">") or suffix.lower() == "peak":  # keep cheapest tier only
                    continue
                pin, pout, pcr, usage = (_parse_money(r[c]) if c < len(r) else None
                                         for c in (col("input"), col("output"), col("cached read"), c_usage))
                if None in (pin, pout, pcr):
                    continue
                # Docs may list several tiers for one base model; keep cheapest.
                prev = prices.get(base)
                if prev is not None and (prev[0], prev[1]) <= (pin, pout):
                    continue
                prices[base] = (pin, pout, pcr, usage or 0.0)
                if suffix:
                    notes[base] = "off-peak pricing" if "off-peak" in suffix.lower() else f"{suffix} tier"
        elif col("model id") is not None:
            for r in rows:
                if len(r) > 1 and re.search(r"[a-z0-9]", r[1]):
                    ids[re.sub(r"\s*\([^()]*\)\s*$", "", r[0]).strip()] = r[1].strip().strip("`")
    for m in re.finditer(r"^- (.+?)\s*[-—–]\s*([\d,]+) input, ([\d,]+) cached, ([\d,]+) output", mdx, re.M):
        for name in expand_shared_names(m.group(1)):
            profiles[name] = tuple(int(x.replace(",", "")) for x in m.groups()[1:])

    def key(s):
        return frozenset(tokens(s))

    # Keep the original display name alongside each value so fuzzy joins can
    # still recover the right per-model note (tier/off-peak) afterwards.
    prof_idx = {key(k): (k, v) for k, v in profiles.items()}
    price_idx = {key(k): (k, v) for k, v in prices.items()}
    id_idx = {key(k): (k, v) for k, v in ids.items()}
    priv_idx = {key(k): (k, v) for k, v in privacy.items()}

    def lookup(idx, k):
        if k in idx:
            orig, v = idx[k]
            return v, True, orig
        cands = [(len(c ^ k), orig, v) for c, (orig, v) in idx.items()
                 if (c <= k or k <= c) and len(c ^ k) <= 2]
        if not cands:
            return None, False, None
        _, orig, v = min(cands)
        return v, False, orig

    models, skipped, missing_priv = [], [], []
    for name, (r5, rw, rm) in reqs.items():
        k = key(name)
        pf, pf_exact, _ = lookup(prof_idx, k)
        pr, pr_exact, pr_orig = lookup(price_idx, k)
        if not pf or not pr:
            skipped.append(name)
            continue
        mid = lookup(id_idx, k)[0] or re.sub(r"[^a-z0-9.]+", "-", name.lower()).strip("-")
        note = notes.get(pr_orig or "", "")
        if not pf_exact or not pr_exact:
            note = "; ".join(filter(None, [note, "docs tables joined fuzzily — verify"]))
        pv, _, _ = lookup(priv_idx, k)
        if pv is None:
            missing_priv.append(name)
        trains, retention = pv if pv is not None else (False, "")
        models.append(GoModel(id=mid, name=name, req_5h=r5, req_week=rw, req_month=rm,
                              price_in=pr[0], price_out=pr[1], price_cache=pr[2],
                              tok_in=pf[0], tok_cache=pf[1], tok_out=pf[2],
                              usage_usd=pr[3], notes=note,
                              trains=trains, retention=retention))
    if not privacy:
        print("[go] warning: no ## Privacy table parsed — training flags default to False",
              file=sys.stderr)
    elif missing_priv:
        print(f"[go] warning: no privacy row for: {', '.join(sorted(missing_priv))} "
              f"— training flags default to False", file=sys.stderr)
    return sorted(models, key=lambda m: m.name), skipped


def fetch_go_catalog(url=GO_DOCS_MDX_URL):
    """Parse the docs catalog fresh from the live MDX. No snapshot, no cache:
    a failed live fetch is a fatal error, never silently served from stale data."""
    try:
        with urllib.request.urlopen(urllib.request.Request(
                url, headers={"User-Agent": "go-value-analyzer/1.0"}), timeout=30) as resp:
            mdx = resp.read().decode("utf-8", "replace")
        models, skipped = parse_go_catalog(mdx)
        if len(models) < 5:
            raise ValueError(f"parsed only {len(models)} models — docs structure may have changed")
    except SystemExit:
        raise
    except Exception as e:
        sys.exit(f"ERROR: Go docs fetch/parse failed: {e}\n"
                 f"  Source: {url}\n"
                 "  Network access is required — no offline snapshot exists.")
    if skipped:
        print(f"[go] warning: docs limit rows without matching price or token-profile (excluded): {', '.join(skipped)}")
    print(f"[go] catalog: parsed {len(models)} models from live docs MDX")
    return models, f"live OpenCode Go docs ({time.strftime('%Y-%m-%d %H:%M')})"


# --------------------------------------------------------------------------
# Name matching: OpenCode Go model -> Artificial Analysis model
# --------------------------------------------------------------------------

def tokens(s):
    return [t for t in re.findall(r"[a-z0-9]+(?:\.[0-9]+)*", s.lower()) if t not in STOP_TOKENS]


def has_digits(t):
    return {x for x in t if any(c.isdigit() for c in x)}


def match_one(go_model, aa_models, overrides):
    """Return (aa_entry|None, quality, matched_name). quality in exact/fuzzy/low/none."""
    forced = overrides.get(go_model.id) or (go_model.aa_hint or None)
    if forced:
        wanted = forced.lower()
        for m in aa_models:
            if wanted in ((m.get("slug") or "").lower(), (m.get("name") or "").lower()):
                return m, "exact (override)", m.get("name")
        print(f"[match] warning: override for {go_model.id!r} -> {forced!r} matched nothing; "
              f"falling back to automatic matching", file=sys.stderr)

    want = set(tokens(go_model.name))
    want_digits = has_digits(want)
    want_variant = want & VARIANT_TOKENS
    best, best_key, best_quality = None, None, None
    for m in aa_models:
        for label in (m.get("name", ""), m.get("slug", "")):
            cand = set(tokens(label))
            if not cand:
                continue
            if has_digits(cand) != want_digits:
                continue
            if (cand & VARIANT_TOKENS) != want_variant:
                continue  # vision/exp/preview must agree; never score base as variant
            if want <= cand or cand <= want:
                extra = len(cand ^ want)
                quality = "exact" if extra == 0 else "fuzzy"
                sim = difflib.SequenceMatcher(None, " ".join(sorted(want)), " ".join(sorted(cand))).ratio()
                key = (extra, -sim)
                if best_key is None or key < best_key:
                    best, best_key, best_quality = m, key, quality
    if best is not None:
        return best, best_quality, best.get("name")
    # Nothing compatible: only suggest when digits agree, variant markers agree,
    # and at least one token overlaps — otherwise stay silent ("none").
    cands = []
    for m in aa_models:
        nm = m.get("name", "")
        if not nm:
            continue
        toks = set(tokens(nm))
        if has_digits(toks) != want_digits:
            continue
        if (toks & VARIANT_TOKENS) != want_variant:
            continue
        if not (toks & want):
            continue
        cands.append(nm)
    close = difflib.get_close_matches(go_model.name, cands, n=1, cutoff=0.75)
    if close:
        entry = next((m for m in aa_models if m.get("name") == close[0]), None)
        return entry, "low", close[0]
    return None, "none", None


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

@dataclass
class Row:
    id: str
    name: str
    req_5h: int
    req_week: int
    req_month: int
    cost_req: float          # $ per typical request at Go prices
    value_mult: float        # included usage $ / subscription price
    tokens_month: float      # est. total tokens/month the request limits allow
    aa_name: str
    match_quality: str
    intelligence: float | None
    tok_s: float | None
    ttft_s: float | None
    notes: str = ""
    bang: float | None = None       # bang-for-buck, computed after scoring pass
    trains: bool = False            # docs ## Privacy: trains on your prompts
    retention: str = ""             # docs ## Privacy: data retention


# Fixed half-saturation constants for the quota axis: the request count that
# scores v=50. Kept FIXED so scores stay comparable across runs as the
# catalog rotates. Ratio 5:1 mirrors the docs (req_month ≈ 5 × req_5h), so
# the two windows agree on v for typical models and the min() only bites
# when a model is unusually burst- or month-constrained. K is deliberately
# small (well below the median quota) so differences above ~30k/mo compress
# to noise: beyond that, ranking is driven by quality, not quota.
QUOTA_K_MO = 8000.0
QUOTA_K_5H = 1600.0

# Peak quotas: v tops out here, then sheds QUOTA_DECAY points per 10x beyond
# the peak — insanely high quotas score a touch BELOW merely-generous ones
# instead of winning forever. Only the far outliers (currently 150k+/226k
# req/mo) land past the peak; the worst penalty in today's catalog is ~3pts.
QUOTA_PEAK_MO = 60000.0
QUOTA_PEAK_5H = 12000.0
QUOTA_DECAY = 5.0

# Quality handling: convex stretch q' = 100*(q/100)^QUALITY_GAMMA (smooth,
# never zero unless q is zero) so perceived gaps count — 57 vs 73 is much
# worse, not 22% worse — without ever zeroing a model out. BANG_ALPHA > 0.5
# makes quality dominate quota: BANG = q'^alpha * v^(1-alpha).
QUALITY_GAMMA = 2.0
BANG_ALPHA = 0.7


def quota_value(req, k, peak):
    """Hill saturation with a slight peak-and-decline: 100*req/(req+k) up to
    peak, then minus QUOTA_DECAY points per 10x over peak. Scarce quota gains
    almost linearly, generous quota plateaus, absurd quota gently degrades —
    still no hard cap, v just stops rewarding excess."""
    if req is None or req <= 0:
        return None
    v = 100.0 * req / (req + k)
    if req > peak:
        v -= QUOTA_DECAY * math.log10(req / peak)
    return v


def score_rows(rows):
    """bang = q'^alpha × v^(1-alpha) where q' = 100*(q/100)^QUALITY_GAMMA is
    the convex-stretched raw AA intelligence index (smooth, only zero if q is
    zero) and v = min(v_mo, v_5h) peaks around 60k req/mo (12k/5h), then
    sheds a few points per 10x beyond the peak, across the monthly and
    5-hour windows (bottleneck: a model must satisfy both)."""
    for r in rows:
        v_mo = quota_value(r.req_month, QUOTA_K_MO, QUOTA_PEAK_MO)
        v_5h = quota_value(r.req_5h, QUOTA_K_5H, QUOTA_PEAK_5H)
        v = None if v_mo is None or v_5h is None else min(v_mo, v_5h)
        q = r.intelligence
        if q is None or v is None:
            r.bang = None
        else:
            qs = 100.0 * (max(q, 0.0) / 100.0) ** QUALITY_GAMMA
            r.bang = round(math.exp(
                BANG_ALPHA * math.log(max(qs, 1e-9)) + (1.0 - BANG_ALPHA) * math.log(max(v, 1e-9))), 1)


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

def fmt_num(x, spec=","):
    return f"{x:{spec}.0f}" if x is not None else "—"


def fmt_tokens(x):
    if x is None:
        return "—"
    if x >= 1e9:
        return f"{x / 1e9:,.1f}B"
    if x >= 1e6:
        return f"{x / 1e6:,.0f}M"
    return f"{x / 1e3:,.0f}K"


def print_table(rows):
    cols = [
        ("model", lambda r: r.name[:26], 26, "<"),
        ("req/5h", lambda r: fmt_num(r.req_5h), 7, ">"),
        ("req/mo", lambda r: fmt_num(r.req_month), 8, ">"),
        ("$/req", lambda r: f"{r.cost_req:.4f}", 7, ">"),
        ("val×", lambda r: f"{r.value_mult:.0f}x", 4, ">"),
        ("tok/mo", lambda r: fmt_tokens(r.tokens_month), 7, ">"),
        ("intell", lambda r: "—" if r.intelligence is None else f"{r.intelligence:.1f}", 6, ">"),
        ("t/s", lambda r: "—" if not r.tok_s else f"{r.tok_s:.0f}", 5, ">"),
        ("BANG", lambda r: "—" if r.bang is None else f"{r.bang:.0f}", 5, ">"),
        ("data", lambda r: "TRAINS" if r.trains else "—", 6, "<"),
        ("match", lambda r: r.match_quality, 15, "<"),
    ]
    line = "  ".join(f"{{:{align}{w}}}" for _, _, w, align in cols)
    print(line.format(*[h for h, _, _, _ in cols]))
    print("-" * (sum(w for _, _, w, _ in cols) + 2 * (len(cols) - 1)))
    for r in sorted(rows, key=lambda r: -(r.bang or -1)):
        print(line.format(*[get(r) for _, get, _, _ in cols]))


def make_charts(rows, path_prefix, sub_price, demo, variant="all"):
    """Render the scatter + bar PNGs. variant="notrain"
    renders the same charts with training models removed (filenames gain a
    "_notrain" suffix); the report swaps the two sets via the checkbox."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter
    except ImportError:
        print("[charts] matplotlib not installed — skipping graphs (pip install matplotlib)", file=sys.stderr)
        return []

    fname_suffix = "" if variant == "all" else "_notrain"

    suffix = "  [DEMO — SYNTHETIC DATA]" if demo else ""
    tag = f"{AA_ATTRIBUTION} · OpenCode Go limits · ${sub_price:.0f}/mo sub"
    palette = plt.get_cmap("tab10")
    charts = []

    label = "intelligence index"
    fscatter, fbar = "_intell_vs_requests", "_bang_for_buck"
    matched = [r for r in rows if r.intelligence is not None and r.req_month and r.req_month > 0]
    excluded = [r.name for r in rows if r.intelligence is None]

    def footnote(fig):
        fig.text(0.01, 0.01, tag, fontsize=7, color="gray")
        if excluded:
            fig.text(0.01, 0.033, "not plotted (no AA " + label + " data yet): "
                     + ", ".join(excluded), fontsize=7.5, color="#666666")

    if matched:
        # Pareto front: no model is both smarter AND gives more requests.
        by_quota = sorted(matched, key=lambda r: r.req_month)
        front, best_q = [], float("-inf")
        for r in reversed(by_quota):
            if r.intelligence > best_q:
                front.append(r)
                best_q = r.intelligence
        front.sort(key=lambda r: r.req_month)

        fig, ax = plt.subplots(figsize=(11, 7.5))
        ax.plot([r.req_month for r in front], [r.intelligence for r in front],
                "--", lw=1.2, color="gray", zorder=1, label="Pareto front (best value)")
        for r in matched:
            size = min(320, 60 + 14 * r.value_mult)  # clamp: huge value_mult must not eat the plot
            ax.scatter(r.req_month, r.intelligence, s=size, zorder=2,
                       color=palette(0 if r in front else 1),
                       edgecolor="black", linewidth=0.5)
            ax.annotate(r.name, (r.req_month, r.intelligence),
                        textcoords="offset points", xytext=(6, 5), fontsize=7.5)
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(FuncFormatter(
            lambda v, _: f"{v:,.0f}" if v < 1000 else (f"{v / 1000:g}K" if v < 1000000 else f"{v / 1000000:g}M")))
        ax.set_xlabel("requests per month included by OpenCode Go (log scale)")
        ax.set_ylabel(f"Artificial Analysis {label}")
        ax.set_title(f"AA {label} vs. Go usage limit (bubble size = included usage $)" + suffix)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        footnote(fig)
        p = f"{path_prefix}{fscatter}{fname_suffix}.png"
        fig.tight_layout(rect=(0, 0.055, 1, 1)); fig.savefig(p, dpi=150); plt.close(fig)
        charts.append(dict(
            path=p, variant=variant,
            title="Quality vs. usage limit &mdash; the money graph",
            caption="x-axis: your monthly request quota (log scale), y-axis: AA intelligence index. Bubble size = included "
                    "usage value for that model (per-model $ tier). Blue dots lie on the <b>Pareto front</b>: no other model "
                    "is both smarter <i>and</i> gives you more requests, so the dashed line marks the rational choices. "
                    "Top-right is the sweet spot."))

        scored = [r for r in rows if r.bang is not None]
        if scored:
            scored.sort(key=lambda r: r.bang)
            fig, ax = plt.subplots(figsize=(10, max(4, 0.32 * len(scored))))
            ax.barh([r.name for r in scored], [r.bang for r in scored],
                    color=[palette(2) if r not in front else palette(0) for r in scored])
            for r in scored:
                ax.text(r.bang + 0.8, r.name, f"{r.bang:.0f}", va="center", fontsize=8)
            ax.set_xlabel("bang-for-buck score  =  quality^0.7 × quota-value^0.3   ·   0–100")
            ax.set_title("OpenCode Go models ranked by quality-per-quota" + suffix)
            ax.set_xlim(0, 108)
            ax.grid(axis="x", alpha=0.3)
            footnote(fig)
            p = f"{path_prefix}{fbar}{fname_suffix}.png"
            fig.tight_layout(rect=(0, 0.055, 1, 1)); fig.savefig(p, dpi=150); plt.close(fig)
            charts.append(dict(
                path=p, variant=variant,
                title="Bang-for-buck ranking",
                caption="BANG = quality<sup>0.7</sup> &times; quota-value<sup>0.3</sup>, where quality is the raw AA intelligence index "
                        "with a smooth convex stretch (100&middot;(q/100)<sup>2</sup>, so 57 vs 73 counts as much worse, not 22% worse, "
                        "and nothing ever zeroes out) "
                        "and quota-value = 100&middot;req/(req+K) peaking around 60k/mo (12k/5h, bottleneck of "
                        "both windows), then shedding a few points per 10&times; beyond the peak — absurd quotas score a touch "
                        "below merely-generous ones instead of winning forever. Blue bars are also Pareto-optimal in "
                        "the graph above. Use it as a shortlist generator, not gospel &mdash; quality deliberately outweighs quota; if you "
                        "never hit quotas, just read the raw score column."))

    return charts


def write_csv(rows, path, sub_price):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["go_model_id", "name", "req_per_5h", "req_per_week", "req_per_month",
                    "usd_per_request_goprice", "included_usage_value_per_month", "value_multiplier_vs_sub",
                    "est_tokens_per_month", "aa_model", "match", "aa_intelligence_index",
                    "aa_tok_per_s", "aa_ttft_s", "bang",
                    "trains_on_data", "data_retention", "notes"])
        for r in sorted(rows, key=lambda r: -(r.bang or -1)):
            w.writerow([r.id, r.name, r.req_5h, r.req_week,
                        r.req_month, round(r.cost_req, 6), round(r.value_mult * sub_price, 2), round(r.value_mult, 3),
                        int(r.tokens_month), r.aa_name, r.match_quality, r.intelligence,
                        r.tok_s or "", r.ttft_s or "", r.bang,
                        "yes" if r.trains else "no", r.retention, r.notes])


# --------------------------------------------------------------------------
# HTML report (single self-contained file, charts embedded as base64)
# --------------------------------------------------------------------------

REPORT_TEMPLATE = Template("""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OpenCode Go — bang for buck</title>
<style>
:root{--bg:#0d1117;--panel:#161b22;--border:#30363d;--fg:#e6edf3;--mut:#8d96a0;--acc:#58a6ff;--good:#3fb950;--warn:#d29922;--bad:#f85149}
*{box-sizing:border-box}
body{margin:0;font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;background:var(--bg);color:var(--fg)}
.wrap{max-width:1000px;margin:0 auto;padding:44px 20px 80px}
h1{font-size:30px;margin:0 0 6px;letter-spacing:-.01em}
.sub{color:var(--mut);margin:0}
h2{font-size:20px;margin:52px 0 14px;border-bottom:1px solid var(--border);padding-bottom:8px;letter-spacing:-.01em}
h3{font-size:15px;margin:26px 0 8px}
p{margin:10px 0}
.banner{background:rgba(210,153,34,.12);border:1px solid var(--warn);color:#ffd9a0;padding:12px 16px;border-radius:8px;font-weight:600}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px;margin:22px 0 4px}
.card{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:16px 18px}
.card .tag{margin:0 0 4px;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:var(--mut)}
.card .big{font-size:21px;font-weight:650;color:var(--acc)}
.card p{margin:6px 0 0;color:var(--mut);font-size:13px}
img.chart{width:100%;border:1px solid var(--border);border-radius:10px;background:#fff}
.caption{color:var(--mut);font-size:13px;margin:8px 0 26px}
.tscroll{overflow-x:auto;background:var(--panel);border:1px solid var(--border);border-radius:10px}
table{width:100%;border-collapse:collapse;font-size:13.5px}
th,td{padding:8px 10px;text-align:right;border-bottom:1px solid var(--border);white-space:nowrap}
tbody tr:last-child td{border-bottom:none}
th{color:var(--mut);font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.05em}
table.sortable th{cursor:pointer;user-select:none;position:relative;padding-right:20px}
table.sortable th:hover{color:var(--fg)}
table.sortable th::after{content:"\\2195";position:absolute;right:7px;opacity:.35;font-weight:400}
table.sortable th[data-sort=asc]::after{content:"\\2191";opacity:1;color:var(--acc)}
table.sortable th[data-sort=desc]::after{content:"\\2193";opacity:1;color:var(--acc)}
th.nos{cursor:default}
th.nos::after{content:none}
td.l,th.l{text-align:left}
tr.top td{background:rgba(63,185,80,.07)}
td.name{font-weight:600}
.bang{font-weight:700;color:var(--good)}
.badge{display:inline-block;padding:1px 8px;border-radius:20px;font-size:11px;font-weight:600}
.badge.ok{background:rgba(63,185,80,.15);color:var(--good)}
.badge.warn{background:rgba(210,153,34,.15);color:var(--warn)}
.badge.bad{background:rgba(248,81,73,.15);color:var(--bad)}
dl.glossary dt{font-weight:650;margin-top:12px}
dl.glossary dd{margin:2px 0 0;color:var(--mut)}
code{background:var(--panel);border:1px solid var(--border);border-radius:5px;padding:1px 6px;font-size:12.5px}
footer{margin-top:60px;color:var(--mut);font-size:12.5px;border-top:1px solid var(--border);padding-top:16px}
a{color:var(--acc)}
.mbnote{color:var(--mut);font-size:12.5px}
.privbar{display:flex;align-items:center;gap:8px;margin:14px 0 2px;font-size:13.5px;color:var(--mut)}
.privbar input{accent-color:var(--warn);width:15px;height:15px}
.privbar b{color:var(--fg)}
.callout{background:rgba(210,153,34,.10);border:1px solid var(--warn);border-radius:10px;padding:12px 16px;margin:18px 0;font-size:13.5px}
.callout ul{margin:8px 0 2px;padding-left:20px}
.callout li{margin:4px 0;color:var(--mut)}
.callout li b{color:var(--fg)}
.sugg{color:var(--mut);font-size:11.5px}
</style>
</head>
<body>
<div class="wrap">
$demo_banner
<h1>OpenCode Go: bang for buck</h1>
<p class="sub">Artificial Analysis quality &times; OpenCode Go usage limits &nbsp;&middot;&nbsp; $$$sub/mo subscription &nbsp;&middot;&nbsp; generated $generated</p>
$privacy_filter
$unmatched
$privacy_callout
<div class="cards cards-all"$cards_all_style>
$cards
</div><div class="cards cards-notrain"$cards_notrain_style>
$cards_notrain
</div>
<p><b>BANG</b> is this report's composite score: <code>quality<sup>0.7</sup> &times; quota-value<sup>0.3</sup></code>, where quality is the raw
AA intelligence index with a smooth convex stretch &mdash; 57 vs 73 counts as much worse, not 22%
worse, and weak scores shrink toward zero without ever zeroing out &mdash; and quota-value is <code>100&middot;req/(req+K)</code> with
K=8k req/mo and K=1.6k req/5h, bottlenecked as the <i>lower</i> of the two windows, peaking around 60k req/mo (12k req/5h)
and shedding a few points per 10&times; beyond that: absurd quotas score a touch below merely-generous ones instead of
winning forever. A model only scores high if it is
<i>good first</i> and <i>sufficiently metered</i>.
It is a heuristic for this specific $$$sub/mo subscription, not an official benchmark.</p>
$charts
<h2>Full leaderboard &mdash; ranked by BANG</h2>
<p class="mbnote">Rows without a BANG score have no Artificial Analysis benchmark yet. The pairing between Go
model names and AA names is automated, so give the <b>Match</b> column a glance before trusting a score.
Click any column header to sort; click again to reverse.</p>
<div class="tscroll"><table class="sortable">
<thead><tr><th class="l nos">#</th><th class="l">Model</th><th class="l">Match</th><th class="l">Privacy</th><th>req/5h</th><th>req/mo</th><th>$/req</th><th>value/mo</th><th>tokens/mo</th><th>AA intell.</th><th>tok/s</th><th data-sort="desc">BANG</th></tr></thead>
<tbody>
$tbody
</tbody></table></div>
<h3>Column glossary</h3>
<dl class="glossary">
<dt>req/5h &middot; req/mo</dt><dd>Requests allowed per 5-hour window / per month. Go limits are defined in dollar value
($$12 / 5h, $$30 / week, $$60 / month) and the docs translate that into per-model request counts using a typical request profile.</dd>
<dt>$/req</dt><dd>Estimated cost of one typical request at Go token prices (docs' input / cached / output token mix per model).</dd>
<dt>value/mo</dt><dd>Included usage value at the monthly limit (the per-model $ tier, e.g. $15/$30/$60/$100) and its multiplier over the subscription price.</dd>
<dt>tokens/mo</dt><dd>Roughly how many tokens/month the request quota allows (req/mo &times; typical tokens per request). Cached-read tokens dominate this figure.</dd>
<dt>AA intell. / tok/s</dt><dd>Artificial Analysis intelligence index and median output speed (tokens/s). Higher is better; the index goes 0&ndash;100 and measures general reasoning ability.</dd>
<dt>BANG</dt><dd>The bang-for-buck score explained above (0&ndash;100, higher is better value). Quality enters as the raw AA intelligence index, quota as the saturating quota-value (bottleneck of both windows).</dd>
<dt>Match</dt><dd>Confidence the AA scores shown belong to this exact model: <b>exact</b> name match, <b>fuzzy</b> (name variant, e.g. publisher prefix), <b>none</b> (no AA data yet, or only an unverified guess which is never scored).</dd>
<dt>Privacy</dt><dd>From the Go docs <i>Privacy</i> table: whether the provider trains models on your prompts, plus data retention. The checkbox above (on by default) swaps charts, cards and leaderboard rows to versions without the flagged <b>trains</b> models.</dd>
</dl>
<footer>
Sources: <a href="https://opencode.ai/docs/go/">OpenCode Go docs</a> &amp; the <code>zen/go/v1/models</code> endpoint
(catalog: $go_src)
&middot; benchmarks &amp; speed: <a href="https://artificialanalysis.ai/">Artificial Analysis</a>
(attribution required by their free-API terms). Generated by <code>go_value.py</code> on $generated.
</footer>
</div>
<script>
document.querySelectorAll("table.sortable").forEach(function (tb) {
  var ths = Array.prototype.slice.call(tb.querySelectorAll("thead th"));
  ths.forEach(function (th, ci) {
    if (th.classList.contains("nos")) return;
    th.onclick = function () {
      var dir = th.dataset.sort === "asc" ? "desc" : "asc";
      ths.forEach(function (h) { delete h.dataset.sort; });
      th.dataset.sort = dir;
      var tbody = tb.tBodies[0];
      var rows = Array.prototype.slice.call(tbody.rows);
      var numeric = rows.some(function (tr) {
        var v = tr.cells[ci].dataset.v || "";
        return /^[-.0-9]/.test(v) && !isNaN(+v);
      });
      rows.sort(function (a, b) {
        var av = a.cells[ci].dataset.v || "", bv = b.cells[ci].dataset.v || "";
        if (!av && !bv) return 0;
        if (!av) return 1;   // missing values always sink to the bottom
        if (!bv) return -1;
        var c = numeric ? (+av) - (+bv) : av.localeCompare(bv);
        return dir === "asc" ? c : -c;
      });
      rows.forEach(function (tr) { tbody.appendChild(tr); });
      renumber(tb);
    };
  });
});
function renumber(tb) {
  var n = 1;
  Array.prototype.forEach.call(tb.tBodies[0].rows, function (tr) {
    if (tr.style.display !== "none") { tr.cells[0].textContent = n++; }
  });
}
function applyHideTrain(hide) {
  document.querySelectorAll(".chart-all").forEach(function (el) { el.style.display = hide ? "none" : ""; });
  document.querySelectorAll(".chart-notrain").forEach(function (el) { el.style.display = hide ? "" : "none"; });
  document.querySelectorAll(".cards-all").forEach(function (el) { el.style.display = hide ? "none" : ""; });
  document.querySelectorAll(".cards-notrain").forEach(function (el) { el.style.display = hide ? "" : "none"; });
  document.querySelectorAll('tr[data-trains="1"]').forEach(function (tr) {
    tr.style.display = hide ? "none" : "";
  });
  document.querySelectorAll("table.sortable").forEach(renumber);
}
var ht = document.getElementById("hide-train");
if (ht) {
  ht.addEventListener("change", function () { applyHideTrain(ht.checked); });
  // browsers can restore checked state on reload/bfcache without firing change;
  // sync the page to whatever state was restored
  var syncHideTrain = function () { applyHideTrain(ht.checked); };
  syncHideTrain();
  window.addEventListener("pageshow", syncHideTrain);
}
</script>
</body></html>
""")

def write_html(rows, charts, sub_price, demo, go_src, path, excluded_training=False):
    def card(tag, name, stat, note):
        return (f'<div class="card"><p class="tag">{tag}</p><div class="big">{esc(name)}</div>'
                f'<p>{stat}</p><p>{note}</p></div>')

    # charts -> concatenated <h2>+<img>+<caption> html, grouped by title.
    # Each chart pairs the full-data PNG with its training-excluded twin
    # (when present); the checkbox (checked by default) swaps the two via JS.
    # default_hide is True when a filtered view exists, so the initial HTML
    # already shows the filtered view (correct even with JS disabled); the
    # JS syncs to the checkbox state on load/pageshow.
    default_hide = any(r.trains for r in rows) and any(not r.trains for r in rows)

    def chart_img(c, extra_cls, hidden):
        try:
            with open(c["path"], "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
        except OSError as e:
            print(f"[html] warning: chart {c['path']} unreadable ({e}), skipped", file=sys.stderr)
            return ""
        return (f'<img class="chart {extra_cls}" alt="{esc(c["title"])}" '
                f'{"style=\"display:none\"" if hidden else ""}'
                f'src="data:image/png;base64,{b64}">')

    charts_html = ""
    grouped, order = {}, []
    for c in charts:
        k = c["title"]
        if k not in grouped:
            grouped[k] = {}
            order.append(k)
        grouped[k][c.get("variant", "all")] = c
    for title in order:
        pair = grouped[title]
        if "all" not in pair:
            continue
        imgs = chart_img(pair["all"], "chart-all", default_hide if "notrain" in pair else False)
        if "notrain" in pair:
            imgs += chart_img(pair["notrain"], "chart-notrain", not default_hide)
        if not imgs:
            continue
        charts_html += (
            f'<h2>{pair["all"]["title"]}</h2>{imgs}'
            f'<p class="caption">{pair["all"]["caption"]}</p>\n')

    def build_view(vrows):
        if not vrows:
            return "<p>No models in catalog.</p>", ""
        matched = [r for r in vrows if r.intelligence is not None]
        scored = sorted((r for r in vrows if r.bang is not None), key=lambda r: -r.bang)
        # Front here must match make_charts.
        by_quota = sorted(matched, key=lambda r: r.req_month)
        front, best_q = [], float("-inf")
        for r in reversed(by_quota):
            if r.intelligence > best_q:
                front.append(r)
                best_q = r.intelligence
        front_ids = {r.id for r in front}

        cards = []
        train_note = ' <span class="badge warn" title="uses prompts to train future models">trains on data</span>'
        if scored:
            b = scored[0]
            pareto = "On the Pareto front." if b.id in front_ids else ""
            cards.append(card("Top value (BANG)", b.name,
                              f"BANG {b.bang:.0f} &middot; intelligence {b.intelligence:.1f} &middot; {b.req_month:,} req/mo"
                              + (train_note if b.trains else ""),
                              f"Best balance of quality and quota. {pareto}"))
        if matched:
            t = max(matched, key=lambda r: r.intelligence)
            cards.append(card("Highest intelligence quality", t.name,
                              f"intelligence index {t.intelligence:.1f} &middot; {t.req_month:,} req/mo"
                              + (train_note if t.trains else ""),
                              "Quality leader &mdash; but note how tight its request quota is. Pick it if you rarely "
                              "exhaust the 5-hour window." if t.req_month < 10000 else
                              "Smart <i>and</i> generously metered."))
        q = max(vrows, key=lambda r: r.req_month)
        cards.append(card("Quota king", q.name,
                          f"{q.req_month:,} req/mo &middot; " + ("quality n/a" if q.intelligence is None else f"intelligence {q.intelligence:.1f}")
                          + (train_note if q.trains else ""),
                          "You will run out of subscription months before you run out of requests here."))

        body = []
        ordered = scored + [r for r in vrows if r.bang is None]
        for i, r in enumerate(ordered, 1):
            q_txt = esc(r.match_quality)
            cls = "ok" if q_txt.startswith("exact") else ("warn" if q_txt in ("fuzzy", "low") else "bad")
            badge = f'<span class="badge {cls}" title="match confidence">{q_txt}</span>'
            if r.aa_name.startswith("suggested: "):
                badge += f' <span class="sugg">&asymp; {esc(r.aa_name[11:])}?</span>'
            val = f"${r.value_mult * sub_price:g}&nbsp;({r.value_mult:.0f}&times;)"
            sv = r.bang
            num = lambda x: "&mdash;" if x is None else f"{x:.1f}"
            bang_cls = ' class="bang"' if sv is not None else ""
            bang = f"{sv:.0f}" if sv is not None else "&mdash;"
            conf_rank = {"ok": 0, "warn": 1, "bad": 2}[cls]
            dv_int = "" if r.intelligence is None else f"{r.intelligence:.1f}"
            if r.trains:
                ret = esc(r.retention) if r.retention else "unknown"
                priv = (f'<span class="badge warn" title="uses prompts to train future models &middot; '
                        f'retention: {ret}">trains</span>')
            else:
                ret = f"retention: {esc(r.retention)}" if r.retention else "no training per Go docs"
                priv = f'<span class="badge ok" title="{ret}">no train</span>'
            body.append(
                f'<tr{" class=\"top\"" if i <= 3 else ""}{" data-trains=\"1\"" if r.trains else ""}'
                f'{" style=\"display:none\"" if r.trains and default_hide else ""}>'
                f'<td class="l">{i}</td>'
                f'<td class="l name" data-v="{esc(r.name.lower())}">{esc(r.name)}</td>'
                f'<td class="l" data-v="{conf_rank}">{badge}</td>'
                f'<td class="l" data-v="{1 if r.trains else 0}">{priv}</td>'
                f'<td data-v="{r.req_5h}">{r.req_5h:,}</td>'
                f'<td data-v="{r.req_month}">{r.req_month:,}</td>'
                f'<td data-v="{r.cost_req:.6f}">{r.cost_req:.4f}</td>'
                f'<td data-v="{r.value_mult * sub_price:.4g}">{val}</td>'
                f'<td data-v="{r.tokens_month:.0f}">{fmt_tokens(r.tokens_month)}</td>'
                f'<td data-v="{dv_int}">{num(r.intelligence)}</td>'
                f'<td data-v="{"" if not r.tok_s else f"{r.tok_s:.1f}"}">{"&mdash;" if not r.tok_s else f"{r.tok_s:.0f}"}</td>'
                f'<td{bang_cls} data-v="{"" if sv is None else f"{sv:.1f}"}">{bang}</td></tr>')
        return "\n".join(cards), "\n".join(body)

    training = [r for r in rows if r.trains]
    cards, tbody = build_view(rows)
    rows_notrain = [r for r in rows if not r.trains]
    if training and rows_notrain:
        cards_notrain, _ = build_view(rows_notrain)
    else:
        cards_notrain = ""

    missing_aa = [r for r in rows if r.intelligence is None]
    if missing_aa:
        items = "".join(
            f'<li><b>{esc(r.name)}</b> ({r.req_month:,} req/mo included)'
            + (f' &mdash; AA has no benchmark entry yet; possibly <b>{esc(r.aa_name[11:])}</b>?'
               if r.aa_name.startswith("suggested: ") else " &mdash; AA has no benchmark entry yet")
            + "</li>" for r in missing_aa)
        unmatched = (f'<div class="callout"><b>{len(missing_aa)} catalog model(s) are in the leaderboard but cannot '
                     f'be scored or plotted yet:</b><ul>{items}</ul>These are typically brand-new releases that '
                     'Artificial Analysis has not benchmarked yet.</div>')
    else:
        unmatched = ""

    if training:
        items = "".join(
            f'<li><b>{esc(r.name)}</b> (retention: {esc(r.retention) or "unknown"})</li>'
            for r in training)
        privacy_callout = (f'<div class="callout"><b>{len(training)} model(s) use your prompts '
                           f'to train future models</b> (per the Go docs <i>Privacy</i> table):<ul>{items}</ul>'
                           f'They are hidden from the charts, cards and leaderboards by default &mdash; '
                           f'untick the checkbox above to show them, or re-run with '
                           f'<code>--exclude-training</code> to drop them from the whole report.</div>')
        privacy_filter = (f'<div class="privbar"><input type="checkbox" id="hide-train" checked>'
                          f'<label for="hide-train">Hide the <b>{len(training)} model(s)</b> that train '
                          f'on your data (Go docs <i>Privacy</i> table)</label></div>'
                          f'<p class="mbnote">The checkbox is on by default, showing versions without the '
                          f'flagged models &mdash; untick to show them. Re-run with '
                          f'<code>--exclude-training</code> to drop them from the whole report.</p>')
    elif excluded_training:
        privacy_callout = ""
        privacy_filter = ('<p class="mbnote">Models that train on your data were excluded '
                          'with <code>--exclude-training</code>.</p>')
    else:
        privacy_callout = ""
        privacy_filter = ""

    html_out = REPORT_TEMPLATE.safe_substitute(
        unmatched=unmatched,
        privacy_callout=privacy_callout,
        privacy_filter=privacy_filter,
        demo_banner='<div class="banner">&#9888; DEMO MODE: the benchmark scores below are deterministic '
                    'SYNTHETIC placeholders used to preview the layout &mdash; they are NOT real model quality.</div>\n'
        if demo else "",
        sub=f"{sub_price:g}",
        go_src=esc(go_src),
        generated=time.strftime("%Y-%m-%d %H:%M"),
        cards=cards, cards_notrain=cards_notrain,
        cards_all_style=' style="display:none"' if default_hide else "",
        cards_notrain_style="" if default_hide else ' style="display:none"',
        charts=charts_html, tbody=tbody,
    )
    with open(path, "w") as f:
        f.write(html_out)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api-key", default=os.environ.get("AA_API_KEY") or os.environ.get("ARTIFICIAL_ANALYSIS_API_KEY"),
                    help="Artificial Analysis API key (default: $AA_API_KEY)")
    ap.add_argument("--demo", action="store_true", help="use deterministic SYNTHETIC AA data (no AA key needed); Go catalog is still fetched live")
    ap.add_argument("--go-docs-url", default=GO_DOCS_MDX_URL, help="Go docs MDX source URL (default: upstream docs)")
    ap.add_argument("--sub-price", type=float, default=10.0, help="Go subscription price, $/month (default 10)")
    ap.add_argument("--prefix", default="go_value", help="output basename for charts/json (default go_value)")
    ap.add_argument("--csv", metavar="PATH", help="also write a CSV export")
    ap.add_argument("--exclude-training", action="store_true",
                    help="drop models that train on your data (docs Privacy table) from the whole "
                         "report; other models' BANG scores are unaffected (fixed anchors, no ranks)")
    ap.add_argument("--show", action="store_true", help="open charts in a viewer window")
    args = ap.parse_args()

    if not args.sub_price or args.sub_price <= 0:
        ap.error("--sub-price must be > 0")

    # 0. Go catalog: always parsed fresh from the live docs. No snapshot, no cache.
    go_models, go_src = fetch_go_catalog(url=args.go_docs_url)
    if not go_models:
        sys.exit("ERROR: Go catalog is empty — docs parse returned no models.")
    if args.exclude_training:
        n_before = len(go_models)
        go_models = [g for g in go_models if not g.trains]
        print(f"[go] excluding {n_before - len(go_models)} model(s) that train on your data (--exclude-training)")
        if not go_models:
            sys.exit("ERROR: --exclude-training removed every model — nothing to score.")

    overrides = {}
    if os.path.exists(DEFAULT_OVERRIDES):
        try:
            with open(DEFAULT_OVERRIDES) as f:
                overrides = json.load(f)
            if not isinstance(overrides, dict):
                print(f"[match] warning: {os.path.basename(DEFAULT_OVERRIDES)} is not a JSON object; ignoring",
                      file=sys.stderr)
                overrides = {}
            else:
                print(f"[match] loaded {len(overrides)} name override(s) from {os.path.basename(DEFAULT_OVERRIDES)}")
        except (json.JSONDecodeError, OSError) as e:
            print(f"[match] warning: could not read overrides ({e}); continuing without them", file=sys.stderr)
            overrides = {}

    # 1. Artificial Analysis data
    if args.demo:
        aa_models = demo_aa_models(go_models)
        print("!! DEMO MODE: benchmark numbers below are SYNTHETIC placeholders, not real data !!")
    else:
        aa_models = fetch_aa_models(args.api_key)

    # 2. live endpoint cross-check (always fresh, never cached)
    try:
        live_ids = fetch_zen_model_ids()
    except Exception as e:
        print(f"[zen] warning: could not reach {ZEN_MODELS_URL}: {e}", file=sys.stderr)
        live_ids = None

    # 3. build rows
    rows, fuzzy_warnings = [], []
    for gm in go_models:
        aa, quality, matched_name = match_one(gm, aa_models, overrides)
        if quality == "low":  # unverified guess: never let it score, only suggest
            fuzzy_warnings.append((gm.name, matched_name))
            aa, quality = None, "none"
            hint = f"suggested: {matched_name}" if matched_name else ""
            matched_display = hint
        else:
            matched_display = matched_name or ""
        ev = (aa or {}).get("evaluations", {}) or {}
        if matched_display and matched_display != gm.name and not matched_display.startswith("suggested: "):
            aa_note = f"AA name: {matched_display}"
        else:
            aa_note = ""
        # Never surface the "suggested:" guess in notes/CSV — it lives only in
        # the terminal hint and the HTML callout for genuinely unscored rows.
        rows.append(Row(
            id=gm.id, name=gm.name, req_5h=gm.req_5h, req_week=gm.req_week, req_month=gm.req_month,
            cost_req=(gm.tok_in * gm.price_in + gm.tok_cache * gm.price_cache + gm.tok_out * gm.price_out) / 1e6,
            value_mult=gm.usage_usd / args.sub_price,
            tokens_month=gm.req_month * (gm.tok_in + gm.tok_cache + gm.tok_out),
            aa_name=matched_display, match_quality=quality,
            intelligence=ev.get("artificial_analysis_intelligence_index"),
            tok_s=(aa or {}).get("median_output_tokens_per_second"),
            ttft_s=(aa or {}).get("median_time_to_first_token_seconds"),
            notes="; ".join(filter(None, [gm.notes, aa_note])),
            trains=gm.trains,
            retention=gm.retention,
        ))

    score_rows(rows)

    # 4. report
    print(f"\nOpenCode Go value analysis  ·  subscription ${args.sub_price:.0f}/mo  ·  "
          f"{time.strftime('%Y-%m-%d %H:%M')}\n")
    print_table(rows)

    unmatched = [r for r in rows if r.match_quality == "none"]
    if unmatched:
        print("\nNo Artificial Analysis data for: " + ", ".join(r.name for r in unmatched))
    training_rows = [r for r in rows if r.trains]
    if training_rows and not args.exclude_training:
        print("\nTrain on your data (docs Privacy table — hide with the report checkbox or "
              "--exclude-training): " + ", ".join(r.name for r in training_rows))
    if fuzzy_warnings:
        print("\nLow-confidence name matches — verify or pin them in aa_match_overrides.json "
              '({"<go-id>": "<aa slug or name>"}):')
        for go, cand in fuzzy_warnings:
            print(f"  {go}  ->  {cand}?")

    if live_ids is not None:
        embedded = {gm.id for gm in go_models}
        extra = [i for i in live_ids if i not in embedded]
        missing = sorted(embedded - set(live_ids))
        print(f"\nLive endpoint check ({ZEN_MODELS_URL}):")
        print(f"  {len(live_ids)} models served; docs catalog covers {len(embedded) - len(missing)} (source: {go_src})")
        if extra:
            print(f"  ! on live endpoint but no published limits in docs: {', '.join(extra)}")
        if missing:
            print(f"  ! in catalog but NOT on live endpoint any more: {', '.join(missing)}")
        if not extra and not missing:
            print("  docs catalog matches the live model list ✓")

    ranked = [r for r in rows if r.bang is not None]
    if ranked:
        print("\nTop value (bang-for-buck = quality^0.7 × saturating quota-value^0.3, 0–100):")
        for r in sorted(ranked, key=lambda r: -r.bang)[:5]:
            print(f"  {r.name:<28} bang {r.bang:>5.0f}   "
                  f"int {'—' if r.intelligence is None else f'{r.intelligence:.1f}'}   "
                  f"{r.req_month:,} req/mo   ({r.value_mult:.0f}x sub value)")

    # 5. artifacts
    charts = make_charts(rows, args.prefix, args.sub_price, args.demo)
    if not args.exclude_training and any(r.trains for r in rows):
        charts += make_charts([r for r in rows if not r.trains], args.prefix,
                              args.sub_price, args.demo, variant="notrain")
    for c in charts:
        print(f"\nwrote {c['path']}")
    if args.csv:
        write_csv(rows, args.csv, args.sub_price)
        print(f"wrote {args.csv}")
    with open(f"{args.prefix}_data.json", "w") as f:
        json.dump([asdict(r) for r in rows], f, indent=1)
    print(f"wrote {args.prefix}_data.json")
    html_path = f"{args.prefix}_report.html"
    write_html(rows, charts, args.sub_price, args.demo, go_src, html_path,
               excluded_training=args.exclude_training)
    print(f"wrote {html_path}")
    print(f"\n{AA_ATTRIBUTION}")

    if args.show and charts:
        try:
            import subprocess
            paths = [c["path"] for c in charts]
            if sys.platform == "darwin":
                subprocess.run(["open", *paths], check=False)
            elif sys.platform.startswith("win"):
                os.startfile(paths[0])  # noqa: S606 — local chart file, user-requested
                for p in paths[1:]:
                    os.startfile(p)  # noqa: S606
            else:
                subprocess.run(["xdg-open", *paths], check=False)
        except Exception as e:
            print(f"[show] {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
