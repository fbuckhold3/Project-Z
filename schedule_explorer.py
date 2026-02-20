"""
Project Z — Schedule Explorer (v16)
Interactive Streamlit app for exploring ABC residency scheduling configurations.
Senior schedule: 96 weeks (2 years). Intern schedule: 48 weeks (1 year).
Deploy to Posit Connect or run locally: streamlit run schedule_explorer.py
"""

import streamlit as st
import random
import collections
import io
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import string

# ═══════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Project Z — Schedule Explorer",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for compact grid and styling
st.markdown("""
<style>
    .stApp { max-width: 100%; }
    .schedule-grid { font-size: 10px; overflow-x: auto; }
    .schedule-grid table { border-collapse: collapse; width: auto; }
    .schedule-grid th { background: #2F5496; color: white; padding: 3px 4px;
        font-size: 9px; text-align: center; position: sticky; top: 0; z-index: 2; }
    .schedule-grid td { padding: 2px 3px; text-align: center; border: 1px solid #ddd;
        font-size: 9px; white-space: nowrap; min-width: 32px; }
    .kpi-card { background: white; border-radius: 10px; border: 1px solid #dfe3ea;
        padding: 16px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,.06); }
    .kpi-val { font-size: 28px; font-weight: 700; color: #4472C4; }
    .kpi-label { font-size: 11px; color: #888; margin-top: 4px; }
    .section-hdr { background: #D6DCE4; font-weight: 700; padding: 5px 8px; }
    .summary-hdr { background: #FFF2CC; font-weight: 700; }
    div[data-testid="stMetric"] { background: white; border: 1px solid #dfe3ea;
        border-radius: 8px; padding: 12px; }
    .year-hdr { background: #1a3a6b; color: #FFD700; font-weight: 700;
        font-size: 10px; text-align: center; padding: 2px 4px; }
    .jeo-dim { opacity: 0.2; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
# ROTATION COLORS AND LABELS
# ═══════════════════════════════════════════════════════════════════════
COLORS = {
    'SLUH': '#B3D9FF', 'VA': '#FFCC99', 'NF': '#D9B3FF', 'MICU': '#FFB3B3',
    'Cards': '#FFFFB3', 'OP': '#B3FFB3', 'Clinic': '#FFFFFF', 'Jeopardy': '#FFE0B2',
    'ID': '#C8E6C9', 'Bronze': '#FFE0B2', 'Diamond': '#E1BEE7', 'Gold': '#FFF9C4',
    'ICU*': '#FFD9D9', 'Elective': '#BDD7EE',
    'IP Other 1': '#B0BEC5', 'IP Other 2': '#90A4AE',
}
ABBREV = {
    'SLUH': 'SLUH', 'VA': 'VA', 'NF': 'NF', 'MICU': 'MICU', 'OP': 'OP',
    'Clinic': 'CL', 'ID': 'ID', 'Bronze': 'BRZ', 'Cards': 'CRD',
    'Diamond': 'DIA', 'Gold': 'GLD', 'Jeopardy': 'JEO', 'ICU*': 'ICU*',
    'IP Other 1': 'IP1', 'IP Other 2': 'IP2',
}
# Text colors for dark backgrounds
DARK_BG = {'SLUH', 'VA', 'NF', 'MICU'}

# Rotations that get role letters (A, B, C...) when multiple residents on same week
ROLE_LETTER_ROTS = {'MICU', 'Bronze', 'Cards', 'NF'}


# ═══════════════════════════════════════════════════════════════════════
# ROLE LETTER ASSIGNMENT (post-processing)
# ═══════════════════════════════════════════════════════════════════════
def assign_role_letters(residents, tw):
    """Assign role letters (A, B, C...) for MICU/Bronze/Cards/NF per week.
    Returns dict: (resident_id, week) -> letter string (e.g. 'A', 'B')
    """
    labels = {}
    for w in range(tw):
        # Group residents by rotation this week
        rot_groups = collections.defaultdict(list)
        for r in residents:
            rot = r['schedule'][w]
            if rot in ROLE_LETTER_ROTS:
                rot_groups[rot].append(r['id'])
        # Assign letters
        for rot, rids in rot_groups.items():
            sorted_rids = sorted(rids)
            for idx, rid in enumerate(sorted_rids):
                if len(sorted_rids) > 1:
                    labels[(rid, w)] = string.ascii_uppercase[idx % 26]
    return labels


# ═══════════════════════════════════════════════════════════════════════
# SENIOR SCHEDULER (v16: 96 weeks, min/max, stagger, ABABA post-clinic)
# ═══════════════════════════════════════════════════════════════════════
def build_senior(params):
    """Build senior PGY-2/3 schedule over 96 weeks (2 years)."""
    random.seed(params['seed'])

    N_PGY3 = params['n_pgy3']
    N_PGY2 = params['n_pgy2']
    N = N_PGY3 + N_PGY2
    TW = 96
    CYCLE = params.get('clinic_freq', 6)
    MAX_CONSEC = params.get('max_consec', 3)
    WARD_LEN = params.get('ward_len', 3)
    NF_LEN = params.get('nf_len', 2)
    JEOP_CAP = params.get('jeop_cap', 4)
    MAX_STAGGER = params.get('max_stagger', 2)

    TARGETS = {
        'SLUH': params['t_sluh'], 'VA': params['t_va'], 'ID': params['t_id'],
        'NF': params['t_nf'], 'MICU': params['t_micu'], 'Bronze': params['t_bronze'],
        'Cards': params['t_cards'], 'Diamond': params['t_diamond'], 'Gold': params['t_gold'],
        'IP Other 1': params.get('t_other1', 0), 'IP Other 2': params.get('t_other2', 0),
    }
    IP_ROTS = (set(TARGETS.keys()) - {'Jeopardy'}) | {'Elective'}

    # Remove zero-target rotations from TARGETS
    TARGETS = {k: v for k, v in TARGETS.items() if v > 0}

    IDEAL = {rot: (tgt * TW) / max(N, 1) for rot, tgt in TARGETS.items()}

    # Min/max per resident per rotation
    MIN_PER_RES = {}
    MAX_PER_RES = {}
    minmax = params.get('min_max', {})
    for rot in TARGETS:
        ideal_val = IDEAL.get(rot, 0)
        MIN_PER_RES[rot] = minmax.get(rot, {}).get('min', max(0, int(ideal_val * 0.4)))
        MAX_PER_RES[rot] = minmax.get(rot, {}).get('max', int(ideal_val * 1.8) + 1)

    # Build residents
    residents = []
    for pgy in [3, 2]:
        n = N_PGY3 if pgy == 3 else N_PGY2
        for i in range(n):
            residents.append({'id': f'R{pgy}_{i+1:02d}', 'pgy': pgy})

    # Clinic positions (6 staggered positions)
    pos_counts = {p: N // 6 + (1 if (p - 1) < (N % 6) else 0) for p in range(1, 7)}
    pos_list = []
    for p in sorted(pos_counts):
        pos_list.extend([p] * pos_counts[p])
    random.shuffle(pos_list)
    for i, r in enumerate(residents):
        r['cpos'] = pos_list[i] if i < len(pos_list) else (i % 6) + 1

    schedule = {r['id']: [''] * TW for r in residents}
    coverage = collections.defaultdict(lambda: [0] * TW)
    res_weeks = collections.defaultdict(lambda: collections.defaultdict(int))

    # Track block starts for stagger constraint
    start_count = collections.defaultdict(lambda: [0] * TW)

    # Assign clinics
    for r in residents:
        for c in range(TW // CYCLE):
            w = c * CYCLE + (r['cpos'] - 1)
            if w < TW:
                schedule[r['id']][w] = 'Clinic'
                coverage['Clinic'][w] += 1

    def is_free(rid, w):
        return 0 <= w < TW and schedule[rid][w] == ''

    def assign(rid, w, rot):
        schedule[rid][w] = rot
        coverage[rot][w] += 1

    # ── Scheduling quality parameters ──
    IP_WINDOW = params.get('ip_window', 6)       # sliding window size
    MAX_IP_WIN = params.get('max_ip_win', 3)     # max IP weeks in any window
    NF_MIN_GAP = params.get('nf_min_gap', 6)     # min weeks between NF blocks
    LOW_PRIO_ROTS = {'IP Other 1', 'IP Other 2'}  # lower-priority rotations

    def ip_window_ok(rid, weeks_to_fill, relaxed=False):
        """Check that placing IP in weeks_to_fill keeps every 6-wk window ≤ limit.
           If relaxed=True, allows MAX_IP_WIN + 1 (for repair passes)."""
        limit = MAX_IP_WIN + 1 if relaxed else MAX_IP_WIN
        temp = list(schedule[rid])
        for w in weeks_to_fill:
            temp[w] = 'IP'
        for start in range(max(0, min(weeks_to_fill) - IP_WINDOW + 1),
                           min(TW - IP_WINDOW + 1, max(weeks_to_fill) + 1)):
            count = sum(1 for w in range(start, start + IP_WINDOW)
                        if temp[w] in IP_ROTS or temp[w] == 'IP')
            if count > limit:
                return False
        return True

    def nf_gap_ok(rid, weeks_to_fill):
        """Check that NF placement respects minimum gap from existing NF blocks."""
        for w in weeks_to_fill:
            for w2 in range(max(0, w - NF_MIN_GAP + 1), min(TW, w + NF_MIN_GAP)):
                if w2 not in weeks_to_fill and schedule[rid][w2] == 'NF':
                    return False
        return True

    def nf_not_adjacent_ip(rid, weeks_to_fill):
        """Check that NF weeks are not directly adjacent to other IP rotations."""
        for w in weeks_to_fill:
            if w > 0:
                nb = schedule[rid][w - 1]
                if nb in IP_ROTS and nb != 'NF' and (w - 1) not in weeks_to_fill:
                    return False
            if w < TW - 1:
                nb = schedule[rid][w + 1]
                if nb in IP_ROTS and nb != 'NF' and (w + 1) not in weeks_to_fill:
                    return False
        return True

    def nf_adjacency_ok(rid, w, rot):
        """Check that placing rot at w doesn't create NF-IP adjacency.
           If rot is NF, neighbors can't be other IP.
           If rot is non-NF IP, neighbors can't be NF."""
        s = schedule[rid]
        if rot == 'NF':
            if w > 0 and s[w - 1] in IP_ROTS and s[w - 1] != 'NF':
                return False
            if w < TW - 1 and s[w + 1] in IP_ROTS and s[w + 1] != 'NF':
                return False
        elif rot in IP_ROTS:
            if w > 0 and s[w - 1] == 'NF':
                return False
            if w < TW - 1 and s[w + 1] == 'NF':
                return False
        return True

    def jeo_buffer_ok(rid, w):
        """Check that jeopardy at week w is not directly adjacent to IP."""
        s = schedule[rid]
        if w > 0 and s[w - 1] in IP_ROTS:
            return False
        if w < TW - 1 and s[w + 1] in IP_ROTS:
            return False
        return True

    def micu_cl_sandwich(rid, weeks_to_fill, rot):
        """Soft check: avoid MICU weeks sandwiching a Clinic week."""
        if rot != 'MICU':
            return True  # only applies to MICU
        s = schedule[rid]
        for w in weeks_to_fill:
            # Check if this MICU week is adjacent to Clinic that's adjacent to another MICU
            if w > 1 and s[w - 1] == 'Clinic' and (s[w - 2] == 'MICU' or (w - 2) in weeks_to_fill):
                return False
            if w < TW - 2 and s[w + 1] == 'Clinic' and (s[w + 2] == 'MICU' or (w + 2) in weeks_to_fill):
                return False
        return True

    def check_max(rid, rot, extra):
        """Check if adding extra weeks would exceed per-resident max."""
        return res_weeks[rid][rot] + extra <= MAX_PER_RES.get(rot, 999)

    def balance_score(rid, rot, extra_weeks=1):
        current = res_weeks[rid][rot]
        ideal = IDEAL.get(rot, 1.0)
        rot_excess = (current + extra_weeks) / max(ideal, 0.5)
        total_ip = sum(res_weeks[rid][r] for r in TARGETS)
        ideal_mean_ip = sum(TARGETS.values()) * TW / max(N, 1)
        base = rot_excess + 0.3 * total_ip / max(ideal_mean_ip, 1)
        # Lower-priority rotations score higher = less preferred for early filling
        if rot in LOW_PRIO_ROTS:
            base += 5.0
        return base

    # Build block types dict from params
    block_types = params.get('block_types', {})
    BT = {}
    for rot in ['SLUH', 'VA', 'ID']:
        BT[rot] = block_types.get(rot, 'MarioKart (3wk)')
    BT['NF'] = block_types.get('NF', '2-week')
    for rot in ['MICU', 'Bronze', 'Cards']:
        BT[rot] = block_types.get(rot, 'ABABA (3×1wk)')
    for rot in ['Diamond', 'Gold', 'IP Other 1', 'IP Other 2']:
        BT[rot] = block_types.get(rot, '1-week')

    # Identify rotations by block type
    mario_kart_rots = [r for r in TARGETS if BT.get(r) == 'MarioKart (3wk)']
    two_week_rots = [r for r in TARGETS if BT.get(r) == '2-week']
    ababa_rots = [r for r in TARGETS if BT.get(r) == 'ABABA (3×1wk)']
    single_rots = [r for r in TARGETS if BT.get(r) == '1-week']

    def has_op_sandwich(rid, w):
        """Check that placing an IP single at week w leaves non-IP neighbors."""
        s = schedule[rid]
        if w > 0 and s[w - 1] != '' and s[w - 1] not in ('OP', 'Clinic', 'Jeopardy'):
            return False
        if w < TW - 1 and s[w + 1] != '' and s[w + 1] not in ('OP', 'Clinic', 'Jeopardy'):
            return False
        return True

    # Pass 1: MarioKart (3wk) rotations with stagger constraint
    for ward_rot in mario_kart_rots:
        target = TARGETS[ward_rot]
        for w in range(TW - WARD_LEN + 1):
            while coverage[ward_rot][w] < target:
                cands = [r for r in residents
                         if all(is_free(r['id'], w + i) for i in range(WARD_LEN))
                         and ip_window_ok(r['id'], list(range(w, w + WARD_LEN)))
                         and check_max(r['id'], ward_rot, WARD_LEN)
                         and start_count[ward_rot][w] < MAX_STAGGER]
                if not cands:
                    break
                cands.sort(key=lambda r: (balance_score(r['id'], ward_rot, WARD_LEN), random.random()))
                rid = cands[0]['id']
                for i in range(WARD_LEN):
                    assign(rid, w + i, ward_rot)
                res_weeks[rid][ward_rot] += WARD_LEN
                start_count[ward_rot][w] += 1

    # Pass 2: 2-week rotations with stagger constraint
    for two_week_rot in two_week_rots:
        target = TARGETS[two_week_rot]
        for w in range(TW - NF_LEN + 1):
            while coverage[two_week_rot][w] < target:
                wks = list(range(w, w + NF_LEN))
                cands = [r for r in residents
                         if all(is_free(r['id'], w + i) for i in range(NF_LEN))
                         and ip_window_ok(r['id'], wks)
                         and check_max(r['id'], two_week_rot, NF_LEN)
                         and start_count[two_week_rot][w] < MAX_STAGGER
                         and (two_week_rot != 'NF' or (nf_gap_ok(r['id'], wks) and nf_not_adjacent_ip(r['id'], wks)))]
                if not cands:
                    break
                cands.sort(key=lambda r: (balance_score(r['id'], two_week_rot, NF_LEN), random.random()))
                rid = cands[0]['id']
                for i in range(NF_LEN):
                    assign(rid, w + i, two_week_rot)
                res_weeks[rid][two_week_rot] += NF_LEN
                start_count[two_week_rot][w] += 1

    # Pass 3a: ABABA mini-blocks — prefer starting after clinic week
    ababa_block_rots = sorted(ababa_rots, key=lambda r: TARGETS[r])
    block_used = {r['id']: set() for r in residents}

    # Build per-resident clinic week list for ABABA anchoring
    clinic_weeks = {}
    for r in residents:
        clinic_weeks[r['id']] = [w for w in range(TW) if schedule[r['id']][w] == 'Clinic']

    for rot in ababa_block_rots:
        target = TARGETS[rot]
        # First pass: try post-clinic anchoring
        for r in residents:
            if rot in block_used[r['id']]:
                continue
            if not check_max(r['id'], rot, 3):
                continue
            placed = False
            # Try each clinic week as anchor
            for cw in clinic_weeks[r['id']]:
                w0 = cw + 1  # Start ABABA right after clinic
                weeks3 = [w0, w0 + 2, w0 + 4]
                if any(ww >= TW for ww in weeks3):
                    continue
                if not all(coverage[rot][ww] < target for ww in weeks3):
                    continue
                if not all(is_free(r['id'], ww) for ww in weeks3):
                    continue
                if not all(has_op_sandwich(r['id'], ww) for ww in weeks3):
                    continue
                if not ip_window_ok(r['id'], weeks3):
                    continue
                if not micu_cl_sandwich(r['id'], weeks3, rot):
                    continue
                # Place it
                for ww in weeks3:
                    assign(r['id'], ww, rot)
                    res_weeks[r['id']][rot] += 1
                block_used[r['id']].add(rot)
                placed = True
                break
            # Fallback: any valid position
            if not placed:
                for w0 in range(TW - 4):
                    weeks3 = [w0, w0 + 2, w0 + 4]
                    if not all(coverage[rot][ww] < target for ww in weeks3):
                        continue
                    if not all(is_free(r['id'], ww) for ww in weeks3):
                        continue
                    if not all(has_op_sandwich(r['id'], ww) for ww in weeks3):
                        continue
                    if not ip_window_ok(r['id'], weeks3):
                        continue
                    if not micu_cl_sandwich(r['id'], weeks3, rot):
                        continue
                    for ww in weeks3:
                        assign(r['id'], ww, rot)
                        res_weeks[r['id']][rot] += 1
                    block_used[r['id']].add(rot)
                    break

        # Second pass: coverage-driven (fill any remaining gaps)
        for w0 in range(TW - 4):
            weeks3 = [w0, w0 + 2, w0 + 4]
            if not all(coverage[rot][ww] < target for ww in weeks3):
                continue
            done = True
            while done:
                done = False
                if not all(coverage[rot][ww] < target for ww in weeks3):
                    break
                cands = [r for r in residents
                         if rot not in block_used[r['id']]
                         and all(is_free(r['id'], ww) for ww in weeks3)
                         and all(has_op_sandwich(r['id'], ww) for ww in weeks3)
                         and ip_window_ok(r['id'], weeks3)
                         and micu_cl_sandwich(r['id'], weeks3, rot)
                         and check_max(r['id'], rot, 3)]
                if not cands:
                    break
                cands.sort(key=lambda r: (balance_score(r['id'], rot, 3), random.random()))
                rid = cands[0]['id']
                for ww in weeks3:
                    assign(rid, ww, rot)
                    res_weeks[rid][rot] += 1
                block_used[rid].add(rot)
                done = True

    # Pass 3b: Fill remaining single-week gaps (ABABA and 1-week rotations)
    all_single_rots = ababa_rots + single_rots
    for w in range(TW):
        week_rots = sorted(all_single_rots, key=lambda r: TARGETS[r])
        for rot in week_rots:
            target = TARGETS[rot]
            while coverage[rot][w] < target:
                cands = [r for r in residents
                         if is_free(r['id'], w)
                         and has_op_sandwich(r['id'], w)
                         and ip_window_ok(r['id'], [w])
                         and check_max(r['id'], rot, 1)]
                if not cands:
                    break
                cands.sort(key=lambda r: (balance_score(r['id'], rot, 1), random.random()))
                rid = cands[0]['id']
                assign(rid, w, rot)
                res_weeks[rid][rot] += 1

    # Pass 5: OP
    for r in residents:
        for w in range(TW):
            if schedule[r['id']][w] == '':
                assign(r['id'], w, 'OP')

    # Pass 5b: Repair — OP→rot conversion for ALL understaffed rotations
    all_repair_rots = list(TARGETS.keys())
    for rot in all_repair_rots:
        if rot == 'Jeopardy':
            continue
        target = TARGETS[rot]
        for w in range(TW):
            while coverage[rot][w] < target:
                cands = [r for r in residents
                         if schedule[r['id']][w] == 'OP'
                         and ip_window_ok(r['id'], [w])
                         and nf_adjacency_ok(r['id'], w, rot)
                         and (rot != 'NF' or nf_gap_ok(r['id'], [w]))
                         and check_max(r['id'], rot, 1)]
                if not cands:
                    break
                cands.sort(key=lambda r: (
                    0 if res_weeks[r['id']][rot] < MIN_PER_RES.get(rot, 0) else 1,
                    balance_score(r['id'], rot, 1), random.random()
                ))
                rid = cands[0]['id']
                schedule[rid][w] = rot
                coverage['OP'][w] -= 1
                coverage[rot][w] += 1
                res_weeks[rid][rot] += 1

    # Pass 5c: Swap repair for ALL understaffed rotations
    for rot in all_repair_rots:
        if rot == 'Jeopardy':
            continue
        target = TARGETS[rot]
        for w in range(TW):
            if coverage[rot][w] >= target:
                continue
            surplus_wks = [w2 for w2 in range(TW) if coverage[rot][w2] > target]
            fixed = False
            for w2 in surplus_wks:
                if fixed:
                    break
                donors = [r for r in residents if schedule[r['id']][w2] == rot]
                for donor in donors:
                    if fixed:
                        break
                    did = donor['id']
                    if (schedule[did][w] == 'OP' and ip_window_ok(did, [w])
                            and nf_adjacency_ok(did, w, rot)
                            and (rot != 'NF' or nf_gap_ok(did, [w]))):
                        schedule[did][w2] = 'OP'
                        schedule[did][w] = rot
                        coverage[rot][w2] -= 1
                        coverage[rot][w] += 1
                        coverage['OP'][w2] += 1
                        coverage['OP'][w] -= 1
                        fixed = True

    # Pass 5d: Relaxed repair — allow MAX_IP_WIN+1 as last resort for coverage
    for rot in all_repair_rots:
        if rot == 'Jeopardy':
            continue
        target = TARGETS[rot]
        for w in range(TW):
            while coverage[rot][w] < target:
                cands = [r for r in residents
                         if schedule[r['id']][w] == 'OP'
                         and ip_window_ok(r['id'], [w], relaxed=True)
                         and check_max(r['id'], rot, 1)]
                if not cands:
                    break
                cands.sort(key=lambda r: (
                    0 if res_weeks[r['id']][rot] < MIN_PER_RES.get(rot, 0) else 1,
                    balance_score(r['id'], rot, 1), random.random()
                ))
                rid = cands[0]['id']
                schedule[rid][w] = rot
                coverage['OP'][w] -= 1
                coverage[rot][w] += 1
                res_weeks[rid][rot] += 1

    # Pass 5e: Relaxed swap repair
    for rot in all_repair_rots:
        if rot == 'Jeopardy':
            continue
        target = TARGETS[rot]
        for w in range(TW):
            if coverage[rot][w] >= target:
                continue
            surplus_wks = [w2 for w2 in range(TW) if coverage[rot][w2] > target]
            fixed = False
            for w2 in surplus_wks:
                if fixed:
                    break
                donors = [r for r in residents if schedule[r['id']][w2] == rot]
                for donor in donors:
                    if fixed:
                        break
                    did = donor['id']
                    if schedule[did][w] == 'OP' and ip_window_ok(did, [w], relaxed=True):
                        schedule[did][w2] = 'OP'
                        schedule[did][w] = rot
                        coverage[rot][w2] -= 1
                        coverage[rot][w] += 1
                        coverage['OP'][w2] += 1
                        coverage['OP'][w] -= 1
                        fixed = True

    # Pass 6: Jeopardy — prefer slots not adjacent to IP
    jeop_counts = collections.defaultdict(int)
    for w in range(TW):
        # First try: OP slots with buffer from IP
        pool = [(r, jeop_counts[r['id']]) for r in residents
                if schedule[r['id']][w] == 'OP' and jeop_counts[r['id']] < JEOP_CAP
                and jeo_buffer_ok(r['id'], w)]
        if not pool:
            # Fallback: any OP slot (relax jeo buffer)
            pool = [(r, jeop_counts[r['id']]) for r in residents
                    if schedule[r['id']][w] == 'OP' and jeop_counts[r['id']] < JEOP_CAP]
        if pool:
            pool.sort(key=lambda x: (x[1], random.random()))
            chosen = pool[0][0]
            schedule[chosen['id']][w] = 'Jeopardy'
            coverage['OP'][w] -= 1
            coverage['Jeopardy'][w] += 1
            jeop_counts[chosen['id']] += 1

    # Compute stats
    TARGETS['Jeopardy'] = 1
    fully = sum(1 for w in range(TW)
                if all(coverage[rot][w] >= TARGETS[rot] for rot in TARGETS))

    res_data = []
    for r in residents:
        rid = r['id']
        sched = schedule[rid]
        counts = collections.Counter(sched)
        ip = sum(1 for s in sched if s in IP_ROTS)
        # Max consecutive IP
        mx = c = 0
        for s in sched:
            if s in IP_ROTS:
                c += 1
                mx = max(mx, c)
            else:
                c = 0
        # Max IP in any 6-week window
        max_win = 0
        for start in range(TW - IP_WINDOW + 1):
            wc = sum(1 for w2 in range(start, start + IP_WINDOW) if sched[w2] in IP_ROTS)
            max_win = max(max_win, wc)
        # Check min violations
        min_violations = []
        for rot in [rt for rt in TARGETS if rt != 'Jeopardy']:
            if counts.get(rot, 0) < MIN_PER_RES.get(rot, 0):
                min_violations.append(rot)

        res_data.append({
            'id': rid, 'pgy': r['pgy'], 'pos': r['cpos'],
            'schedule': sched, 'counts': dict(counts),
            'ip': ip, 'op': counts.get('OP', 0) + counts.get('Jeopardy', 0),
            'clinic': counts.get('Clinic', 0), 'maxConsec': mx,
            'maxIPWin': max_win,
            'jeopardy': counts.get('Jeopardy', 0),
            'min_violations': min_violations,
        })

    # Sort residents by clinic position for cascade display
    res_data.sort(key=lambda r: (r['pos'], r['id']))

    # Assign role letters
    role_labels = assign_role_letters(res_data, TW)

    return {
        'residents': res_data,
        'coverage': {rot: list(coverage[rot]) for rot in TARGETS},
        'targets': TARGETS,
        'fully_staffed': fully,
        'total_weeks': TW,
        'max_consec': max((r['maxConsec'] for r in res_data), default=0),
        'violations': sum(1 for r in res_data if r['maxIPWin'] > MAX_IP_WIN),
        'ip_window': IP_WINDOW,
        'max_ip_win': MAX_IP_WIN,
        'ideal': IDEAL,
        'role_labels': role_labels,
        'min_per_res': MIN_PER_RES,
        'max_per_res': MAX_PER_RES,
    }


# ═══════════════════════════════════════════════════════════════════════
# INTERN SCHEDULER (v16: min/max, stagger, role letters, ABABA post-clinic)
# ═══════════════════════════════════════════════════════════════════════
def build_intern(params):
    """Build intern PGY-1 schedule (48 weeks) with rotators."""
    random.seed(params['seed'])

    N_CAT = params['n_cat']
    N_PRELIM = params['n_prelim']
    N = N_CAT + N_PRELIM
    TW = 48
    CYCLE = params.get('clinic_freq', 6)
    MAX_CONSEC = params.get('max_consec', 3)
    WARD_LEN = params.get('ward_len', 3)
    NF_LEN = params.get('nf_len', 2)
    JEOP_CAP = params.get('jeop_cap', 3)
    MAX_STAGGER = params.get('max_stagger', 2)

    ALL_IP = {'SLUH', 'VA', 'NF', 'MICU', 'Cards', 'IP Other 1', 'IP Other 2'}
    STAG = ['MICU', 'Cards']
    FULL = {
        'SLUH': params['t_sluh'], 'VA': params['t_va'],
        'NF': params['t_nf'], 'MICU': params['t_micu'], 'Cards': params['t_cards'],
        'IP Other 1': params.get('t_other1', 0), 'IP Other 2': params.get('t_other2', 0),
    }

    # Build block types
    block_types = params.get('block_types', {})
    BT = {}
    for rot in ['SLUH', 'VA']:
        BT[rot] = block_types.get(rot, 'MarioKart (3wk)')
    BT['NF'] = block_types.get('NF', '2-week')
    for rot in STAG:
        BT[rot] = block_types.get(rot, 'ABABA (3×1wk)')
    for rot in ['IP Other 1', 'IP Other 2']:
        BT[rot] = block_types.get(rot, '1-week')

    mario_kart_rots_intern = [r for r in ['SLUH', 'VA'] if BT.get(r) == 'MarioKart (3wk)']
    two_week_rots_intern = [r for r in ['NF'] if BT.get(r) == '2-week']
    ababa_rots_intern = [r for r in STAG if BT.get(r) == 'ABABA (3×1wk)']
    single_rots_intern = [r for r in STAG if BT.get(r) == '1-week']

    def month_weeks(m):
        return list(range(m * 4, min(m * 4 + 4, TW)))

    # ── Rotator schedules ──
    n_neuro = params.get('n_neuro', 6)
    n_anes = params.get('n_anes', 10)
    n_psych = params.get('n_psych', 8)
    n_em = params.get('n_em', 8)
    neuro_months = params.get('neuro_months', 4)

    neuro = [{'id': f'Neuro_{i+1}', 'type': 'Neuro', 'schedule': [''] * TW} for i in range(n_neuro)]
    slots = []
    for m in range(12):
        slots.append((m, 'SLUH'))
        slots.append((m, 'VA'))
    random.shuffle(slots)
    si = 0
    for nr in neuro:
        for _ in range(neuro_months):
            if si < len(slots):
                m, rot = slots[si]
                si += 1
                for w in month_weeks(m):
                    nr['schedule'][w] = rot

    anes = []
    am = list(range(12))
    random.shuffle(am)
    for i in range(n_anes):
        ar = {'id': f'Anes_{i+1}', 'type': 'Anes', 'schedule': [''] * TW}
        for w in month_weeks(am[i % 12]):
            ar['schedule'][w] = 'SLUH'
        anes.append(ar)

    psych = []
    pm = list(range(12))
    random.shuffle(pm)
    for i in range(n_psych):
        pr = {'id': f'Psych_{i+1}', 'type': 'Psych', 'schedule': [''] * TW}
        for w in month_weeks(pm[i % 12]):
            pr['schedule'][w] = 'VA'
        psych.append(pr)

    emr = []
    em_m = list(range(12))
    random.shuffle(em_m)
    for i in range(n_em):
        er = {'id': f'EM_{i+1}', 'type': 'EM', 'schedule': [''] * TW}
        for w in month_weeks(em_m[i % 12]):
            er['schedule'][w] = 'ICU*'
        emr.append(er)

    # Rotator coverage per week
    r_sluh = [0] * TW
    r_va = [0] * TW
    for rot_res in neuro + anes + psych:
        for w in range(TW):
            if rot_res['schedule'][w] == 'SLUH':
                r_sluh[w] += 1
            if rot_res['schedule'][w] == 'VA':
                r_va[w] += 1

    # Dynamic intern targets
    it_s = [max(0, FULL['SLUH'] - r_sluh[w]) for w in range(TW)]
    it_v = [max(0, FULL['VA'] - r_va[w]) for w in range(TW)]

    # Build interns
    residents = []
    for i in range(1, N_CAT + 1):
        residents.append({'id': f'I_cat{i:02d}', 'type': 'cat'})
    for i in range(1, N_PRELIM + 1):
        residents.append({'id': f'I_pre{i:02d}', 'type': 'prelim'})

    positions = {r['id']: (i % 6) + 1 for i, r in enumerate(residents)}

    FULL = {k: v for k, v in FULL.items() if v > 0}

    IDEAL = {
        'SLUH': sum(it_s) / max(N, 1), 'VA': sum(it_v) / max(N, 1),
        'NF': FULL['NF'] * TW / max(N, 1), 'MICU': FULL['MICU'] * TW / max(N, 1),
        'Cards': FULL['Cards'] * TW / max(N, 1),
    }
    for rot in ['IP Other 1', 'IP Other 2']:
        if rot in FULL:
            IDEAL[rot] = FULL[rot] * TW / max(N, 1)

    # Min/max per resident
    MIN_PER_RES = {}
    MAX_PER_RES = {}
    minmax = params.get('min_max', {})
    for rot in FULL:
        ideal_val = IDEAL.get(rot, 0)
        MIN_PER_RES[rot] = minmax.get(rot, {}).get('min', max(0, int(ideal_val * 0.4)))
        MAX_PER_RES[rot] = minmax.get(rot, {}).get('max', int(ideal_val * 1.8) + 1)

    schedule = {r['id']: [''] * TW for r in residents}
    coverage = {rot: [0] * TW for rot in ['SLUH', 'VA', 'NF', 'MICU', 'Cards', 'IP Other 1', 'IP Other 2', 'Jeopardy']}
    rw = {r['id']: collections.Counter() for r in residents}
    start_count = collections.defaultdict(lambda: [0] * TW)

    for r in residents:
        p = positions[r['id']]
        for c in range(TW // CYCLE):
            w = c * CYCLE + (p - 1)
            if w < TW:
                schedule[r['id']][w] = 'Clinic'

    def free(rid, w):
        return 0 <= w < TW and schedule[rid][w] == ''

    def asgn(rid, w, rot):
        schedule[rid][w] = rot
        rw[rid][rot] += 1
        coverage[rot][w] += 1

    # ── Scheduling quality parameters ──
    IP_WINDOW = params.get('ip_window', 6)
    MAX_IP_WIN = params.get('max_ip_win', 3)
    NF_MIN_GAP = params.get('nf_min_gap', 6)
    LOW_PRIO_ROTS = {'IP Other 1', 'IP Other 2'}

    def ip_window_ok(rid, weeks_to_fill, relaxed=False):
        limit = MAX_IP_WIN + 1 if relaxed else MAX_IP_WIN
        temp = list(schedule[rid])
        for w in weeks_to_fill:
            temp[w] = 'IP'
        for start in range(max(0, min(weeks_to_fill) - IP_WINDOW + 1),
                           min(TW - IP_WINDOW + 1, max(weeks_to_fill) + 1)):
            count = sum(1 for w in range(start, start + IP_WINDOW)
                        if temp[w] in ALL_IP or temp[w] == 'IP')
            if count > limit:
                return False
        return True

    def nf_gap_ok(rid, weeks_to_fill):
        for w in weeks_to_fill:
            for w2 in range(max(0, w - NF_MIN_GAP + 1), min(TW, w + NF_MIN_GAP)):
                if w2 not in weeks_to_fill and schedule[rid][w2] == 'NF':
                    return False
        return True

    def nf_not_adjacent_ip(rid, weeks_to_fill):
        for w in weeks_to_fill:
            if w > 0:
                nb = schedule[rid][w - 1]
                if nb in ALL_IP and nb != 'NF' and (w - 1) not in weeks_to_fill:
                    return False
            if w < TW - 1:
                nb = schedule[rid][w + 1]
                if nb in ALL_IP and nb != 'NF' and (w + 1) not in weeks_to_fill:
                    return False
        return True

    def nf_adjacency_ok(rid, w, rot):
        s = schedule[rid]
        if rot == 'NF':
            if w > 0 and s[w - 1] in ALL_IP and s[w - 1] != 'NF':
                return False
            if w < TW - 1 and s[w + 1] in ALL_IP and s[w + 1] != 'NF':
                return False
        elif rot in ALL_IP:
            if w > 0 and s[w - 1] == 'NF':
                return False
            if w < TW - 1 and s[w + 1] == 'NF':
                return False
        return True

    def jeo_buffer_ok(rid, w):
        s = schedule[rid]
        if w > 0 and s[w - 1] in ALL_IP:
            return False
        if w < TW - 1 and s[w + 1] in ALL_IP:
            return False
        return True

    def micu_cl_sandwich(rid, weeks_to_fill, rot):
        if rot != 'MICU':
            return True
        s = schedule[rid]
        for w in weeks_to_fill:
            if w > 1 and s[w - 1] == 'Clinic' and (s[w - 2] == 'MICU' or (w - 2) in weeks_to_fill):
                return False
            if w < TW - 2 and s[w + 1] == 'Clinic' and (s[w + 2] == 'MICU' or (w + 2) in weeks_to_fill):
                return False
        return True

    def check_max_i(rid, rot, extra):
        return rw[rid][rot] + extra <= MAX_PER_RES.get(rot, 999)

    avg_ip = sum(it_s + it_v) + (FULL.get('NF', 0) + FULL.get('MICU', 0) + FULL.get('Cards', 0)) * TW
    avg_ip /= max(N, 1)

    def bs(rid, rot, ew=1):
        cur = rw[rid][rot]
        ideal = IDEAL.get(rot, 1.0)
        base = (cur + ew) / max(ideal, 0.5) + 0.3 * sum(rw[rid][r] for r in ALL_IP) / max(avg_ip, 1)
        if rot in LOW_PRIO_ROTS:
            base += 5.0
        return base

    def has_op_sandwich(rid, w):
        s = schedule[rid]
        if w > 0 and s[w - 1] != '' and s[w - 1] not in ('OP', 'Clinic', 'Jeopardy'):
            return False
        if w < TW - 1 and s[w + 1] != '' and s[w + 1] not in ('OP', 'Clinic', 'Jeopardy'):
            return False
        return True

    # Build per-resident clinic week list for ABABA anchoring
    clinic_weeks_i = {}
    for r in residents:
        clinic_weeks_i[r['id']] = [w for w in range(TW) if schedule[r['id']][w] == 'Clinic']

    # Pass 1: MarioKart (3wk) with stagger
    if 'SLUH' in mario_kart_rots_intern:
        for w in range(TW - WARD_LEN + 1):
            while coverage['SLUH'][w] < it_s[w]:
                cs = [r for r in residents
                      if all(free(r['id'], w + i) for i in range(WARD_LEN))
                      and ip_window_ok(r['id'], list(range(w, w + WARD_LEN)))
                      and check_max_i(r['id'], 'SLUH', WARD_LEN)
                      and start_count['SLUH'][w] < MAX_STAGGER]
                if not cs:
                    break
                cs.sort(key=lambda r: (bs(r['id'], 'SLUH', WARD_LEN), random.random()))
                for i in range(WARD_LEN):
                    asgn(cs[0]['id'], w + i, 'SLUH')
                start_count['SLUH'][w] += 1

    if 'VA' in mario_kart_rots_intern:
        for w in range(TW - WARD_LEN + 1):
            while coverage['VA'][w] < it_v[w]:
                cs = [r for r in residents
                      if all(free(r['id'], w + i) for i in range(WARD_LEN))
                      and ip_window_ok(r['id'], list(range(w, w + WARD_LEN)))
                      and check_max_i(r['id'], 'VA', WARD_LEN)
                      and start_count['VA'][w] < MAX_STAGGER]
                if not cs:
                    break
                cs.sort(key=lambda r: (bs(r['id'], 'VA', WARD_LEN), random.random()))
                for i in range(WARD_LEN):
                    asgn(cs[0]['id'], w + i, 'VA')
                start_count['VA'][w] += 1

    # Pass 2: 2-week (NF) with stagger + NF gap/adjacency checks
    for w in range(TW - NF_LEN + 1):
        while coverage['NF'][w] < FULL.get('NF', 0):
            wks = list(range(w, w + NF_LEN))
            cs = [r for r in residents
                  if all(free(r['id'], w + i) for i in range(NF_LEN))
                  and ip_window_ok(r['id'], wks)
                  and check_max_i(r['id'], 'NF', NF_LEN)
                  and start_count['NF'][w] < MAX_STAGGER
                  and nf_gap_ok(r['id'], wks)
                  and nf_not_adjacent_ip(r['id'], wks)]
            if not cs:
                break
            cs.sort(key=lambda r: (bs(r['id'], 'NF', NF_LEN), random.random()))
            for i in range(NF_LEN):
                asgn(cs[0]['id'], w + i, 'NF')
            start_count['NF'][w] += 1

    # Pass 3a: ABABA mini-blocks — prefer post-clinic anchoring + MICU sandwich avoidance
    ababa_block_rots_intern = sorted(ababa_rots_intern, key=lambda r: FULL[r])
    i_block_used = {r['id']: set() for r in residents}

    for sr in ababa_block_rots_intern:
        tgt = FULL[sr]
        # Per-resident post-clinic anchoring
        for r in residents:
            if sr in i_block_used[r['id']]:
                continue
            if not check_max_i(r['id'], sr, 3):
                continue
            placed = False
            for cw in clinic_weeks_i[r['id']]:
                w0 = cw + 1
                weeks3 = [w0, w0 + 2, w0 + 4]
                if any(ww >= TW for ww in weeks3):
                    continue
                if not all(coverage[sr][ww] < tgt for ww in weeks3):
                    continue
                if not all(free(r['id'], ww) for ww in weeks3):
                    continue
                if not all(has_op_sandwich(r['id'], ww) for ww in weeks3):
                    continue
                if not ip_window_ok(r['id'], weeks3):
                    continue
                if not micu_cl_sandwich(r['id'], weeks3, sr):
                    continue
                for ww in weeks3:
                    asgn(r['id'], ww, sr)
                i_block_used[r['id']].add(sr)
                placed = True
                break
            if not placed:
                for w0 in range(TW - 4):
                    weeks3 = [w0, w0 + 2, w0 + 4]
                    if not all(coverage[sr][ww] < tgt for ww in weeks3):
                        continue
                    if not all(free(r['id'], ww) for ww in weeks3):
                        continue
                    if not all(has_op_sandwich(r['id'], ww) for ww in weeks3):
                        continue
                    if not ip_window_ok(r['id'], weeks3):
                        continue
                    if not micu_cl_sandwich(r['id'], weeks3, sr):
                        continue
                    for ww in weeks3:
                        asgn(r['id'], ww, sr)
                    i_block_used[r['id']].add(sr)
                    break

        # Coverage-driven fill
        for w0 in range(TW - 4):
            weeks3 = [w0, w0 + 2, w0 + 4]
            if not all(coverage[sr][ww] < tgt for ww in weeks3):
                continue
            placing = True
            while placing:
                placing = False
                if not all(coverage[sr][ww] < tgt for ww in weeks3):
                    break
                cs = [r for r in residents
                      if sr not in i_block_used[r['id']]
                      and all(free(r['id'], ww) for ww in weeks3)
                      and all(has_op_sandwich(r['id'], ww) for ww in weeks3)
                      and ip_window_ok(r['id'], weeks3)
                      and micu_cl_sandwich(r['id'], weeks3, sr)
                      and check_max_i(r['id'], sr, 3)]
                if not cs:
                    break
                cs.sort(key=lambda r: (bs(r['id'], sr, 3), random.random()))
                rid = cs[0]['id']
                for ww in weeks3:
                    asgn(rid, ww, sr)
                i_block_used[rid].add(sr)
                placing = True

    # Pass 3b: single-week gap fill
    all_single_rots_intern = ababa_rots_intern + single_rots_intern
    for w in range(TW):
        for sr in sorted(all_single_rots_intern, key=lambda r: FULL[r]):
            tgt = FULL[sr]
            while coverage[sr][w] < tgt:
                cs = [r for r in residents
                      if free(r['id'], w)
                      and has_op_sandwich(r['id'], w)
                      and ip_window_ok(r['id'], [w])
                      and check_max_i(r['id'], sr, 1)]
                if not cs:
                    break
                cs.sort(key=lambda r: (bs(r['id'], sr, 1), random.random()))
                asgn(cs[0]['id'], w, sr)

    # Pass 4: OP fill
    for r in residents:
        for w in range(TW):
            if schedule[r['id']][w] == '':
                schedule[r['id']][w] = 'OP'

    # Pass 4b: Repair — OP→rot conversion for ALL understaffed rotations
    # Build dynamic target lookup for SLUH/VA (intern targets are per-week)
    intern_targets_by_week = {}
    for rot in FULL:
        if rot == 'SLUH':
            intern_targets_by_week[rot] = it_s
        elif rot == 'VA':
            intern_targets_by_week[rot] = it_v
        else:
            intern_targets_by_week[rot] = [FULL[rot]] * TW

    for rot in FULL:
        if rot == 'Jeopardy':
            continue
        for w in range(TW):
            tgt_w = intern_targets_by_week[rot][w]
            while coverage[rot][w] < tgt_w:
                cs = [r for r in residents
                      if schedule[r['id']][w] == 'OP'
                      and ip_window_ok(r['id'], [w])
                      and nf_adjacency_ok(r['id'], w, rot)
                      and (rot != 'NF' or nf_gap_ok(r['id'], [w]))
                      and check_max_i(r['id'], rot, 1)]
                if not cs:
                    break
                cs.sort(key=lambda r: (
                    0 if rw[r['id']][rot] < MIN_PER_RES.get(rot, 0) else 1,
                    bs(r['id'], rot, 1), random.random()
                ))
                rid = cs[0]['id']
                schedule[rid][w] = rot
                rw[rid][rot] += 1
                coverage[rot][w] += 1

    # Pass 4c: Swap repair for remaining gaps
    for rot in FULL:
        if rot == 'Jeopardy':
            continue
        for w in range(TW):
            tgt_w = intern_targets_by_week[rot][w]
            if coverage[rot][w] >= tgt_w:
                continue
            surplus_wks = [w2 for w2 in range(TW) if coverage[rot][w2] > intern_targets_by_week[rot][w2]]
            fixed = False
            for w2 in surplus_wks:
                if fixed:
                    break
                donors = [r for r in residents if schedule[r['id']][w2] == rot]
                for donor in donors:
                    if fixed:
                        break
                    did = donor['id']
                    if (schedule[did][w] == 'OP' and ip_window_ok(did, [w])
                            and nf_adjacency_ok(did, w, rot)
                            and (rot != 'NF' or nf_gap_ok(did, [w]))):
                        schedule[did][w2] = 'OP'
                        schedule[did][w] = rot
                        coverage[rot][w2] -= 1
                        coverage[rot][w] += 1
                        fixed = True

    # Pass 4d: Relaxed repair — allow MAX_IP_WIN+1 as last resort
    for rot in FULL:
        if rot == 'Jeopardy':
            continue
        for w in range(TW):
            tgt_w = intern_targets_by_week[rot][w]
            while coverage[rot][w] < tgt_w:
                cs = [r for r in residents
                      if schedule[r['id']][w] == 'OP'
                      and ip_window_ok(r['id'], [w], relaxed=True)
                      and check_max_i(r['id'], rot, 1)]
                if not cs:
                    break
                cs.sort(key=lambda r: (
                    0 if rw[r['id']][rot] < MIN_PER_RES.get(rot, 0) else 1,
                    bs(r['id'], rot, 1), random.random()
                ))
                rid = cs[0]['id']
                schedule[rid][w] = rot
                rw[rid][rot] += 1
                coverage[rot][w] += 1

    # Pass 4e: Relaxed swap repair
    for rot in FULL:
        if rot == 'Jeopardy':
            continue
        for w in range(TW):
            tgt_w = intern_targets_by_week[rot][w]
            if coverage[rot][w] >= tgt_w:
                continue
            surplus_wks = [w2 for w2 in range(TW) if coverage[rot][w2] > intern_targets_by_week[rot][w2]]
            fixed = False
            for w2 in surplus_wks:
                if fixed:
                    break
                donors = [r for r in residents if schedule[r['id']][w2] == rot]
                for donor in donors:
                    if fixed:
                        break
                    did = donor['id']
                    if schedule[did][w] == 'OP' and ip_window_ok(did, [w], relaxed=True):
                        schedule[did][w2] = 'OP'
                        schedule[did][w] = rot
                        coverage[rot][w2] -= 1
                        coverage[rot][w] += 1
                        fixed = True

    # Pass 5: Jeopardy — prefer slots not adjacent to IP
    jc = {r['id']: 0 for r in residents}
    for w in range(TW):
        cs = [r for r in residents if schedule[r['id']][w] == 'OP' and jc[r['id']] < JEOP_CAP
              and jeo_buffer_ok(r['id'], w)]
        if not cs:
            cs = [r for r in residents if schedule[r['id']][w] == 'OP' and jc[r['id']] < JEOP_CAP]
        if cs:
            cs.sort(key=lambda r: (jc[r['id']], random.random()))
            rid = cs[0]['id']
            schedule[rid][w] = 'Jeopardy'
            coverage['Jeopardy'][w] += 1
            jc[rid] += 1
            rw[rid]['Jeopardy'] += 1

    # Stats
    FULL['Jeopardy'] = 1
    fully = sum(1 for w in range(TW) if
                coverage['SLUH'][w] + r_sluh[w] >= FULL.get('SLUH', 0) and
                coverage['VA'][w] + r_va[w] >= FULL.get('VA', 0) and
                coverage['NF'][w] >= FULL.get('NF', 0) and
                coverage['MICU'][w] >= FULL.get('MICU', 0) and
                coverage['Cards'][w] >= FULL.get('Cards', 0) and
                coverage['Jeopardy'][w] >= 1)

    res_data = []
    for r in residents:
        rid = r['id']
        sched = schedule[rid]
        counts = collections.Counter(sched)
        ip = sum(1 for s in sched if s in ALL_IP)
        mx = c = 0
        for s in sched:
            if s in ALL_IP:
                c += 1
                mx = max(mx, c)
            else:
                c = 0
        # Max IP weeks in any sliding window
        max_win = 0
        for start in range(TW - IP_WINDOW + 1):
            wc = sum(1 for w2 in range(start, start + IP_WINDOW) if sched[w2] in ALL_IP)
            max_win = max(max_win, wc)
        min_violations = []
        for rot in [rt for rt in FULL if rt != 'Jeopardy']:
            if counts.get(rot, 0) < MIN_PER_RES.get(rot, 0):
                min_violations.append(rot)

        res_data.append({
            'id': rid, 'pgy': 1, 'pos': positions[rid], 'type': r['type'],
            'schedule': sched, 'counts': dict(counts),
            'ip': ip, 'op': counts.get('OP', 0) + counts.get('Jeopardy', 0),
            'clinic': counts.get('Clinic', 0), 'maxConsec': mx,
            'maxIPWin': max_win,
            'jeopardy': counts.get('Jeopardy', 0),
            'min_violations': min_violations,
        })

    # Sort by clinic position for cascade display
    res_data.sort(key=lambda r: (r['pos'], r['id']))

    role_labels = assign_role_letters(res_data, TW)

    return {
        'residents': res_data,
        'coverage': {rot: list(coverage[rot]) for rot in coverage},
        'targets': FULL,
        'rotators': neuro + anes + psych + emr,
        'rotator_coverage': {'SLUH': r_sluh, 'VA': r_va},
        'fully_staffed': fully,
        'total_weeks': TW,
        'max_consec': max((r['maxConsec'] for r in res_data), default=0),
        'violations': sum(1 for r in res_data if r['maxIPWin'] > MAX_IP_WIN),
        'ip_window': IP_WINDOW,
        'max_ip_win': MAX_IP_WIN,
        'ideal': IDEAL,
        'role_labels': role_labels,
        'min_per_res': MIN_PER_RES,
        'max_per_res': MAX_PER_RES,
    }


# ═══════════════════════════════════════════════════════════════════════
# SEED SEARCH
# ═══════════════════════════════════════════════════════════════════════
def find_best_seed(params, level, max_seed=99):
    best_seed = 0
    best_staffed = 0
    tw = 96 if level == 'Senior' else 48
    for s in range(max_seed + 1):
        p = dict(params)
        p['seed'] = s
        result = build_senior(p) if level == 'Senior' else build_intern(p)
        if result['fully_staffed'] > best_staffed:
            best_staffed = result['fully_staffed']
            best_seed = s
            if best_staffed == tw:
                break
    return best_seed, best_staffed


# ═══════════════════════════════════════════════════════════════════════
# SCHEDULE GRID HTML
# ═══════════════════════════════════════════════════════════════════════
def render_schedule_html(data, level, params, highlight_jeo=False, sort_mode='clinic'):
    """Render schedule as HTML table with colors, role letters, year headers."""
    TW = data.get('total_weeks', 48)
    residents = list(data['residents'])
    if sort_mode == 'pgy_id':
        residents.sort(key=lambda r: (r.get('pgy', 1), r['id']))
    coverage = data['coverage']
    targets = data['targets']
    role_labels = data.get('role_labels', {})

    rot_list = list(targets.keys())
    if 'Jeopardy' in rot_list:
        rot_list.remove('Jeopardy')

    html = '<div class="schedule-grid"><table>'

    # Year header row (only for 96-week senior)
    if TW > 48:
        html += '<tr><th></th>'
        html += f'<th colspan="48" class="year-hdr">Year 1 (Weeks 1–48)</th>'
        html += f'<th colspan="{TW - 48}" class="year-hdr">Year 2 (Weeks 49–{TW})</th>'
        html += f'<th colspan="{5 + len(rot_list)}"></th></tr>'

    # Header
    html += '<tr><th style="min-width:65px;">Resident</th>'
    for w in range(1, TW + 1):
        html += f'<th>W{w}</th>'
    html += '<th>IP</th><th>OP</th><th>CL</th><th>Jeo</th><th>IPW</th>'
    for rot in rot_list:
        html += f'<th>{ABBREV.get(rot, rot)}</th>'
    html += '</tr>'

    # Resident rows
    for r in residents:
        html += f'<tr><td style="text-align:left;font-weight:500;">{r["id"]}</td>'
        for w in range(TW):
            val = r['schedule'][w]
            bg = COLORS.get(val, '#fff')
            fc = '#fff' if val in DARK_BG else '#333'

            # Role letter
            letter = role_labels.get((r['id'], w), '')
            base_abbrev = ABBREV.get(val, val or '')
            if letter and val in ROLE_LETTER_ROTS:
                # Compact: MC-A, BZ-B, etc.
                short = {'MICU': 'MC', 'Bronze': 'BZ', 'Cards': 'CR', 'NF': 'NF'}
                txt = f'{short.get(val, base_abbrev[:2])}-{letter}'
            else:
                txt = base_abbrev

            # Jeopardy highlight: dim non-jeopardy cells
            dim = ''
            if highlight_jeo and val != 'Jeopardy':
                dim = 'opacity:0.15;'

            html += f'<td style="background:{bg};color:{fc};font-size:8px;{dim}">{txt}</td>'

        html += f'<td>{r["ip"]}</td><td>{r["op"]}</td><td>{r["clinic"]}</td>'
        html += f'<td>{r.get("jeopardy", 0)}</td>'
        mxw = r.get('maxIPWin', r.get('maxConsec', 0))
        ip_lim = params.get('max_ip_win', 3)
        html += f'<td style="{"background:#FFC7CE;" if mxw > ip_lim else ""}">{mxw}</td>'
        for rot in rot_list:
            ct = r['counts'].get(rot, 0)
            # Flag if below min
            mn = data.get('min_per_res', {}).get(rot, 0)
            mx_r = data.get('max_per_res', {}).get(rot, 999)
            style = ''
            if ct < mn:
                style = 'background:#FFC7CE;'
            elif ct > mx_r:
                style = 'background:#FFC7CE;'
            html += f'<td style="{style}">{ct}</td>'
        html += '</tr>'

    # Rotator rows (intern only)
    if level == 'Intern' and 'rotators' in data:
        type_labels = {
            'Neuro': 'NEUROLOGY', 'Anes': 'ANESTHESIA',
            'Psych': 'PSYCHIATRY', 'EM': 'EMERGENCY MED',
        }
        current_type = None
        for rot_res in data['rotators']:
            if rot_res['type'] != current_type:
                current_type = rot_res['type']
                label = type_labels.get(current_type, current_type)
                html += f'<tr><td colspan="{TW + 6 + len(rot_list)}" class="section-hdr">{label} ROTATORS</td></tr>'
            html += f'<tr><td style="text-align:left;font-style:italic;">{rot_res["id"]}</td>'
            for w in range(TW):
                val = rot_res['schedule'][w]
                if val:
                    bg = COLORS.get(val, '#fff')
                    fc = '#fff' if val in DARK_BG else '#333'
                    html += f'<td style="background:{bg};color:{fc};font-size:8px;">{ABBREV.get(val, val)}</td>'
                else:
                    html += '<td style="background:#f0f0f0;"></td>'
            html += f'<td colspan="{5 + len(rot_list)}"></td></tr>'

    # Coverage summary
    html += f'<tr><td colspan="{TW + 6 + len(rot_list)}" class="summary-hdr">WEEKLY COVERAGE SUMMARY</td></tr>'
    cov_rots = list(targets.keys())
    for rot in cov_rots:
        tgt = targets[rot]
        html += f'<tr><td style="text-align:left;font-weight:600;">{rot}</td>'
        for w in range(TW):
            count = coverage.get(rot, [0] * TW)[w]
            if level == 'Intern' and rot in ('SLUH', 'VA') and 'rotator_coverage' in data:
                count += data['rotator_coverage'].get(rot, [0] * TW)[w]
            ok = count >= tgt
            bg = '#B3FFB3' if ok else '#FF9999'
            html += f'<td style="background:{bg};font-weight:600;font-size:9px;">{count}</td>'
        html += f'<td colspan="{5 + len(rot_list)}"></td></tr>'

    # OP and Clinic weekly counts
    for extra_rot in ['OP', 'Clinic']:
        counts = [0] * TW
        for res in data['residents']:
            for w in range(TW):
                if res['schedule'][w] == extra_rot:
                    counts[w] += 1
        html += f'<tr><td style="text-align:left;font-weight:600;">{extra_rot}</td>'
        for w in range(TW):
            html += f'<td style="font-size:9px;">{counts[w]}</td>'
        html += f'<td colspan="{5 + len(rot_list)}"></td></tr>'

    html += '</table></div>'
    return html


# ═══════════════════════════════════════════════════════════════════════
# EXCEL EXPORT
# ═══════════════════════════════════════════════════════════════════════
def export_to_excel(data, level, params):
    """Export schedule data to an Excel file in memory."""
    output = io.BytesIO()
    TW = data.get('total_weeks', 48)
    targets = data['targets']
    role_labels = data.get('role_labels', {})

    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Sheet 1: Schedule Grid (with role letters)
        rows = []
        for r in data['residents']:
            row = {'Resident': r['id'], 'PGY': r.get('pgy', ''), 'Position': r.get('pos', '')}
            for w in range(TW):
                val = r['schedule'][w]
                letter = role_labels.get((r['id'], w), '')
                if letter and val in ROLE_LETTER_ROTS:
                    row[f'Week {w+1}'] = f'{val} {letter}'
                else:
                    row[f'Week {w+1}'] = val
            row['IP Weeks'] = r['ip']
            row['OP Weeks'] = r['op']
            row['Clinic'] = r['clinic']
            row['Jeopardy'] = r.get('jeopardy', 0)
            row['Max Consec'] = r['maxConsec']
            rows.append(row)
        pd.DataFrame(rows).to_excel(writer, sheet_name='Schedule', index=False)

        # Sheet 2: Coverage
        cov_rows = []
        for rot in targets:
            row = {'Rotation': rot, 'Target': targets[rot]}
            vals = data['coverage'].get(rot, [0] * TW)
            for w in range(TW):
                row[f'Week {w+1}'] = vals[w]
            met = sum(1 for v in vals if v >= targets[rot])
            row['Weeks Met'] = f'{met}/{TW}'
            cov_rows.append(row)
        pd.DataFrame(cov_rows).to_excel(writer, sheet_name='Coverage', index=False)

        # Sheet 3: Balance
        bal_rows = []
        for r in data['residents']:
            row = {'Resident': r['id'], 'IP': r['ip'], 'OP': r['op'], 'Clinic': r['clinic'],
                   'Jeopardy': r.get('jeopardy', 0), 'Max Consec': r['maxConsec']}
            for rot in sorted(targets.keys()):
                row[rot] = r['counts'].get(rot, 0)
            bal_rows.append(row)
        pd.DataFrame(bal_rows).to_excel(writer, sheet_name='Balance', index=False)

        # Sheet 4: Parameters
        param_rows = [{'Parameter': k, 'Value': str(v)} for k, v in sorted(params.items())]
        pd.DataFrame(param_rows).to_excel(writer, sheet_name='Parameters', index=False)

        for sheet_name in writer.sheets:
            ws = writer.sheets[sheet_name]
            ws.column_dimensions['A'].width = 18
            ws.column_dimensions['B'].width = 12

    output.seek(0)
    return output


# ═══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════
st.sidebar.markdown("## Project Z — Schedule Explorer")
st.sidebar.markdown("*ABC Model • Interactive Design Tool*")
st.sidebar.divider()
st.sidebar.caption("Adjust parameters below to regenerate the schedule in real-time. "
                   "Senior schedules cover 96 weeks (2 years); Intern schedules cover 48 weeks (1 year).")

level = st.sidebar.radio("Schedule Level", ["Senior", "Intern"], horizontal=True)

st.sidebar.markdown("### Roster")
if level == "Senior":
    n_pgy3 = st.sidebar.number_input("PGY-3 Count", 1, 50, 27)
    n_pgy2 = st.sidebar.number_input("PGY-2 Count", 1, 50, 28)
else:
    n_cat = st.sidebar.number_input("Categorical Interns", 1, 50, 26)
    n_prelim = st.sidebar.number_input("Preliminary Interns", 0, 20, 5)

if level == "Intern":
    st.sidebar.markdown("### Rotators")
    col_a, col_b = st.sidebar.columns(2)
    n_neuro = col_a.number_input("Neuro", 0, 20, 6)
    neuro_mo = col_b.number_input("Mo/res", 1, 6, 4, key="neuro_mo")
    col_c, col_d = st.sidebar.columns(2)
    n_anes = col_c.number_input("Anes", 0, 20, 10)
    n_psych = col_d.number_input("Psych", 0, 20, 8)
    n_em = st.sidebar.number_input("EM", 0, 20, 8)

st.sidebar.markdown("### Weekly Targets")
st.sidebar.caption("Number of residents needed on each rotation per week.")
if level == "Senior":
    col1, col2, col3 = st.sidebar.columns(3)
    t_sluh = col1.number_input("SLUH", 0, 15, 6)
    t_va = col2.number_input("VA", 0, 15, 5)
    t_id = col3.number_input("ID", 0, 10, 1)
    col4, col5, col6 = st.sidebar.columns(3)
    t_nf = col4.number_input("NF", 0, 15, 5)
    t_micu = col5.number_input("MICU", 0, 10, 4)
    t_bronze = col6.number_input("Bronze", 0, 10, 2)
    col7, col8, col9 = st.sidebar.columns(3)
    t_cards = col7.number_input("Cards", 0, 10, 2)
    t_diamond = col8.number_input("Diamond", 0, 5, 0)  # Default 0 (disabled)
    t_gold = col9.number_input("Gold", 0, 5, 1)
    col10, col11 = st.sidebar.columns(2)
    t_other1 = col10.number_input("IP Other 1", 0, 10, 0)
    t_other2 = col11.number_input("IP Other 2", 0, 10, 0)
else:
    col1, col2 = st.sidebar.columns(2)
    t_sluh = col1.number_input("SLUH (total)", 0, 15, 4)
    t_va = col2.number_input("VA (total)", 0, 15, 5)
    col3, col4, col5 = st.sidebar.columns(3)
    t_nf = col3.number_input("NF", 0, 15, 4)
    t_micu = col4.number_input("MICU", 0, 10, 2)
    t_cards = col5.number_input("Cards", 0, 10, 1)
    col6, col7 = st.sidebar.columns(2)
    t_other1 = col6.number_input("IP Other 1", 0, 10, 0, key="i_other1")
    t_other2 = col7.number_input("IP Other 2", 0, 10, 0, key="i_other2")

# ── Min/Max Per Resident ──
st.sidebar.markdown("### Min/Max Weeks Per Resident")
st.sidebar.caption("Hard constraints: each resident must have between min and max weeks of each rotation. "
                   "Defaults are calculated from ideal distribution.")

min_max = {}
if level == "Senior":
    TW_display = 96
    N_display = n_pgy3 + n_pgy2
    active_rots = ['SLUH', 'VA', 'ID', 'NF', 'MICU', 'Bronze', 'Cards']
    if t_diamond > 0:
        active_rots.append('Diamond')
    if t_gold > 0:
        active_rots.append('Gold')
    target_map = {'SLUH': t_sluh, 'VA': t_va, 'ID': t_id, 'NF': t_nf,
                  'MICU': t_micu, 'Bronze': t_bronze, 'Cards': t_cards,
                  'Diamond': t_diamond, 'Gold': t_gold}
else:
    TW_display = 48
    N_display = n_cat + n_prelim
    active_rots = ['SLUH', 'VA', 'NF', 'MICU', 'Cards']
    target_map = {'SLUH': t_sluh, 'VA': t_va, 'NF': t_nf, 'MICU': t_micu, 'Cards': t_cards}

for rot in active_rots:
    tgt = target_map.get(rot, 0)
    if tgt == 0:
        continue
    ideal = (tgt * TW_display) / max(N_display, 1)
    default_min = max(0, int(ideal * 0.4))
    default_max = int(ideal * 1.8) + 1
    mm_cols = st.sidebar.columns(3)
    mm_cols[0].markdown(f"**{rot}** (ideal: {ideal:.1f})")
    mn = mm_cols[1].number_input(f"Min", 0, 50, default_min, key=f"min_{rot}")
    mx = mm_cols[2].number_input(f"Max", 1, 50, default_max, key=f"max_{rot}")
    min_max[rot] = {'min': mn, 'max': mx}

# ── Block Types ──
st.sidebar.markdown("### Block Types")
st.sidebar.caption("Choose how each rotation is structured: contiguous blocks, alternating ABABA, or single weeks.")

block_types = {}
block_options = ["ABABA (3×1wk)", "MarioKart (3wk)", "2-week", "1-week"]

if level == "Senior":
    defaults = {
        'SLUH': 'MarioKart (3wk)', 'VA': 'MarioKart (3wk)', 'ID': 'MarioKart (3wk)',
        'NF': '2-week',
        'MICU': 'ABABA (3×1wk)', 'Bronze': 'ABABA (3×1wk)', 'Cards': 'ABABA (3×1wk)',
        'Diamond': '1-week', 'Gold': '1-week',
    }
    bt_col1, bt_col2, bt_col3 = st.sidebar.columns(3)
    with bt_col1:
        block_types['SLUH'] = st.selectbox("SLUH", block_options,
                                           index=block_options.index(defaults['SLUH']), key='bt_sluh')
        block_types['MICU'] = st.selectbox("MICU", block_options,
                                           index=block_options.index(defaults['MICU']), key='bt_micu')
        if t_diamond > 0:
            block_types['Diamond'] = st.selectbox("Diamond", block_options,
                                                 index=block_options.index(defaults['Diamond']), key='bt_diamond')
    with bt_col2:
        block_types['VA'] = st.selectbox("VA", block_options,
                                         index=block_options.index(defaults['VA']), key='bt_va')
        block_types['Bronze'] = st.selectbox("Bronze", block_options,
                                            index=block_options.index(defaults['Bronze']), key='bt_bronze')
        block_types['Gold'] = st.selectbox("Gold", block_options,
                                          index=block_options.index(defaults['Gold']), key='bt_gold')
    with bt_col3:
        block_types['ID'] = st.selectbox("ID", block_options,
                                         index=block_options.index(defaults['ID']), key='bt_id')
        block_types['Cards'] = st.selectbox("Cards", block_options,
                                           index=block_options.index(defaults['Cards']), key='bt_cards')
        block_types['NF'] = st.selectbox("NF", block_options,
                                        index=block_options.index(defaults['NF']), key='bt_nf')
    if t_other1 > 0 or t_other2 > 0:
        st.sidebar.markdown("*Custom rotations*")
        o_cols = st.sidebar.columns(2)
        if t_other1 > 0:
            block_types['IP Other 1'] = o_cols[0].selectbox("IP Other 1", block_options, index=3, key="bt_other1")
        if t_other2 > 0:
            block_types['IP Other 2'] = o_cols[1].selectbox("IP Other 2", block_options, index=3, key="bt_other2")
else:
    defaults = {
        'SLUH': 'MarioKart (3wk)', 'VA': 'MarioKart (3wk)',
        'NF': '2-week',
        'MICU': 'ABABA (3×1wk)', 'Cards': 'ABABA (3×1wk)',
    }
    bt_col1, bt_col2, bt_col3 = st.sidebar.columns(3)
    with bt_col1:
        block_types['SLUH'] = st.selectbox("SLUH", block_options,
                                           index=block_options.index(defaults['SLUH']), key='bt_sluh')
        block_types['MICU'] = st.selectbox("MICU", block_options,
                                           index=block_options.index(defaults['MICU']), key='bt_micu')
    with bt_col2:
        block_types['VA'] = st.selectbox("VA", block_options,
                                         index=block_options.index(defaults['VA']), key='bt_va')
        block_types['Cards'] = st.selectbox("Cards", block_options,
                                           index=block_options.index(defaults['Cards']), key='bt_cards')
    with bt_col3:
        block_types['NF'] = st.selectbox("NF", block_options,
                                        index=block_options.index(defaults['NF']), key='bt_nf')
    if t_other1 > 0 or t_other2 > 0:
        st.sidebar.markdown("*Custom rotations*")
        o_cols = st.sidebar.columns(2)
        if t_other1 > 0:
            block_types['IP Other 1'] = o_cols[0].selectbox("IP Other 1", block_options, index=3, key="bt_other1")
        if t_other2 > 0:
            block_types['IP Other 2'] = o_cols[1].selectbox("IP Other 2", block_options, index=3, key="bt_other2")

st.sidebar.markdown("### Scheduling Rules")
col_r1, col_r2 = st.sidebar.columns(2)
ward_len = col_r1.number_input("Ward weeks", 1, 5, 3)
nf_len = col_r2.number_input("NF weeks", 1, 4, 2)
col_r3, col_r4 = st.sidebar.columns(2)
max_consec = col_r3.number_input("Max consec IP", 2, 6, 3)
jeop_cap = col_r4.number_input("Jeopardy cap", 1, 8, 4 if level == "Senior" else 3)
clinic_freq = st.sidebar.number_input("Clinic every N weeks", 4, 8, 6)
max_stagger = st.sidebar.number_input("Max block starts/week", 1, 10, 2,
                                       help="Maximum residents starting a multi-week block in the same week")
col_r5, col_r6 = st.sidebar.columns(2)
ip_window = col_r5.number_input("IP window (wks)", 4, 12, 6,
                                 help="Sliding window size for IP cap check")
max_ip_win = col_r6.number_input("Max IP in window", 2, 6, 3,
                                  help="Max inpatient weeks allowed in any window")
nf_min_gap = st.sidebar.number_input("Min weeks between NF blocks", 2, 12, 6,
                                      help="Minimum gap between Night Float blocks for same resident")

st.sidebar.markdown("### Seed")
seed = st.sidebar.number_input("Random seed", 0, 9999, 18 if level == "Senior" else 55)

search_seed = st.sidebar.button("Find Best Seed (0-99)")

st.sidebar.divider()
save_baseline = st.sidebar.button("Save as Baseline")

# ═══════════════════════════════════════════════════════════════════════
# BUILD PARAMS AND GENERATE
# ═══════════════════════════════════════════════════════════════════════
params = {
    'seed': seed, 'max_consec': max_consec, 'ward_len': ward_len,
    'nf_len': nf_len, 'jeop_cap': jeop_cap, 'clinic_freq': clinic_freq,
    'max_stagger': max_stagger, 'ip_window': ip_window,
    'max_ip_win': max_ip_win, 'nf_min_gap': nf_min_gap,
    't_sluh': t_sluh, 't_va': t_va, 't_nf': t_nf, 't_micu': t_micu, 't_cards': t_cards,
    't_other1': t_other1, 't_other2': t_other2,
    'block_types': block_types,
    'min_max': min_max,
}

if level == "Senior":
    params.update({
        'n_pgy3': n_pgy3, 'n_pgy2': n_pgy2,
        't_id': t_id, 't_bronze': t_bronze, 't_diamond': t_diamond, 't_gold': t_gold,
    })
else:
    params.update({
        'n_cat': n_cat, 'n_prelim': n_prelim,
        'n_neuro': n_neuro, 'n_anes': n_anes, 'n_psych': n_psych, 'n_em': n_em,
        'neuro_months': neuro_mo,
    })

# Seed search
if search_seed:
    tw = 96 if level == 'Senior' else 48
    with st.spinner(f"Searching seeds 0-99 for best coverage out of {tw} weeks..."):
        best_seed, best_staffed = find_best_seed(params, level)
    st.sidebar.success(f"Best seed: **{best_seed}** ({best_staffed}/{tw} staffed)")
    params['seed'] = best_seed
    seed = best_seed

# Generate schedule
if level == "Senior":
    data = build_senior(params)
else:
    data = build_intern(params)

TW = data.get('total_weeks', 48)

# Save baseline to session state
if save_baseline:
    st.session_state['baseline'] = data
    st.session_state['baseline_level'] = level
    st.session_state['baseline_params'] = dict(params)
    st.toast("Baseline saved!", icon="📌")

# ═══════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════
years_label = "2-Year" if level == "Senior" else "1-Year"
st.markdown(f"# Project Z — {level} Schedule ({years_label}, PGY-{'2/3' if level == 'Senior' else '1'})")

st.markdown(f"""
*Welcome to the Project Z Schedule Explorer. This tool lets you design and test ABC (X+Y+Z)
residency schedules interactively. {"Senior schedules span 96 weeks (2 years)." if level == "Senior" else "Intern schedules span 48 weeks (1 year)."}
Adjust parameters in the sidebar, explore the generated schedule across tabs, and export your results.
Role letters (A, B, C...) distinguish residents sharing the same rotation in a given week.*
""")

# KPIs
n_residents = len(data['residents'])
n_rotators = len(data.get('rotators', []))
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Residents", f"{n_residents}" + (f" + {n_rotators} rotators" if n_rotators else ""))
staffed_gaps = TW - data['fully_staffed']
staffed_msg = "Perfect!" if staffed_gaps == 0 else f"{staffed_gaps} gaps"
k2.metric("Fully Staffed", f"{data['fully_staffed']}/{TW}", delta=staffed_msg)
max_ip_w = data.get('max_ip_win', max_ip_win)
k3.metric(f"Max IP/{ip_window}wk", max((r.get('maxIPWin', r.get('maxConsec', 0)) for r in data['residents']), default=0),
          delta="OK" if data['violations'] == 0 else "VIOLATION",
          delta_color="normal" if data['violations'] == 0 else "inverse")
k4.metric("IP Window Violations", data['violations'],
          delta="None" if data['violations'] == 0 else f"{data['violations']} residents",
          delta_color="normal" if data['violations'] == 0 else "inverse")

total_op = sum(r['counts'].get('OP', 0) + r['counts'].get('Jeopardy', 0) for r in data['residents'])
total_clinic = sum(r['counts'].get('Clinic', 0) for r in data['residents'])
k5.metric("Total Clinic", total_clinic)
k6.metric("Total OP", total_op)

# ═══════════════════════════════════════════════════════════════════════
# APPLY MANUAL EDITS FROM SESSION STATE
# ═══════════════════════════════════════════════════════════════════════
if 'manual_edits' in st.session_state:
    for edit in st.session_state['manual_edits']:
        rid = edit['resident']
        w = edit['week']
        new_rot = edit['rotation']
        for r in data['residents']:
            if r['id'] == rid and 0 <= w < TW:
                old_rot = r['schedule'][w]
                r['schedule'][w] = new_rot
                r['counts'][old_rot] = r['counts'].get(old_rot, 0) - 1
                r['counts'][new_rot] = r['counts'].get(new_rot, 0) + 1
                if old_rot in data['coverage']:
                    data['coverage'][old_rot][w] -= 1
                if new_rot in data['coverage']:
                    data['coverage'][new_rot][w] += 1
                else:
                    data['coverage'][new_rot] = [0] * TW
                    data['coverage'][new_rot][w] = 1
                break

# ═══════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Schedule Grid", "By Rotation", "Coverage & Staffing", "Balance & Fairness", "Compare", "Edit Schedule"
])

# ── TAB 1: Schedule Grid ──
with tab1:
    # Legend
    legend_html = '<div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:12px;">'
    if level == "Senior":
        rots = ['SLUH', 'VA', 'ID', 'NF', 'MICU', 'Bronze', 'Cards', 'Gold', 'OP', 'Clinic', 'Jeopardy']
        if t_diamond > 0:
            rots.insert(7, 'Diamond')
        if t_other1 > 0:
            rots.insert(-2, 'IP Other 1')
        if t_other2 > 0:
            rots.insert(-2, 'IP Other 2')
    else:
        rots = ['SLUH', 'VA', 'NF', 'MICU', 'Cards', 'ICU*', 'OP', 'Clinic', 'Jeopardy']
        if t_other1 > 0:
            rots.insert(-2, 'IP Other 1')
        if t_other2 > 0:
            rots.insert(-2, 'IP Other 2')
    for rot in rots:
        bg = COLORS.get(rot, '#fff')
        fc = '#fff' if rot in DARK_BG else '#333'
        legend_html += f'<span style="background:{bg};color:{fc};padding:2px 8px;border-radius:4px;font-size:11px;border:1px solid #ccc;">{ABBREV.get(rot, rot)}</span>'
    legend_html += '</div>'
    st.markdown(legend_html, unsafe_allow_html=True)

    # Sort and highlight controls
    ctrl1, ctrl2 = st.columns(2)
    sort_mode = ctrl1.selectbox("Sort residents by", ["Clinic position (cascade)", "PGY / Resident ID"],
                                 index=0)
    sort_key = 'clinic' if 'Clinic' in sort_mode else 'pgy_id'
    highlight_jeo = ctrl2.checkbox("Highlight Jeopardy weeks (dim everything else)", value=False)

    st.caption(f"Each row is one resident. "
               f"Scroll right to see all {TW} weeks. Role letters (A/B/C) distinguish residents "
               f"on MICU, Bronze, Cards, and NF in the same week. Jeo column shows jeopardy count.")

    grid_html = render_schedule_html(data, level, params, highlight_jeo=highlight_jeo, sort_mode=sort_key)
    st.markdown(grid_html, unsafe_allow_html=True)

    # Export button
    excel_data = export_to_excel(data, level, params)
    st.download_button(
        label="📥 Export to Excel",
        data=excel_data,
        file_name=f"Project_Z_{level}_Schedule.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

# ── TAB 2: By Rotation ──
with tab2:
    st.caption("Gantt-style view: each rotation section shows one row per resident assigned to it. "
               "Colored cells mark weeks on that rotation; blank cells are other assignments.")

    br_residents = data['residents']
    role_labels = data.get('role_labels', {})

    # Build rotation order
    if level == "Senior":
        rot_order = ['SLUH', 'VA', 'ID', 'NF', 'MICU', 'Bronze', 'Cards', 'Gold', 'Jeopardy']
        if t_diamond > 0:
            rot_order.insert(7, 'Diamond')
        if t_other1 > 0:
            rot_order.insert(-1, 'IP Other 1')
        if t_other2 > 0:
            rot_order.insert(-1, 'IP Other 2')
    else:
        rot_order = ['SLUH', 'VA', 'NF', 'MICU', 'Cards', 'ICU*', 'Jeopardy']
        if t_other1 > 0:
            rot_order.insert(-1, 'IP Other 1')
        if t_other2 > 0:
            rot_order.insert(-1, 'IP Other 2')

    # For each rotation, find which residents are ever assigned to it
    rot_residents = {}
    for rot in rot_order:
        rids = []
        for r in br_residents:
            if any(r['schedule'][w] == rot for w in range(TW)):
                rids.append(r)
        # Also include rotators for intern SLUH/VA/ICU*
        if level == "Intern" and rot in ('SLUH', 'VA', 'ICU*'):
            for rr in data.get('rotators', []):
                if any(rr['schedule'][w] == rot for w in range(TW)):
                    rids.append({'id': rr['id'], 'schedule': rr['schedule'],
                                 'pgy': 0, 'type': rr.get('type', 'rotator'),
                                 '_rotator': True})
        rot_residents[rot] = rids

    # Build HTML
    html = '<div class="schedule-grid"><table>'

    # Year header for 96-week
    if TW > 48:
        html += '<tr><th colspan="2"></th>'
        html += f'<th colspan="48" class="year-hdr">Year 1 (Weeks 1–48)</th>'
        html += f'<th colspan="{TW - 48}" class="year-hdr">Year 2 (Weeks 49–{TW})</th>'
        html += '<th></th></tr>'

    # Week number header
    html += '<tr><th style="min-width:60px;">Rotation</th><th style="min-width:65px;">Resident</th>'
    for w in range(TW):
        html += f'<th>{w+1}</th>'
    html += '<th style="min-width:30px;">Wks</th></tr>'

    for rot in rot_order:
        rids = rot_residents[rot]
        if not rids:
            continue

        bg = COLORS.get(rot, '#fff')
        fc = '#fff' if rot in DARK_BG else '#333'
        abbr = ABBREV.get(rot, rot)
        n_res = len(rids)

        for idx, r in enumerate(rids):
            rid = r['id']
            is_rotator = r.get('_rotator', False)
            wk_count = sum(1 for w in range(TW) if r['schedule'][w] == rot)

            # Rotation label only on first row, spanning all rows for this rotation
            html += '<tr>'
            if idx == 0:
                html += (f'<td rowspan="{n_res}" style="background:{bg};color:{fc};'
                         f'font-weight:700;white-space:nowrap;vertical-align:middle;'
                         f'text-align:center;position:sticky;left:0;z-index:1;'
                         f'border-bottom:2px solid #999;">{abbr}<br>'
                         f'<span style="font-size:8px;font-weight:400;">({n_res})</span></td>')

            # Resident name
            rid_style = 'font-style:italic;color:#888;' if is_rotator else ''
            html += (f'<td style="text-align:left;font-size:9px;white-space:nowrap;'
                     f'{rid_style}">{rid}</td>')

            # Week cells
            for w in range(TW):
                if r['schedule'][w] == rot:
                    letter = role_labels.get((rid, w), '')
                    if letter and rot in ROLE_LETTER_ROTS:
                        short = {'MICU': 'MC', 'Bronze': 'BZ', 'Cards': 'CR', 'NF': 'NF'}
                        txt = f'{short.get(rot, abbr[:2])}-{letter}'
                    else:
                        txt = abbr
                    bdr = '2px solid #999' if idx == n_res - 1 else '1px solid #ddd'
                    html += (f'<td style="background:{bg};color:{fc};font-size:8px;'
                             f'text-align:center;border-bottom:{bdr};">{txt}</td>')
                else:
                    bdr = '2px solid #999' if idx == n_res - 1 else '1px solid #ddd'
                    html += f'<td style="background:#fafafa;border-bottom:{bdr};"></td>'

            # Week count
            bdr = '2px solid #999' if idx == n_res - 1 else '1px solid #ddd'
            html += f'<td style="text-align:center;font-weight:600;font-size:9px;border-bottom:{bdr};">{wk_count}</td>'
            html += '</tr>'

    html += '</table></div>'
    st.markdown(html, unsafe_allow_html=True)

    # Target summary beneath
    st.markdown("#### Weekly Targets")
    tgt_display = data['targets']
    tgt_html = '<div style="display:flex;gap:12px;flex-wrap:wrap;">'
    for rot in rot_order:
        if rot in ('Jeopardy',):
            continue
        tgt = tgt_display.get(rot, 0)
        rbg = COLORS.get(rot, '#fff')
        rfc = '#fff' if rot in DARK_BG else '#333'
        tgt_html += (f'<span style="background:{rbg};color:{rfc};padding:4px 10px;'
                     f'border-radius:4px;font-size:12px;border:1px solid #ccc;">'
                     f'{ABBREV.get(rot,rot)}: {tgt}/wk</span>')
    tgt_html += '</div>'
    st.markdown(tgt_html, unsafe_allow_html=True)

# ── TAB 3: Coverage & Staffing ──
with tab3:
    st.caption("Green cells meet the weekly target. Red cells are understaffed. The heatmap gives a bird's-eye view of coverage across all weeks.")

    targets = data['targets']
    cov = data['coverage']
    rot_list = [r for r in targets if r != 'Jeopardy']

    # Heatmap
    z_data = []
    labels = []
    for rot in rot_list:
        vals = cov.get(rot, [0] * TW)
        if level == "Intern" and rot in ('SLUH', 'VA') and 'rotator_coverage' in data:
            vals = [vals[w] + data['rotator_coverage'].get(rot, [0] * TW)[w] for w in range(TW)]
        z_data.append(vals)
        labels.append(rot)

    fig_heat = go.Figure(data=go.Heatmap(
        z=z_data,
        x=[f'W{w+1}' for w in range(TW)],
        y=labels,
        colorscale='RdYlGn',
        text=[[str(v) for v in row] for row in z_data],
        texttemplate='%{text}',
        textfont={'size': 8},
    ))
    fig_heat.update_layout(
        title='Weekly Coverage Heatmap',
        height=max(250, len(labels) * 35 + 100),
        margin=dict(l=80, r=20, t=40, b=30),
        xaxis={'dtick': 1},
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # Intern: Stacked bar for SLUH/VA
    if level == "Intern" and 'rotator_coverage' in data:
        st.markdown("#### SLUH & VA: Intern vs Rotator Contribution")
        for ward in ['SLUH', 'VA']:
            intern_vals = cov.get(ward, [0] * TW)
            rotator_vals = data['rotator_coverage'].get(ward, [0] * TW)
            target = targets.get(ward, 0)

            fig_stack = go.Figure()
            fig_stack.add_trace(go.Bar(
                x=[f'W{w+1}' for w in range(TW)],
                y=intern_vals, name='Intern', marker_color=COLORS[ward],
            ))
            fig_stack.add_trace(go.Bar(
                x=[f'W{w+1}' for w in range(TW)],
                y=rotator_vals, name='Rotator', marker_color='#888',
            ))
            fig_stack.add_hline(y=target, line_dash='dash', line_color='red',
                               annotation_text=f'Target={target}')
            fig_stack.update_layout(
                barmode='stack', title=f'{ward} Coverage',
                height=250, margin=dict(l=40, r=20, t=40, b=30),
            )
            st.plotly_chart(fig_stack, use_container_width=True)

    # Coverage table
    st.markdown("#### Weekly Coverage Table")
    cov_df_data = {}
    for rot in rot_list:
        vals = cov.get(rot, [0] * TW)
        if level == "Intern" and rot in ('SLUH', 'VA') and 'rotator_coverage' in data:
            vals = [vals[w] + data['rotator_coverage'].get(rot, [0] * TW)[w] for w in range(TW)]
        cov_df_data[rot] = vals
    cov_df = pd.DataFrame(cov_df_data, index=[f'W{w+1}' for w in range(TW)])
    st.dataframe(cov_df.T, use_container_width=True)

# ── TAB 4: Balance & Fairness ──
with tab4:
    st.caption("These charts show how evenly rotations are distributed across residents. "
               "Outliers (>1.5 SD from mean) are flagged below. "
               "Red-highlighted cells in the schedule grid indicate residents below their minimum or above their maximum.")

    ip_rots = (set(['SLUH', 'VA', 'ID', 'NF', 'MICU', 'Bronze', 'Cards', 'Gold'])
               if level == "Senior" else set(['SLUH', 'VA', 'NF', 'MICU', 'Cards']))
    if level == "Senior" and t_diamond > 0:
        ip_rots.add('Diamond')
    rot_list_bal = sorted(ip_rots)

    # IP histogram
    ip_vals = [r['ip'] for r in data['residents']]
    fig_hist = px.histogram(x=ip_vals, nbins=15, title='IP Weeks Distribution',
                           labels={'x': 'IP Weeks', 'y': 'Count'})
    fig_hist.update_layout(height=300, margin=dict(l=40, r=20, t=40, b=30))
    st.plotly_chart(fig_hist, use_container_width=True)

    # Box plots per rotation
    box_data = []
    for rot in rot_list_bal:
        for r in data['residents']:
            box_data.append({'Rotation': rot, 'Weeks': r['counts'].get(rot, 0)})
    box_df = pd.DataFrame(box_data)
    fig_box = px.box(box_df, x='Rotation', y='Weeks', title='Rotation Distribution Across Residents',
                     color='Rotation')
    fig_box.update_layout(height=400, margin=dict(l=40, r=20, t=40, b=30), showlegend=False)
    st.plotly_chart(fig_box, use_container_width=True)

    # Outlier analysis
    st.markdown("#### Outlier Analysis (>1.5 SD from mean)")
    outliers = []
    for rot in rot_list_bal:
        vals = [r['counts'].get(rot, 0) for r in data['residents']]
        mean = np.mean(vals)
        std = np.std(vals)
        if std > 0:
            for r in data['residents']:
                v = r['counts'].get(rot, 0)
                if abs(v - mean) > 1.5 * std:
                    outliers.append({
                        'Resident': r['id'],
                        'Rotation': rot,
                        'Weeks': v,
                        'Mean': f'{mean:.1f}',
                        'SD': f'{std:.1f}',
                        'Deviation': f'{(v - mean) / std:+.1f} SD',
                    })
    if outliers:
        st.dataframe(pd.DataFrame(outliers), use_container_width=True, hide_index=True)
    else:
        st.success("No outliers detected — excellent balance!")

    # Min constraint violations
    min_viol_residents = [r for r in data['residents'] if r.get('min_violations')]
    if min_viol_residents:
        st.markdown("#### ⚠️ Residents Below Minimum Weeks")
        min_viol_data = []
        for r in min_viol_residents:
            for rot in r['min_violations']:
                min_viol_data.append({
                    'Resident': r['id'],
                    'Rotation': rot,
                    'Actual': r['counts'].get(rot, 0),
                    'Minimum': data.get('min_per_res', {}).get(rot, 0),
                })
        st.dataframe(pd.DataFrame(min_viol_data), use_container_width=True, hide_index=True)

    # Sortable comparison table
    st.markdown("#### Per-Resident Rotation Counts")
    table_data = []
    for r in data['residents']:
        row = {'Resident': r['id'], 'IP': r['ip'], 'OP': r['op'], 'Clinic': r['clinic'],
               'Jeopardy': r.get('jeopardy', 0), 'MaxConsec': r['maxConsec']}
        for rot in rot_list_bal:
            row[rot] = r['counts'].get(rot, 0)
        table_data.append(row)
    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

# ── TAB 5: Compare ──
with tab5:
    if 'baseline' not in st.session_state:
        st.info("No baseline saved yet. Use the **Save as Baseline** button in the sidebar, change parameters, then come back here to compare.")
    elif st.session_state.get('baseline_level') != level:
        st.warning(f"Baseline was saved for **{st.session_state['baseline_level']}** but you're viewing **{level}**. Switch levels or save a new baseline.")
    else:
        bl = st.session_state['baseline']
        bl_params = st.session_state['baseline_params']
        bl_tw = bl.get('total_weeks', 48)

        st.markdown("### Baseline vs Current")

        c1, c2, c3, c4 = st.columns(4)
        delta_staffed = data['fully_staffed'] - bl['fully_staffed']
        delta_consec = data['max_consec'] - bl['max_consec']
        delta_viol = data['violations'] - bl['violations']

        c1.metric("Fully Staffed", f"{data['fully_staffed']}/{TW}",
                  delta=f"{delta_staffed:+d}" if delta_staffed != 0 else "Same")
        c2.metric("Max Consec IP", data['max_consec'],
                  delta=f"{delta_consec:+d}" if delta_consec != 0 else "Same",
                  delta_color="inverse")
        c3.metric("Violations", data['violations'],
                  delta=f"{delta_viol:+d}" if delta_viol != 0 else "Same",
                  delta_color="inverse")

        st.markdown("#### Parameter Changes")
        changes = []
        for key in sorted(set(list(bl_params.keys()) + list(params.keys()))):
            old = bl_params.get(key)
            new = params.get(key)
            if old != new:
                changes.append({'Parameter': key, 'Baseline': str(old), 'Current': str(new)})
        if changes:
            st.dataframe(pd.DataFrame(changes), use_container_width=True, hide_index=True)
        else:
            st.info("No parameter changes detected.")

        st.markdown("#### Coverage Comparison")
        targets_list = [r for r in data['targets'] if r != 'Jeopardy']
        comp_data = []
        for rot in targets_list:
            bl_vals = bl['coverage'].get(rot, [0] * bl_tw)
            cur_vals = data['coverage'].get(rot, [0] * TW)
            bl_tgt = bl['targets'].get(rot, 0)
            cur_tgt = data['targets'].get(rot, 0)

            bl_met = sum(1 for v in bl_vals if v >= bl_tgt)
            cur_met = sum(1 for v in cur_vals if v >= cur_tgt)

            comp_data.append({
                'Rotation': rot,
                'Baseline Target': bl_tgt,
                'Current Target': cur_tgt,
                'Baseline Weeks Met': f'{bl_met}/{bl_tw}',
                'Current Weeks Met': f'{cur_met}/{TW}',
                'Change': f'{cur_met - bl_met:+d}',
            })
        st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)

# ── TAB 5: Edit Schedule ──
with tab6:
    st.markdown("### Manual Schedule Adjustments")
    st.caption("Changes are applied on top of the generated schedule. They persist until you click 'Clear All Edits' or refresh the page.")

    if 'manual_edits' not in st.session_state:
        st.session_state['manual_edits'] = []

    with st.form("edit_form"):
        ec1, ec2, ec3 = st.columns(3)
        resident_ids = [r['id'] for r in data['residents']]
        edit_resident = ec1.selectbox("Resident", resident_ids)
        edit_week = ec2.number_input("Week", 1, TW, 1)

        if level == "Senior":
            rot_options = ['SLUH', 'VA', 'ID', 'NF', 'MICU', 'Bronze', 'Cards', 'Gold', 'OP', 'Jeopardy']
            if t_diamond > 0:
                rot_options.insert(7, 'Diamond')
            if t_other1 > 0:
                rot_options.insert(-1, 'IP Other 1')
            if t_other2 > 0:
                rot_options.insert(-1, 'IP Other 2')
        else:
            rot_options = ['SLUH', 'VA', 'NF', 'MICU', 'Cards', 'OP', 'Jeopardy']
            if t_other1 > 0:
                rot_options.insert(-1, 'IP Other 1')
            if t_other2 > 0:
                rot_options.insert(-1, 'IP Other 2')
        edit_rot = ec3.selectbox("New Rotation", rot_options)

        submitted = st.form_submit_button("Apply Change")
        if submitted:
            st.session_state['manual_edits'].append({
                'resident': edit_resident,
                'week': edit_week - 1,
                'rotation': edit_rot,
            })
            st.rerun()

    if st.session_state['manual_edits']:
        st.markdown("#### Pending Edits")
        edits_df = pd.DataFrame([
            {'Resident': e['resident'], 'Week': e['week'] + 1, 'New Rotation': e['rotation']}
            for e in st.session_state['manual_edits']
        ])
        st.dataframe(edits_df, use_container_width=True, hide_index=True)

        if st.button("Clear All Edits"):
            st.session_state['manual_edits'] = []
            st.rerun()
