"""
Project Z — Schedule Explorer
Interactive Streamlit app for exploring ABC residency scheduling configurations.
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


# ═══════════════════════════════════════════════════════════════════════
# SENIOR SCHEDULER (exact v15 algorithm)
# ═══════════════════════════════════════════════════════════════════════
def build_senior(params):
    """Build senior PGY-2/3 schedule. Returns dict with schedule, coverage, residents, stats."""
    random.seed(params['seed'])

    N_PGY3 = params['n_pgy3']
    N_PGY2 = params['n_pgy2']
    N = N_PGY3 + N_PGY2
    TW = 48
    CYCLE = params.get('clinic_freq', 6)
    MAX_CONSEC = params.get('max_consec', 3)
    WARD_LEN = params.get('ward_len', 3)
    NF_LEN = params.get('nf_len', 2)
    JEOP_CAP = params.get('jeop_cap', 4)

    TARGETS = {
        'SLUH': params['t_sluh'], 'VA': params['t_va'], 'ID': params['t_id'],
        'NF': params['t_nf'], 'MICU': params['t_micu'], 'Bronze': params['t_bronze'],
        'Cards': params['t_cards'], 'Diamond': params['t_diamond'], 'Gold': params['t_gold'],
        'IP Other 1': params.get('t_other1', 0), 'IP Other 2': params.get('t_other2', 0),
    }
    IP_ROTS = set(TARGETS.keys()) | {'Elective'}
    STAG_ROTS = ['MICU', 'Bronze', 'Cards']

    # Remove zero-target rotations from TARGETS (but keep in IP_ROTS for consecutive-IP checking)
    TARGETS = {k: v for k, v in TARGETS.items() if v > 0}

    IDEAL = {rot: (tgt * TW) / max(N, 1) for rot, tgt in TARGETS.items()}

    # Build residents
    residents = []
    for pgy in [3, 2]:
        n = N_PGY3 if pgy == 3 else N_PGY2
        for i in range(n):
            residents.append({'id': f'R{pgy}_{i+1:02d}', 'pgy': pgy})

    # Clinic positions
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

    def would_exceed(rid, weeks_to_fill):
        temp = list(schedule[rid])
        for w in weeks_to_fill:
            temp[w] = 'IP'
        cur = 0
        for w in range(TW):
            if temp[w] in IP_ROTS or temp[w] == 'IP':
                cur += 1
                if cur > MAX_CONSEC:
                    return True
            else:
                cur = 0
        return False

    def balance_score(rid, rot, extra_weeks=1):
        current = res_weeks[rid][rot]
        ideal = IDEAL.get(rot, 1.0)
        rot_excess = (current + extra_weeks) / max(ideal, 0.5)
        total_ip = sum(res_weeks[rid][r] for r in TARGETS)
        ideal_mean_ip = sum(TARGETS.values()) * TW / max(N, 1)
        return rot_excess + 0.3 * total_ip / max(ideal_mean_ip, 1)

    # Build block types dict from params
    block_types = params.get('block_types', {})
    BT = {}
    for rot in ['SLUH', 'VA', 'ID']:
        BT[rot] = block_types.get(rot, 'MarioKart (3wk)')
    BT['NF'] = block_types.get('NF', '2-week')
    for rot in STAG_ROTS:
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
        # Left neighbor: must be non-IP or edge
        if w > 0 and s[w - 1] != '' and s[w - 1] not in ('OP', 'Clinic', 'Jeopardy'):
            return False
        # Right neighbor: must be non-IP or empty (will become OP later) or edge
        if w < TW - 1 and s[w + 1] != '' and s[w + 1] not in ('OP', 'Clinic', 'Jeopardy'):
            return False
        return True

    # Pass 1: MarioKart (3wk) rotations
    for ward_rot in mario_kart_rots:
        target = TARGETS[ward_rot]
        for w in range(TW - WARD_LEN + 1):
            while coverage[ward_rot][w] < target:
                cands = [r for r in residents
                         if all(is_free(r['id'], w + i) for i in range(WARD_LEN))
                         and not would_exceed(r['id'], list(range(w, w + WARD_LEN)))]
                if not cands:
                    break
                cands.sort(key=lambda r: (balance_score(r['id'], ward_rot, WARD_LEN), random.random()))
                rid = cands[0]['id']
                for i in range(WARD_LEN):
                    assign(rid, w + i, ward_rot)
                res_weeks[rid][ward_rot] += WARD_LEN

    # Pass 2: 2-week rotations
    for two_week_rot in two_week_rots:
        target = TARGETS[two_week_rot]
        for w in range(TW - NF_LEN + 1):
            while coverage[two_week_rot][w] < target:
                cands = [r for r in residents
                         if all(is_free(r['id'], w + i) for i in range(NF_LEN))
                         and not would_exceed(r['id'], list(range(w, w + NF_LEN)))]
                if not cands:
                    break
                cands.sort(key=lambda r: (balance_score(r['id'], two_week_rot, NF_LEN), random.random()))
                rid = cands[0]['id']
                for i in range(NF_LEN):
                    assign(rid, w + i, two_week_rot)
                res_weeks[rid][two_week_rot] += NF_LEN

    # Pass 3a: ABABA mini-blocks for ABABA rotations
    # Place one 3-week alternating block (w, w+2, w+4) per resident per rotation
    ababa_block_rots = sorted(ababa_rots, key=lambda r: TARGETS[r])
    block_used = {r['id']: set() for r in residents}
    for rot in ababa_block_rots:
        target = TARGETS[rot]
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
                         and not would_exceed(r['id'], weeks3)]
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
                         and not would_exceed(r['id'], [w])]
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

    # Pass 5b: Repair — fix remaining single-week IP gaps (ABABA and 1-week rotations)
    # First, simple OP→rot conversion.
    all_single_rots = ababa_rots + single_rots
    for rot in all_single_rots:
        target = TARGETS[rot]
        for w in range(TW):
            while coverage[rot][w] < target:
                cands = [r for r in residents
                         if schedule[r['id']][w] == 'OP'
                         and has_op_sandwich(r['id'], w)
                         and not would_exceed(r['id'], [w])]
                if not cands:
                    break
                cands.sort(key=lambda r: (balance_score(r['id'], rot, 1), random.random()))
                rid = cands[0]['id']
                schedule[rid][w] = rot
                coverage['OP'][w] -= 1
                coverage[rot][w] += 1
                res_weeks[rid][rot] += 1

    # Pass 5c: Swap repair — for each understaffed (rot, week), find a resident
    # who has rot on a week with surplus, and a second resident on the gap week
    # with OP + sandwich clearance. Swap assignments.
    for rot in all_single_rots:
        target = TARGETS[rot]
        for w in range(TW):
            if coverage[rot][w] >= target:
                continue
            # Find surplus weeks for this rotation
            surplus_wks = [w2 for w2 in range(TW) if coverage[rot][w2] > target]
            fixed = False
            for w2 in surplus_wks:
                if fixed:
                    break
                # Residents doing rot at surplus week w2
                donors = [r for r in residents if schedule[r['id']][w2] == rot]
                for donor in donors:
                    if fixed:
                        break
                    did = donor['id']
                    # Can this donor do rot at week w instead?
                    if schedule[did][w] == 'OP' and has_op_sandwich(did, w) and not would_exceed(did, [w]):
                        # Check donor can give up w2 without sandwich violation
                        # (w2 becomes OP, which is always fine for neighbors)
                        schedule[did][w2] = 'OP'
                        schedule[did][w] = rot
                        coverage[rot][w2] -= 1
                        coverage[rot][w] += 1
                        coverage['OP'][w2] += 1
                        coverage['OP'][w] -= 1
                        fixed = True

    # Pass 6: Jeopardy
    jeop_counts = collections.defaultdict(int)
    for w in range(TW):
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
        mx = c = 0
        for s in sched:
            if s in IP_ROTS:
                c += 1
                mx = max(mx, c)
            else:
                c = 0
        res_data.append({
            'id': rid, 'pgy': r['pgy'], 'pos': r['cpos'],
            'schedule': sched, 'counts': dict(counts),
            'ip': ip, 'op': counts.get('OP', 0) + counts.get('Jeopardy', 0),
            'clinic': counts.get('Clinic', 0), 'maxConsec': mx,
        })

    return {
        'residents': res_data,
        'coverage': {rot: list(coverage[rot]) for rot in TARGETS},
        'targets': TARGETS,
        'fully_staffed': fully,
        'max_consec': max((r['maxConsec'] for r in res_data), default=0),
        'violations': sum(1 for r in res_data if r['maxConsec'] > MAX_CONSEC),
    }


# ═══════════════════════════════════════════════════════════════════════
# INTERN SCHEDULER (exact v3 algorithm)
# ═══════════════════════════════════════════════════════════════════════
def build_intern(params):
    """Build intern PGY-1 schedule with rotators. Returns dict with all data."""
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

    ALL_IP = {'SLUH', 'VA', 'NF', 'MICU', 'Cards', 'IP Other 1', 'IP Other 2'}
    STAG = ['MICU', 'Cards']
    FULL = {
        'SLUH': params['t_sluh'], 'VA': params['t_va'],
        'NF': params['t_nf'], 'MICU': params['t_micu'], 'Cards': params['t_cards'],
        'IP Other 1': params.get('t_other1', 0), 'IP Other 2': params.get('t_other2', 0),
    }

    # Build block types dict from params
    block_types = params.get('block_types', {})
    BT = {}
    for rot in ['SLUH', 'VA']:
        BT[rot] = block_types.get(rot, 'MarioKart (3wk)')
    BT['NF'] = block_types.get('NF', '2-week')
    for rot in STAG:
        BT[rot] = block_types.get(rot, 'ABABA (3×1wk)')
    for rot in ['IP Other 1', 'IP Other 2']:
        BT[rot] = block_types.get(rot, '1-week')

    # Identify rotations by block type
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
    em = list(range(12))
    random.shuffle(em)
    for i in range(n_em):
        er = {'id': f'EM_{i+1}', 'type': 'EM', 'schedule': [''] * TW}
        for w in month_weeks(em[i % 12]):
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

    # Remove zero-target rotations from FULL (but keep in ALL_IP for consecutive-IP checking)
    FULL = {k: v for k, v in FULL.items() if v > 0}

    IDEAL = {
        'SLUH': sum(it_s) / max(N, 1), 'VA': sum(it_v) / max(N, 1),
        'NF': FULL['NF'] * TW / max(N, 1), 'MICU': FULL['MICU'] * TW / max(N, 1),
        'Cards': FULL['Cards'] * TW / max(N, 1),
    }
    # Add IP Other rotations to IDEAL if they have targets
    for rot in ['IP Other 1', 'IP Other 2']:
        if rot in FULL:
            IDEAL[rot] = FULL[rot] * TW / max(N, 1)

    schedule = {r['id']: [''] * TW for r in residents}
    coverage = {rot: [0] * TW for rot in ['SLUH', 'VA', 'NF', 'MICU', 'Cards', 'IP Other 1', 'IP Other 2', 'Jeopardy']}
    rw = {r['id']: collections.Counter() for r in residents}

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

    def exc(rid, wks):
        t = schedule[rid][:]
        for w in wks:
            t[w] = 'X'
        c = 0
        for w in range(TW):
            if t[w] in ALL_IP or t[w] == 'X':
                c += 1
                if c > MAX_CONSEC:
                    return True
            else:
                c = 0
        return False

    avg_ip = sum(it_s + it_v) + (FULL['NF'] + FULL['MICU'] + FULL['Cards']) * TW
    avg_ip /= max(N, 1)

    def bs(rid, rot, ew=1):
        cur = rw[rid][rot]
        ideal = IDEAL.get(rot, 1.0)
        return (cur + ew) / max(ideal, 0.5) + 0.3 * sum(rw[rid][r] for r in ALL_IP) / max(avg_ip, 1)

    # OP sandwich check: single-week IP must have non-IP neighbors
    def has_op_sandwich(rid, w):
        s = schedule[rid]
        if w > 0 and s[w - 1] != '' and s[w - 1] not in ('OP', 'Clinic', 'Jeopardy'):
            return False
        if w < TW - 1 and s[w + 1] != '' and s[w + 1] not in ('OP', 'Clinic', 'Jeopardy'):
            return False
        return True

    # Pass 1: MarioKart (3wk) rotations for interns (dynamic targets for SLUH/VA)
    if 'SLUH' in mario_kart_rots_intern:
        for w in range(TW - WARD_LEN + 1):
            while coverage['SLUH'][w] < it_s[w]:
                cs = [r for r in residents
                      if all(free(r['id'], w + i) for i in range(WARD_LEN))
                      and not exc(r['id'], list(range(w, w + WARD_LEN)))]
                if not cs:
                    break
                cs.sort(key=lambda r: (bs(r['id'], 'SLUH', WARD_LEN), random.random()))
                for i in range(WARD_LEN):
                    asgn(cs[0]['id'], w + i, 'SLUH')

    if 'VA' in mario_kart_rots_intern:
        for w in range(TW - WARD_LEN + 1):
            while coverage['VA'][w] < it_v[w]:
                cs = [r for r in residents
                      if all(free(r['id'], w + i) for i in range(WARD_LEN))
                      and not exc(r['id'], list(range(w, w + WARD_LEN)))]
                if not cs:
                    break
                cs.sort(key=lambda r: (bs(r['id'], 'VA', WARD_LEN), random.random()))
                for i in range(WARD_LEN):
                    asgn(cs[0]['id'], w + i, 'VA')

    # Pass 2: 2-week rotations for interns (NF)
    for w in range(TW - NF_LEN + 1):
        while coverage['NF'][w] < FULL['NF']:
            cs = [r for r in residents
                  if all(free(r['id'], w + i) for i in range(NF_LEN))
                  and not exc(r['id'], list(range(w, w + NF_LEN)))]
            if not cs:
                break
            cs.sort(key=lambda r: (bs(r['id'], 'NF', NF_LEN), random.random()))
            for i in range(NF_LEN):
                asgn(cs[0]['id'], w + i, 'NF')

    # Pass 3a: ABABA mini-blocks for ABABA rotations
    ababa_block_rots_intern = sorted(ababa_rots_intern, key=lambda r: FULL[r])
    i_block_used = {r['id']: set() for r in residents}
    for sr in ababa_block_rots_intern:
        tgt = FULL[sr]
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
                      and not exc(r['id'], weeks3)]
                if not cs:
                    break
                cs.sort(key=lambda r: (bs(r['id'], sr, 3), random.random()))
                rid = cs[0]['id']
                for ww in weeks3:
                    asgn(rid, ww, sr)
                i_block_used[rid].add(sr)
                placing = True

    # Pass 3b: Fill remaining single-week gaps (ABABA and 1-week rotations)
    all_single_rots_intern = ababa_rots_intern + single_rots_intern
    for w in range(TW):
        for sr in sorted(all_single_rots_intern, key=lambda r: FULL[r]):
            tgt = FULL[sr]
            while coverage[sr][w] < tgt:
                cs = [r for r in residents
                      if free(r['id'], w) and not exc(r['id'], [w])
                      and has_op_sandwich(r['id'], w)]
                if not cs:
                    break
                cs.sort(key=lambda r: (bs(r['id'], sr, 1), random.random()))
                asgn(cs[0]['id'], w, sr)

    # Pass 4: OP fill
    for r in residents:
        for w in range(TW):
            if schedule[r['id']][w] == '':
                schedule[r['id']][w] = 'OP'

    # Pass 5: Jeopardy
    jc = {r['id']: 0 for r in residents}
    for w in range(TW):
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
                coverage['SLUH'][w] + r_sluh[w] >= FULL['SLUH'] and
                coverage['VA'][w] + r_va[w] >= FULL['VA'] and
                coverage['NF'][w] >= FULL['NF'] and
                coverage['MICU'][w] >= FULL['MICU'] and
                coverage['Cards'][w] >= FULL['Cards'] and
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
        res_data.append({
            'id': rid, 'pgy': 1, 'pos': positions[rid], 'type': r['type'],
            'schedule': sched, 'counts': dict(counts),
            'ip': ip, 'op': counts.get('OP', 0) + counts.get('Jeopardy', 0),
            'clinic': counts.get('Clinic', 0), 'maxConsec': mx,
        })

    return {
        'residents': res_data,
        'coverage': {rot: list(coverage[rot]) for rot in coverage},
        'targets': FULL,
        'rotators': neuro + anes + psych + emr,
        'rotator_coverage': {'SLUH': r_sluh, 'VA': r_va},
        'fully_staffed': fully,
        'max_consec': max((r['maxConsec'] for r in res_data), default=0),
        'violations': sum(1 for r in res_data if r['maxConsec'] > MAX_CONSEC),
    }


# ═══════════════════════════════════════════════════════════════════════
# SEED SEARCH
# ═══════════════════════════════════════════════════════════════════════
def find_best_seed(params, level, max_seed=99):
    best_seed = 0
    best_staffed = 0
    for s in range(max_seed + 1):
        p = dict(params)
        p['seed'] = s
        result = build_senior(p) if level == 'Senior' else build_intern(p)
        if result['fully_staffed'] > best_staffed:
            best_staffed = result['fully_staffed']
            best_seed = s
            if best_staffed == 48:
                break
    return best_seed, best_staffed


# ═══════════════════════════════════════════════════════════════════════
# SCHEDULE GRID HTML
# ═══════════════════════════════════════════════════════════════════════
def render_schedule_html(data, level, params):
    """Render schedule as HTML table with colors."""
    TW = 48
    residents = data['residents']
    coverage = data['coverage']
    targets = data['targets']

    rot_list = list(targets.keys())
    if 'Jeopardy' in rot_list:
        rot_list.remove('Jeopardy')

    html = '<div class="schedule-grid"><table>'

    # Header
    html += '<tr><th style="min-width:65px;">Resident</th>'
    for w in range(1, TW + 1):
        html += f'<th>W{w}</th>'
    html += '<th>IP</th><th>OP</th><th>CL</th><th>MX</th>'
    for rot in rot_list:
        html += f'<th>{ABBREV.get(rot, rot)}</th>'
    html += '</tr>'

    # Resident rows
    for r in residents:
        html += f'<tr><td style="text-align:left;font-weight:500;">{r["id"]}</td>'
        for w in range(TW):
            val = r['schedule'][w]
            bg = COLORS.get(val, '#fff')
            txt = ABBREV.get(val, val or '')
            fc = '#fff' if val in DARK_BG else '#333'
            html += f'<td style="background:{bg};color:{fc};font-size:8px;">{txt}</td>'
        html += f'<td>{r["ip"]}</td><td>{r["op"]}</td><td>{r["clinic"]}</td>'
        html += f'<td style="{"background:#FFC7CE;" if r["maxConsec"] > params.get("max_consec",3) else ""}">{r["maxConsec"]}</td>'
        for rot in rot_list:
            ct = r['counts'].get(rot, 0)
            html += f'<td>{ct}</td>'
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
                html += f'<tr><td colspan="{TW + 5 + len(rot_list)}" class="section-hdr">{label} ROTATORS</td></tr>'
            html += f'<tr><td style="text-align:left;font-style:italic;">{rot_res["id"]}</td>'
            for w in range(TW):
                val = rot_res['schedule'][w]
                if val:
                    bg = COLORS.get(val, '#fff')
                    fc = '#fff' if val in DARK_BG else '#333'
                    html += f'<td style="background:{bg};color:{fc};font-size:8px;">{ABBREV.get(val, val)}</td>'
                else:
                    html += '<td style="background:#f0f0f0;"></td>'
            html += f'<td colspan="{4 + len(rot_list)}"></td></tr>'

    # Coverage summary
    html += f'<tr><td colspan="{TW + 5 + len(rot_list)}" class="summary-hdr">WEEKLY COVERAGE SUMMARY</td></tr>'
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
        html += f'<td colspan="{4 + len(rot_list)}"></td></tr>'

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
        html += f'<td colspan="{4 + len(rot_list)}"></td></tr>'

    html += '</table></div>'
    return html


# ═══════════════════════════════════════════════════════════════════════
# EXCEL EXPORT
# ═══════════════════════════════════════════════════════════════════════
def export_to_excel(data, level, params):
    """Export schedule data to an Excel file in memory."""
    output = io.BytesIO()

    TW = 48
    targets = data['targets']

    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Sheet 1: Schedule Grid
        rows = []
        for r in data['residents']:
            row = {'Resident': r['id'], 'PGY': r.get('pgy', ''), 'Position': r.get('pos', '')}
            for w in range(TW):
                row[f'Week {w+1}'] = r['schedule'][w]
            row['IP Weeks'] = r['ip']
            row['OP Weeks'] = r['op']
            row['Clinic'] = r['clinic']
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
            row['Weeks Met'] = f'{met}/48'
            cov_rows.append(row)
        pd.DataFrame(cov_rows).to_excel(writer, sheet_name='Coverage', index=False)

        # Sheet 3: Balance
        bal_rows = []
        for r in data['residents']:
            row = {'Resident': r['id'], 'IP': r['ip'], 'OP': r['op'], 'Clinic': r['clinic'],
                   'Max Consec': r['maxConsec']}
            for rot in sorted(targets.keys()):
                row[rot] = r['counts'].get(rot, 0)
            bal_rows.append(row)
        pd.DataFrame(bal_rows).to_excel(writer, sheet_name='Balance', index=False)

        # Sheet 4: Parameters
        param_rows = [{'Parameter': k, 'Value': str(v)} for k, v in sorted(params.items())]
        pd.DataFrame(param_rows).to_excel(writer, sheet_name='Parameters', index=False)

        # Format column widths
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
st.sidebar.caption("Adjust parameters below to regenerate the schedule in real-time. Use 'Find Best Seed' to optimize coverage.")

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
    t_diamond = col8.number_input("Diamond", 0, 5, 1)
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

st.sidebar.markdown("### Block Types")
st.sidebar.caption("Choose how each rotation is structured: contiguous blocks, alternating ABABA, or single weeks.")

block_types = {}
block_options = ["ABABA (3×1wk)", "MarioKart (3wk)", "2-week", "1-week"]

if level == "Senior":
    # Default block types for Senior
    defaults = {
        'SLUH': 'MarioKart (3wk)', 'VA': 'MarioKart (3wk)', 'ID': 'MarioKart (3wk)',
        'NF': '2-week',
        'MICU': 'ABABA (3×1wk)', 'Bronze': 'ABABA (3×1wk)', 'Cards': 'ABABA (3×1wk)',
        'Diamond': '1-week', 'Gold': '1-week',
    }
    senior_rots = ['SLUH', 'VA', 'ID', 'NF', 'MICU', 'Bronze', 'Cards', 'Diamond', 'Gold']

    # 3 columns of selectboxes
    bt_col1, bt_col2, bt_col3 = st.sidebar.columns(3)
    with bt_col1:
        block_types['SLUH'] = st.selectbox("SLUH", block_options,
                                           index=block_options.index(defaults['SLUH']), key='bt_sluh')
        block_types['MICU'] = st.selectbox("MICU", block_options,
                                           index=block_options.index(defaults['MICU']), key='bt_micu')
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
    # Custom rotations (IP Other 1 & 2) - only show if target > 0
    if t_other1 > 0 or t_other2 > 0:
        st.sidebar.markdown("*Custom rotations*")
        o_cols = st.sidebar.columns(2)
        if t_other1 > 0:
            block_types['IP Other 1'] = o_cols[0].selectbox("IP Other 1", block_options, index=3, key="bt_other1")
        if t_other2 > 0:
            block_types['IP Other 2'] = o_cols[1].selectbox("IP Other 2", block_options, index=3, key="bt_other2")
else:
    # Intern level
    defaults = {
        'SLUH': 'MarioKart (3wk)', 'VA': 'MarioKart (3wk)',
        'NF': '2-week',
        'MICU': 'ABABA (3×1wk)', 'Cards': 'ABABA (3×1wk)',
    }
    intern_rots = ['SLUH', 'VA', 'NF', 'MICU', 'Cards']

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
    # Custom rotations (IP Other 1 & 2) - only show if target > 0
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

st.sidebar.markdown("### Seed")
seed = st.sidebar.number_input("Random seed", 0, 9999, 18 if level == "Senior" else 6)

search_seed = st.sidebar.button("Find Best Seed (0-99)")

st.sidebar.divider()
save_baseline = st.sidebar.button("Save as Baseline")

# ═══════════════════════════════════════════════════════════════════════
# BUILD PARAMS AND GENERATE
# ═══════════════════════════════════════════════════════════════════════
params = {
    'seed': seed, 'max_consec': max_consec, 'ward_len': ward_len,
    'nf_len': nf_len, 'jeop_cap': jeop_cap, 'clinic_freq': clinic_freq,
    't_sluh': t_sluh, 't_va': t_va, 't_nf': t_nf, 't_micu': t_micu, 't_cards': t_cards,
    't_other1': t_other1, 't_other2': t_other2,
    'block_types': block_types,
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
    with st.spinner("Searching seeds 0-99..."):
        best_seed, best_staffed = find_best_seed(params, level)
    st.sidebar.success(f"Best seed: **{best_seed}** ({best_staffed}/48 staffed)")
    params['seed'] = best_seed
    seed = best_seed

# Generate schedule
if level == "Senior":
    data = build_senior(params)
else:
    data = build_intern(params)

# Save baseline to session state
if save_baseline:
    st.session_state['baseline'] = data
    st.session_state['baseline_level'] = level
    st.session_state['baseline_params'] = dict(params)
    st.toast("Baseline saved!", icon="📌")

# ═══════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════
st.markdown(f"# Project Z — {level} Schedule (PGY-{'2/3' if level == 'Senior' else '1'})")

st.markdown("""
*Welcome to the Project Z Schedule Explorer. This tool lets you design and test ABC (X+Y+Z)
residency schedules interactively. Adjust parameters in the sidebar, explore the generated schedule
across tabs, and export your results.*
""")

# KPIs
n_residents = len(data['residents'])
n_rotators = len(data.get('rotators', []))
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Residents", f"{n_residents}" + (f" + {n_rotators} rotators" if n_rotators else ""))
staffed_gaps = 48 - data['fully_staffed']
staffed_msg = "Perfect!" if staffed_gaps == 0 else f"{staffed_gaps} gaps"
k2.metric("Fully Staffed", f"{data['fully_staffed']}/48", delta=staffed_msg)
k3.metric("Max Consec IP", data['max_consec'],
          delta="OK" if data['max_consec'] <= max_consec else "VIOLATION",
          delta_color="normal" if data['max_consec'] <= max_consec else "inverse")
k4.metric("Violations", data['violations'],
          delta="None" if data['violations'] == 0 else f"{data['violations']} residents",
          delta_color="normal" if data['violations'] == 0 else "inverse")

# Calculate total OP and Clinic across all residents
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
            if r['id'] == rid and 0 <= w < 48:
                old_rot = r['schedule'][w]
                r['schedule'][w] = new_rot
                # Update counts
                r['counts'][old_rot] = r['counts'].get(old_rot, 0) - 1
                r['counts'][new_rot] = r['counts'].get(new_rot, 0) + 1
                # Update coverage
                if old_rot in data['coverage']:
                    data['coverage'][old_rot][w] -= 1
                if new_rot in data['coverage']:
                    if new_rot not in data['coverage']:
                        data['coverage'][new_rot] = [0] * 48
                    data['coverage'][new_rot][w] += 1
                else:
                    data['coverage'][new_rot] = [0] * 48
                    data['coverage'][new_rot][w] = 1
                break

# ═══════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Schedule Grid", "Coverage & Staffing", "Balance & Fairness", "Compare", "Edit Schedule"
])

# ── TAB 1: Schedule Grid ──
with tab1:
    # Legend
    legend_html = '<div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:12px;">'
    if level == "Senior":
        rots = ['SLUH', 'VA', 'ID', 'NF', 'MICU', 'Bronze', 'Cards', 'Diamond', 'Gold', 'OP', 'Clinic', 'Jeopardy']
        if t_other1 > 0:
            rots.insert(-2, 'IP Other 1')  # Insert before OP
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

    st.caption("Each row is one resident. Scroll right to see all 48 weeks. Color legend above. Coverage summary at bottom.")

    grid_html = render_schedule_html(data, level, params)
    st.markdown(grid_html, unsafe_allow_html=True)

    # Export button
    excel_data = export_to_excel(data, level, params)
    st.download_button(
        label="📥 Export to Excel",
        data=excel_data,
        file_name=f"Project_Z_{level}_Schedule.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

# ── TAB 2: Coverage & Staffing ──
with tab2:
    st.caption("Green cells meet the weekly target. Red cells are understaffed. The heatmap gives a bird's-eye view of coverage across all weeks.")

    targets = data['targets']
    cov = data['coverage']
    rot_list = [r for r in targets if r != 'Jeopardy']

    # Heatmap
    z_data = []
    labels = []
    for rot in rot_list:
        vals = cov.get(rot, [0] * 48)
        if level == "Intern" and rot in ('SLUH', 'VA') and 'rotator_coverage' in data:
            vals = [vals[w] + data['rotator_coverage'].get(rot, [0] * 48)[w] for w in range(48)]
        z_data.append(vals)
        labels.append(rot)

    fig_heat = go.Figure(data=go.Heatmap(
        z=z_data,
        x=[f'W{w+1}' for w in range(48)],
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

    # Intern: Stacked bar for SLUH/VA showing intern vs rotator
    if level == "Intern" and 'rotator_coverage' in data:
        st.markdown("#### SLUH & VA: Intern vs Rotator Contribution")
        for ward in ['SLUH', 'VA']:
            intern_vals = cov.get(ward, [0] * 48)
            rotator_vals = data['rotator_coverage'].get(ward, [0] * 48)
            target = targets[ward]

            fig_stack = go.Figure()
            fig_stack.add_trace(go.Bar(
                x=[f'W{w+1}' for w in range(48)],
                y=intern_vals, name='Intern', marker_color=COLORS[ward],
            ))
            fig_stack.add_trace(go.Bar(
                x=[f'W{w+1}' for w in range(48)],
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
        vals = cov.get(rot, [0] * 48)
        if level == "Intern" and rot in ('SLUH', 'VA') and 'rotator_coverage' in data:
            vals = [vals[w] + data['rotator_coverage'].get(rot, [0] * 48)[w] for w in range(48)]
        cov_df_data[rot] = vals
    cov_df = pd.DataFrame(cov_df_data, index=[f'W{w+1}' for w in range(48)])
    st.dataframe(cov_df.T, use_container_width=True)

# ── TAB 3: Balance & Fairness ──
with tab3:
    st.caption("These charts show how evenly rotations are distributed across residents. Outliers (>1.5 SD from mean) are flagged below.")

    ip_rots = (set(['SLUH', 'VA', 'ID', 'NF', 'MICU', 'Bronze', 'Cards', 'Diamond', 'Gold'])
               if level == "Senior" else set(['SLUH', 'VA', 'NF', 'MICU', 'Cards']))
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

    # Sortable comparison table
    st.markdown("#### Per-Resident Rotation Counts")
    table_data = []
    for r in data['residents']:
        row = {'Resident': r['id'], 'IP': r['ip'], 'OP': r['op'], 'Clinic': r['clinic'],
               'MaxConsec': r['maxConsec']}
        for rot in rot_list_bal:
            row[rot] = r['counts'].get(rot, 0)
        table_data.append(row)
    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

# ── TAB 4: Compare ──
with tab4:
    if 'baseline' not in st.session_state:
        st.info("No baseline saved yet. Use the **Save as Baseline** button in the sidebar, change parameters, then come back here to compare.")
    elif st.session_state.get('baseline_level') != level:
        st.warning(f"Baseline was saved for **{st.session_state['baseline_level']}** but you're viewing **{level}**. Switch levels or save a new baseline.")
    else:
        bl = st.session_state['baseline']
        bl_params = st.session_state['baseline_params']

        st.markdown("### Baseline vs Current")

        c1, c2, c3, c4 = st.columns(4)
        delta_staffed = data['fully_staffed'] - bl['fully_staffed']
        delta_consec = data['max_consec'] - bl['max_consec']
        delta_viol = data['violations'] - bl['violations']

        c1.metric("Fully Staffed", f"{data['fully_staffed']}/48",
                  delta=f"{delta_staffed:+d}" if delta_staffed != 0 else "Same")
        c2.metric("Max Consec IP", data['max_consec'],
                  delta=f"{delta_consec:+d}" if delta_consec != 0 else "Same",
                  delta_color="inverse")
        c3.metric("Violations", data['violations'],
                  delta=f"{delta_viol:+d}" if delta_viol != 0 else "Same",
                  delta_color="inverse")

        # Parameter changes
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

        # Coverage comparison
        st.markdown("#### Coverage Comparison")
        targets_list = [r for r in data['targets'] if r != 'Jeopardy']
        comp_data = []
        for rot in targets_list:
            bl_vals = bl['coverage'].get(rot, [0] * 48)
            cur_vals = data['coverage'].get(rot, [0] * 48)
            bl_tgt = bl['targets'].get(rot, 0)
            cur_tgt = data['targets'].get(rot, 0)

            bl_met = sum(1 for v in bl_vals if v >= bl_tgt)
            cur_met = sum(1 for v in cur_vals if v >= cur_tgt)

            comp_data.append({
                'Rotation': rot,
                'Baseline Target': bl_tgt,
                'Current Target': cur_tgt,
                'Baseline Weeks Met': f'{bl_met}/48',
                'Current Weeks Met': f'{cur_met}/48',
                'Change': f'{cur_met - bl_met:+d}',
            })
        st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)

# ── TAB 5: Edit Schedule ──
with tab5:
    st.markdown("### Manual Schedule Adjustments")
    st.caption("Changes are applied on top of the generated schedule. They persist until you click 'Clear All Edits' or refresh the page.")

    # Store edits in session state
    if 'manual_edits' not in st.session_state:
        st.session_state['manual_edits'] = []

    # Input form
    with st.form("edit_form"):
        ec1, ec2, ec3 = st.columns(3)
        resident_ids = [r['id'] for r in data['residents']]
        edit_resident = ec1.selectbox("Resident", resident_ids)
        edit_week = ec2.number_input("Week", 1, 48, 1)

        if level == "Senior":
            rot_options = ['SLUH', 'VA', 'ID', 'NF', 'MICU', 'Bronze', 'Cards', 'Diamond', 'Gold', 'OP', 'Jeopardy']
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
                'week': edit_week - 1,  # 0-indexed
                'rotation': edit_rot,
            })
            st.rerun()

    # Show and apply edits
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
