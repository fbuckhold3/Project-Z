"""
Project Z — Daily Shift Scheduler
Generates day-level MarioKart team rotations from block-schedule exports.
Deploy alongside schedule_explorer.py or run: streamlit run daily_scheduler.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import math
from collections import defaultdict

# ═══════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Project Z — Daily Shift Scheduler",
    page_icon="📅",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { max-width: 100%; }
    .cal-grid { font-size: 10px; overflow-x: auto; }
    .cal-grid table { border-collapse: collapse; width: auto; }
    .cal-grid th { background: #2F5496; color: white; padding: 3px 6px;
        font-size: 9px; text-align: center; position: sticky; top: 0; z-index: 2; }
    .cal-grid td { padding: 2px 4px; text-align: center; border: 1px solid #ddd;
        font-size: 9px; white-space: nowrap; min-width: 28px; }
    .kpi-card { background: white; border-radius: 10px; border: 1px solid #dfe3ea;
        padding: 16px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,.06); }
    .kpi-val { font-size: 28px; font-weight: 700; color: #4472C4; }
    .kpi-label { font-size: 11px; color: #888; margin-top: 4px; }
    td.day-off { background: #F0F0F0 !important; color: #999; }
    td.day-work { font-weight: 600; }
    td.day-violation { border: 2px solid red !important; }
    .team-mario { background: #FFCCCC; }
    .team-luigi { background: #CCFFCC; }
    .team-peach { background: #FFE6FF; }
    .team-yoshi { background: #FFFFCC; }
    .team-bowser { background: #E0E0E0; }
    .team-header { font-weight: 700; font-size: 12px; padding: 4px 8px; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════
SLUH_TEAMS = ["Mario", "Luigi", "Peach", "Yoshi", "Bowser"]
SLUH_FLOORS = ["Red", "Green", "White", "Yellow"]  # 4 active floors
VA_TEAMS = ["A", "B", "C", "D", "E"]
VA_FLOORS = ["A", "B", "C", "D"]
NF_SR_COVERS = ["Medicine", "MICU", "VA", "Cards/Bronze"]
NF_INT_COVERS = ["Medicine-1", "Medicine-2", "MICU"]
TEAM_COLORS = {
    "Mario": "#FFCCCC", "Luigi": "#CCFFCC", "Peach": "#FFE6FF",
    "Yoshi": "#FFFFCC", "Bowser": "#E0E0E0",
    "A": "#B3D9FF", "B": "#FFCC99", "C": "#D9B3FF",
    "D": "#C8E6C9", "E": "#FFE0B2",
}
ROT_COLORS = {
    'SLUH': '#B3D9FF', 'VA': '#FFCC99', 'NF': '#D9B3FF', 'MICU': '#FFB3B3',
    'Cards': '#FFFFB3', 'OP': '#B3FFB3', 'Clinic': '#FFFFFF', 'Jeopardy': '#FFE0B2',
    'ID': '#C8E6C9', 'Bronze': '#FFE0B2', 'Diamond': '#E1BEE7', 'Gold': '#FFF9C4',
    'IP Other 1': '#B0BEC5', 'IP Other 2': '#90A4AE',
}

IP_SINGLE_WEEK = {'MICU', 'Cards', 'Bronze', 'Diamond', 'Gold', 'ID',
                  'IP Other 1', 'IP Other 2', 'Jeopardy'}
MARIOKART_ROTS = {'SLUH', 'VA'}
NF_ROT = 'NF'
OP_ROTS = {'OP', 'Clinic'}

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


# ═══════════════════════════════════════════════════════════════════════
# EXCEL IMPORTER
# ═══════════════════════════════════════════════════════════════════════
@st.cache_data
def parse_schedule_excel(file_bytes):
    """Parse Schedule Explorer Excel export into structured data."""
    xls = pd.ExcelFile(io.BytesIO(file_bytes))
    result = {}
    for sheet_name in xls.sheet_names:
        if 'Schedule' in sheet_name and 'Parameter' not in sheet_name:
            df = pd.read_excel(xls, sheet_name)
            level = 'Senior' if 'Senior' in sheet_name else 'Intern'
            residents = []
            for _, row in df.iterrows():
                rid = str(row.iloc[0])
                weeks = [str(row.iloc[w + 1]) if pd.notna(row.iloc[w + 1]) else 'OP'
                         for w in range(48)]
                residents.append({'id': rid, 'schedule': weeks})
            result[level] = residents
    return result


# ═══════════════════════════════════════════════════════════════════════
# BLOCK DETECTION
# ═══════════════════════════════════════════════════════════════════════
def detect_blocks(residents):
    """Find contiguous blocks of same rotation for each resident.
    Returns dict: resident_id -> list of {rot, start_week, end_week, weeks}
    """
    blocks = {}
    for res in residents:
        rid = res['id']
        res_blocks = []
        sched = res['schedule']
        i = 0
        while i < 48:
            rot = sched[i]
            j = i
            while j < 48 and sched[j] == rot:
                j += 1
            res_blocks.append({
                'rot': rot,
                'start_week': i,
                'end_week': j - 1,
                'n_weeks': j - i,
            })
            i = j
        blocks[rid] = res_blocks
    return blocks


# ═══════════════════════════════════════════════════════════════════════
# TEAM FORMATION
# ═══════════════════════════════════════════════════════════════════════
def form_ward_teams(residents, week, rotation, team_names, team_size_target,
                    level='Senior'):
    """Group residents on a given rotation in a given week into named teams.
    Returns list of team dicts: {name, members: [resident_ids]}
    """
    on_rot = [r['id'] for r in residents if r['schedule'][week] == rotation]
    n_teams = len(team_names)
    teams = [{'name': team_names[i], 'members': []} for i in range(n_teams)]
    for idx, rid in enumerate(sorted(on_rot)):
        team_idx = idx % n_teams
        teams[team_idx]['members'].append(rid)
    return teams


def form_nf_teams(residents, week, level='Senior'):
    """Group NF residents. Seniors: 5 slots (4 active + 1 off).
    Interns: 4 slots (3 active + 1 off)."""
    on_nf = sorted([r['id'] for r in residents if r['schedule'][week] == 'NF'])
    if level == 'Senior':
        # 5 NF seniors: positions 0-3 are coverage, position 4 is off
        teams = []
        covers = NF_SR_COVERS + ["Off"]
        for i, rid in enumerate(on_nf):
            pos = i % 5
            teams.append({'id': rid, 'position': pos, 'cover': covers[pos] if pos < 5 else 'Off'})
        return teams
    else:
        # 4 NF interns: positions 0-2 are coverage, position 3 is off
        teams = []
        covers = NF_INT_COVERS + ["Off"]
        for i, rid in enumerate(on_nf):
            pos = i % 4
            teams.append({'id': rid, 'position': pos, 'cover': covers[pos] if pos < 4 else 'Off'})
        return teams


# ═══════════════════════════════════════════════════════════════════════
# MARIOKART DAY-OFF ROTATION
# ═══════════════════════════════════════════════════════════════════════
def generate_mariokart_schedule(n_teams, n_days, days_off_per_turn,
                                max_consec_days):
    """Generate the MarioKart day-off pattern.

    n_teams: total teams (e.g. 5 for SLUH)
    n_days: total days in block (e.g. 21 for 3-week)
    days_off_per_turn: consecutive days off per team (default 2)
    max_consec_days: max consecutive working days (default 8)

    Returns: list of length n_days, each element is list of team indices that are OFF.
    Also returns: dict team_idx -> list of booleans (True=working, False=off)
    """
    # Cycle length: each team gets days_off_per_turn days off, then next team
    cycle_len = n_teams * days_off_per_turn

    team_working = {t: [True] * n_days for t in range(n_teams)}
    daily_off = [[] for _ in range(n_days)]

    for day in range(n_days):
        # Which team is off?
        pos_in_cycle = day % cycle_len
        off_team = pos_in_cycle // days_off_per_turn
        off_team = off_team % n_teams
        team_working[off_team][day] = False
        daily_off[day].append(off_team)

    # Verify max consecutive constraint
    for t in range(n_teams):
        consec = 0
        max_c = 0
        for day in range(n_days):
            if team_working[t][day]:
                consec += 1
                max_c = max(max_c, consec)
            else:
                consec = 0
        # If exceeding max, insert extra off days
        if max_c > max_consec_days:
            consec = 0
            for day in range(n_days):
                if team_working[t][day]:
                    consec += 1
                    if consec >= max_consec_days:
                        # Force day off next day if possible
                        if day + 1 < n_days and team_working[t][day + 1]:
                            team_working[t][day + 1] = False
                            daily_off[day + 1].append(t)
                            consec = 0
                else:
                    consec = 0

    return team_working, daily_off


def generate_nf_rotation(n_residents, n_days, days_off_per_turn,
                          covers, max_consec_nights):
    """Generate NF rotation: n_residents rotate through coverage areas + off.
    Returns: day -> list of {resident_idx, assignment} dicts
    """
    n_active = len(covers)
    cycle_len = n_residents * days_off_per_turn

    schedule = []
    for day in range(n_days):
        pos_in_cycle = day % cycle_len
        off_idx = (pos_in_cycle // days_off_per_turn) % n_residents
        assignments = []
        active_slot = 0
        for r in range(n_residents):
            if r == off_idx:
                assignments.append({'resident_idx': r, 'assignment': 'Off'})
            else:
                cover = covers[active_slot % len(covers)]
                assignments.append({'resident_idx': r, 'assignment': cover})
                active_slot += 1
        schedule.append(assignments)
    return schedule


# ═══════════════════════════════════════════════════════════════════════
# FLOOR/COVERAGE ASSIGNMENT FOR WARD TEAMS
# ═══════════════════════════════════════════════════════════════════════
def assign_floors(team_working, floors, n_days):
    """Assign active teams to floors each day.
    team_working: dict team_idx -> list[bool]
    floors: list of floor names (len = n_active_teams)
    Returns: day -> dict {team_idx: floor_name or 'Off'}
    """
    n_teams = len(team_working)
    daily_assignments = []
    for day in range(n_days):
        active = [t for t in range(n_teams) if team_working[t][day]]
        assignment = {}
        for t in range(n_teams):
            if not team_working[t][day]:
                assignment[t] = 'Off'
            else:
                # Rotate floor assignments: each active team gets a floor
                slot = active.index(t)
                # Shift floor assignments based on day to rotate coverage
                floor_idx = (slot + day // 2) % len(floors)
                assignment[t] = floors[floor_idx]
        daily_assignments.append(assignment)
    return daily_assignments


# ═══════════════════════════════════════════════════════════════════════
# MASTER DAILY SCHEDULE BUILDER
# ═══════════════════════════════════════════════════════════════════════
def build_daily_schedule(residents, level, params):
    """Build complete daily schedule from block schedule.

    Returns dict with:
    - ward_schedules: {rotation: {block_key: {teams, team_working, floor_assignments}}}
    - nf_schedules: {block_key: {residents, daily_assignments}}
    - ip_schedules: {resident_id: [{rot, days, working}]}
    - op_schedules: {resident_id: [{rot, days, working}]}
    - resident_daily: {resident_id: list of 336 day-entries}
    """
    blocks = detect_blocks(residents)
    days_off = params.get('days_off_per_turn', 2)
    max_consec = params.get('max_consec_days', 8)

    # Master daily schedule: resident_id -> [day0, day1, ..., day335]
    # Each day entry: {rot, assignment, working, team, floor}
    n_total_days = 48 * 7  # 336 days
    resident_daily = {}
    for res in residents:
        resident_daily[res['id']] = [
            {'rot': '', 'assignment': '', 'working': True, 'team': '', 'floor': ''}
            for _ in range(n_total_days)
        ]

    # Track ward block schedules for team rotation display
    ward_block_schedules = {}  # key: (rotation, start_week) -> schedule info
    nf_block_schedules = {}

    # Process each resident's blocks
    for res in residents:
        rid = res['id']
        for block in blocks[rid]:
            rot = block['rot']
            sw = block['start_week']
            ew = block['end_week']
            n_weeks = block['n_weeks']
            n_days = n_weeks * 7
            day_start = sw * 7

            if rot in OP_ROTS:
                # OP/Clinic: Mon-Fri working, Sat-Sun off
                for d in range(n_days):
                    abs_day = day_start + d
                    if abs_day >= n_total_days:
                        break
                    dow = d % 7  # 0=Mon ... 6=Sun
                    working = dow < 5
                    resident_daily[rid][abs_day] = {
                        'rot': rot, 'assignment': rot,
                        'working': working, 'team': '', 'floor': ''
                    }

            elif rot == 'Jeopardy':
                # Jeopardy: available all 7 days
                for d in range(n_days):
                    abs_day = day_start + d
                    if abs_day >= n_total_days:
                        break
                    resident_daily[rid][abs_day] = {
                        'rot': 'Jeopardy', 'assignment': 'Jeopardy',
                        'working': True, 'team': '', 'floor': ''
                    }

            elif rot in IP_SINGLE_WEEK:
                # Single-week IP: work all 7 days (within consec limit)
                for d in range(n_days):
                    abs_day = day_start + d
                    if abs_day >= n_total_days:
                        break
                    resident_daily[rid][abs_day] = {
                        'rot': rot, 'assignment': rot,
                        'working': True, 'team': '', 'floor': ''
                    }

            # MarioKart ward rotations and NF are handled at block level below

    # Now handle ward and NF blocks collectively (need team formation)
    # Group residents by rotation and contiguous block start
    ward_groups = defaultdict(list)  # (rotation, start_week, end_week) -> [resident_ids]
    nf_groups = defaultdict(list)

    for res in residents:
        rid = res['id']
        for block in blocks[rid]:
            rot = block['rot']
            sw = block['start_week']
            ew = block['end_week']
            if rot in MARIOKART_ROTS:
                key = (rot, sw, ew)
                ward_groups[key].append(rid)
            elif rot == NF_ROT:
                key = (rot, sw, ew)
                nf_groups[key].append(rid)

    # Process ward MarioKart blocks
    for (rot, sw, ew), rids in ward_groups.items():
        n_weeks = ew - sw + 1
        n_days = n_weeks * 7
        day_start = sw * 7

        team_names = SLUH_TEAMS if rot == 'SLUH' else VA_TEAMS
        floors = SLUH_FLOORS if rot == 'SLUH' else VA_FLOORS
        n_teams = len(team_names)

        # Assign residents to teams
        sorted_rids = sorted(rids)
        teams = {i: [] for i in range(n_teams)}
        for idx, rid in enumerate(sorted_rids):
            teams[idx % n_teams].append(rid)

        # Generate MarioKart off-day pattern
        team_working, daily_off = generate_mariokart_schedule(
            n_teams, n_days, days_off, max_consec
        )

        # Assign floors
        floor_assign = assign_floors(team_working, floors, n_days)

        # Store for display
        ward_block_schedules[(rot, sw)] = {
            'rotation': rot, 'start_week': sw, 'end_week': ew,
            'n_days': n_days, 'team_names': team_names, 'floors': floors,
            'teams': teams, 'team_working': team_working,
            'floor_assignments': floor_assign, 'rids': sorted_rids,
        }

        # Write to resident daily schedule
        for team_idx in range(n_teams):
            for rid in teams[team_idx]:
                for d in range(n_days):
                    abs_day = day_start + d
                    if abs_day >= n_total_days:
                        break
                    working = team_working[team_idx][d]
                    floor = floor_assign[d].get(team_idx, '')
                    resident_daily[rid][abs_day] = {
                        'rot': rot,
                        'assignment': floor if working else 'Off',
                        'working': working,
                        'team': team_names[team_idx],
                        'floor': floor if working else '',
                    }

    # Process NF blocks
    for (rot, sw, ew), rids in nf_groups.items():
        n_weeks = ew - sw + 1
        n_days = n_weeks * 7
        day_start = sw * 7
        sorted_rids = sorted(rids)
        n_res = len(sorted_rids)

        if level == 'Senior':
            covers = NF_SR_COVERS
        else:
            covers = NF_INT_COVERS

        nf_sched = generate_nf_rotation(n_res, n_days, days_off, covers, max_consec)

        nf_block_schedules[(rot, sw)] = {
            'rotation': rot, 'start_week': sw, 'end_week': ew,
            'n_days': n_days, 'rids': sorted_rids,
            'daily': nf_sched, 'covers': covers,
        }

        for d in range(n_days):
            abs_day = day_start + d
            if abs_day >= n_total_days:
                break
            for entry in nf_sched[d]:
                ridx = entry['resident_idx']
                if ridx < len(sorted_rids):
                    rid = sorted_rids[ridx]
                    working = entry['assignment'] != 'Off'
                    resident_daily[rid][abs_day] = {
                        'rot': 'NF',
                        'assignment': entry['assignment'],
                        'working': working,
                        'team': '',
                        'floor': entry['assignment'] if working else '',
                    }

    # Cross-block boundary check: if two adjacent IP blocks, check consec days
    for res in residents:
        rid = res['id']
        consec = 0
        for d in range(n_total_days):
            entry = resident_daily[rid][d]
            if entry['working'] and entry['rot'] not in OP_ROTS:
                consec += 1
                if consec > max_consec and entry['rot'] != 'Jeopardy':
                    # Insert buffer day off
                    resident_daily[rid][d] = {
                        'rot': entry['rot'], 'assignment': 'Buffer Off',
                        'working': False, 'team': entry.get('team', ''),
                        'floor': ''
                    }
                    consec = 0
            else:
                consec = 0

    return {
        'resident_daily': resident_daily,
        'ward_blocks': ward_block_schedules,
        'nf_blocks': nf_block_schedules,
        'blocks': blocks,
    }


# ═══════════════════════════════════════════════════════════════════════
# COVERAGE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════
def compute_daily_coverage(daily_data, residents):
    """Compute daily coverage: how many residents working per rotation per day."""
    n_days = 48 * 7
    coverage = defaultdict(lambda: [0] * n_days)
    for res in residents:
        rid = res['id']
        for d in range(n_days):
            entry = daily_data['resident_daily'][rid][d]
            if entry['working'] and entry['rot']:
                coverage[entry['rot']][d] += 1
    return dict(coverage)


def compute_resident_stats(daily_data, residents):
    """Per-resident stats: total days worked, days off, max consecutive, weekends."""
    n_days = 48 * 7
    stats = {}
    for res in residents:
        rid = res['id']
        days_on = 0
        days_off = 0
        max_consec = 0
        consec = 0
        weekends_on = 0
        for d in range(n_days):
            entry = daily_data['resident_daily'][rid][d]
            dow = d % 7
            if entry['working'] and entry['rot']:
                days_on += 1
                consec += 1
                max_consec = max(max_consec, consec)
                if dow >= 5:  # Sat or Sun
                    weekends_on += 1
            else:
                if entry['rot']:
                    days_off += 1
                consec = 0
        stats[rid] = {
            'days_on': days_on, 'days_off': days_off,
            'max_consec': max_consec, 'weekends_on': weekends_on,
        }
    return stats


# ═══════════════════════════════════════════════════════════════════════
# HTML RENDERING HELPERS
# ═══════════════════════════════════════════════════════════════════════
def render_calendar_html(daily_data, residents, week_start, week_end,
                         filter_rot=None, max_consec_limit=8):
    """Render a calendar view as HTML table for given week range."""
    day_start = week_start * 7
    day_end = (week_end + 1) * 7
    n_days = day_end - day_start

    # Filter residents
    show_residents = residents
    if filter_rot:
        show_residents = [r for r in residents
                          if any(r['schedule'][w] == filter_rot
                                 for w in range(week_start, week_end + 1))]

    html = '<div class="cal-grid"><table>'

    # Header row: week numbers and day names
    html += '<tr><th rowspan="2" style="min-width:60px">Resident</th>'
    for w in range(week_start, week_end + 1):
        html += f'<th colspan="7" style="border-left:2px solid #1a3a6b">Wk {w+1}</th>'
    html += '</tr><tr>'
    for w in range(week_start, week_end + 1):
        for dn in DAY_NAMES:
            html += f'<th>{dn}</th>'
    html += '</tr>'

    # Resident rows
    for res in show_residents:
        rid = res['id']
        html += f'<tr><td style="text-align:left;font-weight:600;background:#f8f9fa">{rid}</td>'
        consec = 0
        for d in range(day_start, day_end):
            entry = daily_data['resident_daily'].get(rid, [{}] * 336)[d] if d < 336 else {}
            rot = entry.get('rot', '')
            working = entry.get('working', False)
            assignment = entry.get('assignment', '')
            team = entry.get('team', '')

            # Count consecutive for violation detection
            if working and rot not in OP_ROTS:
                consec += 1
            else:
                consec = 0
            violation = consec > max_consec_limit

            # Cell styling
            bg = '#F0F0F0'
            txt_color = '#999'
            cls = 'day-off'
            if working and rot:
                bg = ROT_COLORS.get(rot, '#FFFFFF')
                txt_color = '#333'
                cls = 'day-work'
            if violation:
                cls += ' day-violation'

            # Cell label
            if not working and rot:
                label = 'OFF'
            elif assignment:
                label = assignment[:4]
            elif rot:
                label = rot[:4]
            else:
                label = ''

            # Week border
            border = ''
            if (d - day_start) % 7 == 0:
                border = 'border-left:2px solid #999;'

            html += (f'<td class="{cls}" style="background:{bg};color:{txt_color};'
                     f'{border}" title="{rid} Day {d+1}: {rot} - {assignment}'
                     f' (Team {team})">{label}</td>')
        html += '</tr>'

    html += '</table></div>'
    return html


def render_team_rotation_html(ward_block, week_offset=0):
    """Render MarioKart team rotation for a single ward block."""
    teams = ward_block['teams']
    team_names = ward_block['team_names']
    tw = ward_block['team_working']
    fa = ward_block['floor_assignments']
    n_days = ward_block['n_days']
    rot = ward_block['rotation']

    html = f'<h4>{rot} — Weeks {ward_block["start_week"]+1}–{ward_block["end_week"]+1}</h4>'
    html += '<div class="cal-grid"><table>'

    # Header
    n_weeks = n_days // 7
    html += '<tr><th>Team</th><th>Members</th>'
    for w in range(n_weeks):
        html += f'<th colspan="7" style="border-left:2px solid #1a3a6b">Wk {ward_block["start_week"]+w+1}</th>'
    html += '</tr><tr><th></th><th></th>'
    for w in range(n_weeks):
        for dn in DAY_NAMES:
            html += f'<th>{dn}</th>'
    html += '</tr>'

    # Team rows
    for tidx in range(len(team_names)):
        tname = team_names[tidx]
        members = ', '.join(teams.get(tidx, []))
        bg = TEAM_COLORS.get(tname, '#FFFFFF')
        html += (f'<tr><td style="background:{bg};font-weight:700;text-align:left">'
                 f'{tname}</td>')
        html += f'<td style="text-align:left;font-size:8px">{members}</td>'
        for d in range(n_days):
            working = tw[tidx][d]
            floor = fa[d].get(tidx, '')
            border = 'border-left:2px solid #999;' if d % 7 == 0 else ''
            if working:
                html += (f'<td style="background:{bg};font-weight:600;{border}"'
                         f' title="Day {d+1}: {floor}">{floor}</td>')
            else:
                html += (f'<td class="day-off" style="{border}"'
                         f' title="Day {d+1}: OFF">OFF</td>')
        html += '</tr>'

    html += '</table></div>'
    return html


def render_nf_rotation_html(nf_block):
    """Render NF rotation for a single NF block."""
    rids = nf_block['rids']
    daily = nf_block['daily']
    n_days = nf_block['n_days']
    covers = nf_block['covers']

    html = f'<h4>Night Float — Weeks {nf_block["start_week"]+1}–{nf_block["end_week"]+1}</h4>'
    html += '<div class="cal-grid"><table>'

    n_weeks = n_days // 7
    html += '<tr><th>Resident</th>'
    for w in range(n_weeks):
        html += f'<th colspan="7" style="border-left:2px solid #1a3a6b">Wk {nf_block["start_week"]+w+1}</th>'
    html += '</tr><tr><th></th>'
    for w in range(n_weeks):
        for dn in DAY_NAMES:
            html += f'<th>{dn}</th>'
    html += '</tr>'

    for ridx, rid in enumerate(rids):
        html += f'<tr><td style="text-align:left;font-weight:600;background:#f8f9fa">{rid}</td>'
        for d in range(n_days):
            if d < len(daily):
                entry = next((e for e in daily[d] if e['resident_idx'] == ridx), None)
                if entry:
                    assign = entry['assignment']
                    is_off = assign == 'Off'
                    border = 'border-left:2px solid #999;' if d % 7 == 0 else ''
                    if is_off:
                        html += f'<td class="day-off" style="{border}">OFF</td>'
                    else:
                        bg = '#D9B3FF'
                        html += (f'<td style="background:{bg};font-weight:600;{border}"'
                                 f' title="{assign}">{assign[:4]}</td>')
                else:
                    html += '<td></td>'
            else:
                html += '<td></td>'
        html += '</tr>'

    html += '</table></div>'
    return html


def render_coverage_heatmap_html(coverage, week_start, week_end, rotation):
    """Render daily coverage heatmap for a rotation."""
    day_start = week_start * 7
    day_end = (week_end + 1) * 7
    days = coverage.get(rotation, [0] * 336)

    html = f'<h4>{rotation} Daily Coverage — Weeks {week_start+1}–{week_end+1}</h4>'
    html += '<div class="cal-grid"><table>'

    n_weeks = week_end - week_start + 1
    html += '<tr><th></th>'
    for w in range(week_start, week_end + 1):
        html += f'<th colspan="7" style="border-left:2px solid #1a3a6b">Wk {w+1}</th>'
    html += '</tr><tr><th>Coverage</th>'
    for w in range(n_weeks):
        for dn in DAY_NAMES:
            html += f'<th>{dn}</th>'
    html += '</tr><tr><td style="font-weight:600">Residents On</td>'

    for d in range(day_start, day_end):
        val = days[d] if d < len(days) else 0
        # Color intensity based on coverage level
        if val == 0:
            bg = '#FFCCCC'
        elif val <= 2:
            bg = '#FFE0B2'
        elif val <= 4:
            bg = '#FFFFCC'
        else:
            bg = '#CCFFCC'
        border = 'border-left:2px solid #999;' if (d - day_start) % 7 == 0 else ''
        html += f'<td style="background:{bg};font-weight:600;{border}">{val}</td>'

    html += '</tr></table></div>'
    return html


# ═══════════════════════════════════════════════════════════════════════
# EXCEL EXPORT
# ═══════════════════════════════════════════════════════════════════════
def export_daily_to_excel(daily_data, residents, level):
    """Export daily schedule to Excel."""
    n_days = 48 * 7
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        # Daily schedule sheet
        rows = []
        for res in residents:
            rid = res['id']
            row = {'Resident': rid}
            for d in range(n_days):
                entry = daily_data['resident_daily'][rid][d]
                w = d // 7 + 1
                dow = DAY_NAMES[d % 7]
                col = f'W{w}_{dow}'
                if entry['working']:
                    row[col] = entry.get('assignment', entry.get('rot', ''))
                else:
                    row[col] = 'OFF'
            rows.append(row)
        df = pd.DataFrame(rows)
        df.to_excel(writer, sheet_name=f'{level} Daily', index=False)

        # Team assignments sheet
        team_rows = []
        for res in residents:
            rid = res['id']
            row = {'Resident': rid}
            for d in range(n_days):
                entry = daily_data['resident_daily'][rid][d]
                w = d // 7 + 1
                dow = DAY_NAMES[d % 7]
                col = f'W{w}_{dow}'
                team = entry.get('team', '')
                row[col] = team if team else entry.get('rot', '')
            team_rows.append(row)
        pd.DataFrame(team_rows).to_excel(writer, sheet_name='Teams', index=False)

    buf.seek(0)
    return buf


# ═══════════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════════
st.title("📅 Project Z — Daily Shift Scheduler")
st.caption("Import a block schedule from Schedule Explorer, then generate day-by-day "
           "team assignments with the MarioKart rotation system.")

# ─── Sidebar ───
with st.sidebar:
    st.header("📁 Import Schedule")
    st.caption("Upload the Excel file exported from the Schedule Explorer app. "
               "The file should contain 'Senior Schedule' and/or 'Intern Schedule' sheets.")
    uploaded = st.file_uploader("Upload Schedule Explorer Excel",
                                type=['xlsx'], key='schedule_upload')

    if uploaded:
        data = parse_schedule_excel(uploaded.getvalue())
        available_levels = list(data.keys())
        if not available_levels:
            st.error("No valid schedule sheets found in the uploaded file.")
            st.stop()
        level = st.radio("Level", available_levels, key='level_select')
        residents = data[level]
        st.success(f"Loaded {len(residents)} {level.lower()} residents")

        st.divider()
        st.header("⚙️ MarioKart Parameters")
        st.caption("Control the day-off rotation pattern. In the MarioKart system, "
                   "5 teams rotate: 4 work while 1 is off for a set number of days, "
                   "then the next team rotates off.")

        days_off = st.number_input("Days off per turn", min_value=1, max_value=4,
                                   value=2, key='days_off',
                                   help="Consecutive days off each team gets per rotation turn")
        max_consec = st.number_input("Max consecutive working days", min_value=5,
                                     max_value=14, value=8, key='max_consec',
                                     help="Maximum days a resident can work in a row")
        cross_block_buffer = st.checkbox("Cross-block buffer day", value=True,
                                         key='cross_buffer',
                                         help="Insert a buffer day off when transitioning between IP blocks")

        st.divider()
        st.header("📆 View Settings")
        week_range = st.slider("Week range", 1, 48, (1, 6), key='week_range')
        week_start = week_range[0] - 1
        week_end = week_range[1] - 1

        rot_filter = st.selectbox("Filter by rotation", ["All"] + sorted(set(
            r['schedule'][w] for r in residents for w in range(48)
            if r['schedule'][w] not in ('', 'nan')
        )), key='rot_filter')
    else:
        st.info("👆 Upload a schedule export to get started")
        st.stop()

# ─── Build daily schedule ───
mk_params = {
    'days_off_per_turn': days_off,
    'max_consec_days': max_consec,
    'cross_block_buffer': cross_block_buffer,
}
daily_data = build_daily_schedule(residents, level, mk_params)
coverage = compute_daily_coverage(daily_data, residents)
res_stats = compute_resident_stats(daily_data, residents)

# ─── KPI Cards ───
total_res = len(residents)
avg_days_on = np.mean([s['days_on'] for s in res_stats.values()])
avg_days_off = np.mean([s['days_off'] for s in res_stats.values()])
max_consec_actual = max(s['max_consec'] for s in res_stats.values())
violations = sum(1 for s in res_stats.values() if s['max_consec'] > max_consec)
avg_weekends = np.mean([s['weekends_on'] for s in res_stats.values()])

cols = st.columns(6)
kpis = [
    (f"{total_res}", "Residents"),
    (f"{avg_days_on:.0f}", "Avg Days On"),
    (f"{avg_days_off:.0f}", "Avg Days Off"),
    (f"{max_consec_actual}", "Max Consecutive"),
    (f"{violations}", "Consec Violations"),
    (f"{avg_weekends:.0f}", "Avg Weekend Days"),
]
for col, (val, label) in zip(cols, kpis):
    col.markdown(f'<div class="kpi-card"><div class="kpi-val">{val}</div>'
                 f'<div class="kpi-label">{label}</div></div>', unsafe_allow_html=True)

# ─── Tabs ───
tab1, tab2, tab3, tab4 = st.tabs([
    "📅 Calendar View", "🎮 Team Rotation", "📊 Coverage Dashboard",
    "✅ Compliance & Export"
])

# ── Tab 1: Calendar View ──
with tab1:
    st.caption("Day-by-day schedule for each resident. Colored by rotation, grey = off. "
               "Red border = consecutive-day violation. Hover over a cell for details.")
    filter_r = rot_filter if rot_filter != 'All' else None
    html = render_calendar_html(daily_data, residents, week_start, week_end,
                                filter_rot=filter_r, max_consec_limit=max_consec)
    st.markdown(html, unsafe_allow_html=True)

# ── Tab 2: Team Rotation ──
with tab2:
    st.caption("Visual display of the MarioKart team rotation. Shows which team is on "
               "which floor/coverage area each day, and who is off.")

    # Filter ward blocks in view range
    visible_ward = {k: v for k, v in daily_data['ward_blocks'].items()
                    if v['start_week'] <= week_end and v['end_week'] >= week_start}
    visible_nf = {k: v for k, v in daily_data['nf_blocks'].items()
                  if v['start_week'] <= week_end and v['end_week'] >= week_start}

    if not visible_ward and not visible_nf:
        st.info("No MarioKart or NF blocks in the selected week range.")
    else:
        for key in sorted(visible_ward.keys()):
            st.markdown(render_team_rotation_html(visible_ward[key]),
                        unsafe_allow_html=True)
            st.divider()

        for key in sorted(visible_nf.keys()):
            st.markdown(render_nf_rotation_html(visible_nf[key]),
                        unsafe_allow_html=True)
            st.divider()

# ── Tab 3: Coverage Dashboard ──
with tab3:
    st.caption("Daily coverage counts per rotation. Green = well-staffed, "
               "yellow = borderline, red = gap.")
    active_rots = sorted(set(
        r['schedule'][w] for r in residents for w in range(week_start, week_end + 1)
        if r['schedule'][w] not in ('', 'nan')
    ))
    for rot in active_rots:
        if rot in coverage:
            st.markdown(render_coverage_heatmap_html(coverage, week_start, week_end, rot),
                        unsafe_allow_html=True)

# ── Tab 4: Compliance & Export ──
with tab4:
    st.caption("Per-resident compliance stats and Excel export.")

    # Stats table
    stat_rows = []
    for res in residents:
        rid = res['id']
        s = res_stats[rid]
        stat_rows.append({
            'Resident': rid,
            'Days On': s['days_on'],
            'Days Off': s['days_off'],
            'Max Consecutive': s['max_consec'],
            'Weekend Days On': s['weekends_on'],
            'Violation': '⚠️' if s['max_consec'] > max_consec else '✓',
        })
    stat_df = pd.DataFrame(stat_rows)
    st.dataframe(stat_df, use_container_width=True, height=400)

    # Fairness summary
    st.subheader("Fairness Summary")
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        days_on_vals = [s['days_on'] for s in res_stats.values()]
        st.metric("Days On Range", f"{min(days_on_vals)}–{max(days_on_vals)}")
    with fc2:
        days_off_vals = [s['days_off'] for s in res_stats.values()]
        st.metric("Days Off Range", f"{min(days_off_vals)}–{max(days_off_vals)}")
    with fc3:
        weekend_vals = [s['weekends_on'] for s in res_stats.values()]
        st.metric("Weekend Days Range", f"{min(weekend_vals)}–{max(weekend_vals)}")

    # Export
    st.divider()
    st.subheader("📥 Export to Excel")
    if st.button("Generate Excel Export", key='export_btn'):
        buf = export_daily_to_excel(daily_data, residents, level)
        st.download_button(
            "⬇️ Download Daily Schedule Excel",
            data=buf,
            file_name=f"daily_schedule_{level.lower()}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
