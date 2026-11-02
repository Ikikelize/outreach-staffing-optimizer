"""Integer programming model for weekly staff-to-site assignment (PuLP).

Decision: x[w, s, d] = 1 if worker w works at site s on day d.

Solved in two steps (a lexicographic objective):
  1. fill as many shifts as possible;
  2. holding that number fixed, minimize total round-trip travel km.

Both steps use the same rules:
  - a worker works at most one site per day, never on their day off
  - a worker works at most `max_shifts` days in the week
  - workers only fill shifts that match their skill
  - a site never gets more staff than it asked for
  - optional site floor: every site gets at least `min_fill` of its weekly shifts
  - optional travel cap: no worker travels more than `max_worker_km` in the week
"""

import pulp


def solve(inst, min_fill: float = 0.0, max_worker_km: float | None = None,
          time_limit: int = 120, verbose: bool = False):
    W = inst.workers.set_index("id")
    need = {(r.site, r.skill, r.day): int(r.need) for r in inst.demand.itertuples()}

    # Only create variables that could be used: right skill, demand exists,
    # worker is not on their day off.
    keys = [(w, s, d)
            for w, wr in W.iterrows()
            for (s, k, d) in need
            if k == wr.skill and d != wr.day_off]

    prob = pulp.LpProblem("outreach_staffing", pulp.LpMinimize)
    # PuLP 4 creates variables through the problem object.
    x = {k: prob.add_variable("x_" + "_".join(k), cat="Binary") for k in keys}
    short = {k: prob.add_variable("short_" + "_".join(k), lowBound=0) for k in need}

    travel = pulp.lpSum(inst.round_trip(w, s) * x[(w, s, d)] for (w, s, d) in keys)
    unfilled = pulp.lpSum(short.values())

    by_wd, by_w, by_skd, by_ws = {}, {}, {}, {}
    for (w, s, d) in keys:
        by_wd.setdefault((w, d), []).append(x[(w, s, d)])
        by_w.setdefault(w, []).append(x[(w, s, d)])
        by_skd.setdefault((s, W.loc[w, "skill"], d), []).append(x[(w, s, d)])
        by_ws.setdefault(w, []).append(inst.round_trip(w, s) * x[(w, s, d)])

    for (w, d), xs in by_wd.items():
        prob += pulp.lpSum(xs) <= 1, f"one_site_{w}_{d}"
    for w, xs in by_w.items():
        prob += pulp.lpSum(xs) <= int(W.loc[w, "max_shifts"]), f"max_shifts_{w}"
    for key, n in need.items():
        assigned = pulp.lpSum(by_skd.get(key, []))
        prob += assigned + short[key] == n, f"cover_{'_'.join(key)}"

    if min_fill > 0:
        for s in inst.sites["id"]:
            site_keys = [k for k in need if k[0] == s]
            total = sum(need[k] for k in site_keys)
            filled = total - pulp.lpSum(short[k] for k in site_keys)
            prob += filled >= min_fill * total, f"equity_{s}"

    if max_worker_km is not None:
        for w, terms in by_ws.items():
            prob += pulp.lpSum(terms) <= max_worker_km, f"fair_km_{w}"

    solver = pulp.HiGHS(msg=verbose, timeLimit=time_limit)

    # Step 1: the fewest unfilled shifts any valid roster can leave.
    prob.setObjective(unfilled)
    stats = prob.solve(solver)  # PuLP 4 returns an LpSolveStats object
    if stats.status != pulp.LpSolveStatus.Optimal:
        return None, stats.status.name
    fewest_unfilled = round(pulp.value(unfilled))

    # Step 2: least travel among rosters that leave no more shifts unfilled.
    prob += unfilled <= fewest_unfilled, "keep_max_fill"
    prob.setObjective(travel)
    stats = prob.solve(solver)
    status_name = stats.status.name
    if stats.status != pulp.LpSolveStatus.Optimal:
        return None, status_name

    rows = [{"worker": w, "site": s, "day": d}
            for (w, s, d) in keys if x[(w, s, d)].value() > 0.5]
    return rows, status_name
