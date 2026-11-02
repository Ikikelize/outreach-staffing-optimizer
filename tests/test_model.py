"""Checks that every roster is valid and that the optimizer behaves as described.

Run with:  python -m pytest -q
"""

import pulp

from src.baseline import greedy_nearest
from src.data import make_instance
from src.evaluate import check_feasible, score
from src.model import solve

INST = make_instance()


def km(rows):
    return score(INST, rows, "")[0]["total_travel_km"]


def max_shifts_possible(inst):
    """Independent check: the most shifts any roster could fill, ignoring travel."""
    W = inst.workers.set_index("id")
    need = {(r.site, r.skill, r.day): int(r.need) for r in inst.demand.itertuples()}
    keys = [(w, s, d) for w, wr in W.iterrows() for (s, k, d) in need
            if k == wr.skill and d != wr.day_off]
    prob = pulp.LpProblem("max_fill", pulp.LpMaximize)
    x = {k: prob.add_variable("x_" + "_".join(k), cat="Binary") for k in keys}
    prob += pulp.lpSum(x.values())
    for w, wr in W.iterrows():
        prob += pulp.lpSum(x[k] for k in keys if k[0] == w) <= int(wr.max_shifts)
        for d in {k[2] for k in keys if k[0] == w}:
            prob += pulp.lpSum(x[k] for k in keys if k[0] == w and k[2] == d) <= 1
    for (s, sk, d), n in need.items():
        prob += pulp.lpSum(x[k] for k in keys
                           if k[1] == s and k[2] == d and W.loc[k[0], "skill"] == sk) <= n
    prob.solve(pulp.HiGHS(msg=False))
    return round(pulp.value(prob.objective))


def test_demand_matches_the_stated_setup():
    d = INST.demand
    assert d["need"].sum() == 133
    rural = INST.sites.loc[INST.sites["type"] == "rural", "id"]
    assert set(d.loc[d["site"].isin(rural), "day"]) == {"Mon", "Wed", "Fri"}


def test_optimizer_fills_the_maximum_possible():
    rows, _ = solve(INST)
    assert len(rows) == max_shifts_possible(INST) == 118


def test_baseline_is_valid():
    assert check_feasible(INST, greedy_nearest(INST))


def test_optimizer_is_valid_and_beats_baseline():
    rows, status = solve(INST)
    assert status == "Optimal"
    assert check_feasible(INST, rows)
    assert km(rows) <= km(greedy_nearest(INST))


def test_equity_floor_is_respected():
    rows, _ = solve(INST, min_fill=0.8)
    assert score(INST, rows, "")[0]["lowest_site_fill_pct"] >= 80


def test_worker_cap_is_respected():
    rows, _ = solve(INST, min_fill=0.8, max_worker_km=280)
    assert score(INST, rows, "")[0]["max_worker_km"] <= 280


def test_impossible_floor_is_reported():
    rows, status = solve(INST, min_fill=0.84)
    assert rows is None and status == "Infeasible"


def test_tiny_case_by_hand():
    """Two workers, one site, one shift: the closer worker must be chosen."""
    import pandas as pd
    from src.data import Instance
    workers = pd.DataFrame([
        {"id": "A", "skill": "clinical", "x": 0, "y": 0, "day_off": "Sun", "max_shifts": 5},
        {"id": "B", "skill": "clinical", "x": 9, "y": 0, "day_off": "Sun", "max_shifts": 5}])
    sites = pd.DataFrame([{"id": "S", "type": "urban", "x": 10, "y": 0}])
    demand = pd.DataFrame([{"site": "S", "skill": "clinical", "day": "Mon", "need": 1}])
    dist = pd.DataFrame([[10.0], [1.0]], index=["A", "B"], columns=["S"])
    rows, _ = solve(Instance(workers, sites, demand, dist))
    assert rows == [{"worker": "B", "site": "S", "day": "Mon"}]


def test_results_hold_on_other_regions():
    """The optimizer never fills fewer shifts or travels more than the manual rule."""
    for seed in (1, 2, 3):
        inst = make_instance(seed=seed)
        manual = greedy_nearest(inst)
        best, status = solve(inst)
        assert status == "Optimal" and check_feasible(inst, best)
        assert len(best) >= len(manual)
        if len(best) == len(manual):
            assert score(inst, best, "")[0]["total_travel_km"] <= \
                   score(inst, manual, "")[0]["total_travel_km"]
