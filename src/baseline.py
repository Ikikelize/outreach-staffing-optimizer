"""A simple rule that imitates how rosters are often made by hand.

Each day, the coordinator fills the biggest sites first (urban, then
peri-urban, then rural). For each open shift they pick the nearest available
worker with the right skill who still has shifts left this week.
"""

from .data import DAYS

TYPE_ORDER = {"urban": 0, "peri-urban": 1, "rural": 2}


def greedy_nearest(inst):
    W = inst.workers.set_index("id")
    site_type = inst.sites.set_index("id")["type"].to_dict()
    shifts_left = W["max_shifts"].to_dict()
    rows = []

    for d in DAYS:
        busy_today = set()
        today = inst.demand[inst.demand["day"] == d].copy()
        today["order"] = today["site"].map(lambda s: TYPE_ORDER[site_type[s]])
        today = today.sort_values(["order", "site", "skill"])
        for r in today.itertuples():
            for _ in range(int(r.need)):
                options = [w for w, wr in W.iterrows()
                           if wr.skill == r.skill and wr.day_off != d
                           and w not in busy_today and shifts_left[w] > 0]
                if not options:
                    break
                best = min(options, key=lambda w: inst.dist_km.loc[w, r.site])
                rows.append({"worker": best, "site": r.site, "day": d, "shift": r.skill})
                busy_today.add(best)
                shifts_left[best] -= 1
    return rows
