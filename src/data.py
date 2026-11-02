"""Synthetic data for a one-week outreach staffing problem.

Everything here is invented. No real staff, sites or organizations are used.
The setup mimics a common situation in community health programs: most staff
live near the city, but some service sites are rural and far away, and there
are slightly fewer staff-days than shifts to fill.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]
SKILLS = ["clinical", "counselling"]
ROAD_FACTOR = 1.3  # straight-line distance understates road distance


@dataclass
class Instance:
    workers: pd.DataFrame          # id, skill, x, y, day_off, max_shifts
    sites: pd.DataFrame            # id, type, x, y
    demand: pd.DataFrame           # site, skill, day, need
    dist_km: pd.DataFrame          # one-way road km, workers x sites
    meta: dict = field(default_factory=dict)

    def round_trip(self, worker, site):
        return 2.0 * self.dist_km.loc[worker, site]


def make_instance(seed: int = 42,
                  n_clinical: int = 18,
                  n_counsellors: int = 12,
                  max_shifts: int = 4) -> Instance:
    rng = np.random.default_rng(seed)
    centre = np.array([50.0, 50.0])

    # Sites: urban close to the centre, peri-urban in a ring, rural far out.
    site_rows = []
    for i in range(5):
        p = centre + rng.normal(0, 6, 2)
        site_rows.append(("U", p))
    for i in range(4):
        ang = rng.uniform(0, 2 * np.pi)
        r = rng.uniform(18, 28)
        site_rows.append(("P", centre + r * np.array([np.cos(ang), np.sin(ang)])))
    rural_angles = [0.6, 2.4, 4.3]
    for ang in rural_angles:
        r = rng.uniform(38, 46)
        site_rows.append(("R", centre + r * np.array([np.cos(ang), np.sin(ang)])))

    type_name = {"U": "urban", "P": "peri-urban", "R": "rural"}
    counters = {"U": 0, "P": 0, "R": 0}
    sites = []
    for t, p in site_rows:
        counters[t] += 1
        sites.append({"id": f"{t}{counters[t]}", "type": type_name[t],
                      "x": round(float(p[0]), 1), "y": round(float(p[1]), 1)})
    sites = pd.DataFrame(sites)

    # Workers: about 75% live within ~12 km of the centre, the rest scattered.
    workers = []
    skills = ["clinical"] * n_clinical + ["counselling"] * n_counsellors
    for i, skill in enumerate(skills):
        if rng.random() < 0.75:
            p = centre + rng.normal(0, 8, 2)
        else:
            p = rng.uniform(10, 90, 2)
        workers.append({"id": f"W{i + 1:02d}", "skill": skill,
                        "x": round(float(np.clip(p[0], 0, 100)), 1),
                        "y": round(float(np.clip(p[1], 0, 100)), 1),
                        "day_off": DAYS[int(rng.integers(0, 5))],
                        "max_shifts": max_shifts})
    workers = pd.DataFrame(workers)

    # Demand: staff needed per site, skill and day.
    # Rural sites run outreach on Mon, Wed and Fri only.
    per_day = {"urban": {"clinical": 2, "counselling": 1},
               "peri-urban": {"clinical": 1, "counselling": 1},
               "rural": {"clinical": 1, "counselling": 1}}
    rural_days = {"Mon", "Wed", "Fri"}
    demand = []
    for _, s in sites.iterrows():
        for d in DAYS:
            if s["type"] == "rural" and d not in rural_days:
                continue
            for k in SKILLS:
                demand.append({"site": s["id"], "skill": k, "day": d,
                               "need": per_day[s["type"]][k]})
    demand = pd.DataFrame(demand)

    # One-way road distance in km.
    W = workers[["x", "y"]].to_numpy()
    S = sites[["x", "y"]].to_numpy()
    d = np.sqrt(((W[:, None, :] - S[None, :, :]) ** 2).sum(-1)) * ROAD_FACTOR
    dist_km = pd.DataFrame(d.round(1), index=workers["id"], columns=sites["id"])

    supply = {k: int((workers["skill"] == k).sum() * max_shifts) for k in SKILLS}
    need = demand.groupby("skill")["need"].sum().to_dict()
    meta = {"seed": seed, "supply_shifts": supply, "demand_shifts": need}
    return Instance(workers, sites, demand, dist_km, meta)


if __name__ == "__main__":
    inst = make_instance()
    print(inst.sites.to_string(index=False))
    print(inst.workers.head().to_string(index=False))
    print(inst.meta)
