"""Check that a roster follows every rule, and score it."""

import pandas as pd


def check_feasible(inst, rows):
    """Raise AssertionError if the roster breaks any hard rule."""
    W = inst.workers.set_index("id")
    need = {(r.site, r.skill, r.day): int(r.need) for r in inst.demand.itertuples()}
    df = pd.DataFrame(rows, columns=["worker", "site", "day"])
    if df.empty:
        return True
    df["skill"] = df["worker"].map(W["skill"])

    assert not df.duplicated(["worker", "day"]).any(), "worker in two places on one day"
    assert (df["day"] != df["worker"].map(W["day_off"])).all(), "worker rostered on day off"
    per_w = df.groupby("worker").size()
    assert (per_w <= W.loc[per_w.index, "max_shifts"]).all(), "worker over weekly shift cap"
    for (s, k, d), n in df.groupby(["site", "skill", "day"]).size().items():
        assert (s, k, d) in need, f"shift assigned where no demand: {s},{k},{d}"
        assert n <= need[(s, k, d)], f"overstaffed {s},{k},{d}"
    return True


def score(inst, rows, name):
    W = inst.workers.set_index("id")
    sites = inst.sites.set_index("id")
    df = pd.DataFrame(rows, columns=["worker", "site", "day"])
    df["km"] = [inst.round_trip(w, s) for w, s in zip(df["worker"], df["site"])]

    need_site = inst.demand.groupby("site")["need"].sum()
    got_site = df.groupby("site").size().reindex(need_site.index, fill_value=0)
    fill = (got_site / need_site)

    worker_km = df.groupby("worker")["km"].sum().reindex(W.index, fill_value=0)
    rural = sites.index[sites["type"] == "rural"]

    summary = {
        "plan": name,
        "total_travel_km": round(df["km"].sum(), 1),
        "shifts_filled": int(len(df)),
        "shifts_unfilled": int(need_site.sum() - len(df)),
        "overall_fill_pct": round(100 * len(df) / need_site.sum(), 1),
        "rural_fill_pct": round(100 * got_site[rural].sum() / need_site[rural].sum(), 1),
        "lowest_site_fill_pct": round(100 * fill.min(), 1),
        "max_worker_km": round(worker_km.max(), 1),
        "avg_km_per_shift": round(df["km"].mean(), 1),
    }
    site_detail = pd.DataFrame({"site": need_site.index,
                                "type": sites.loc[need_site.index, "type"].values,
                                "needed": need_site.values,
                                "filled": got_site.values,
                                "fill_pct": (100 * fill).round(1).values,
                                "plan": name})
    return summary, site_detail, worker_km.rename(name)
