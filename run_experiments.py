"""Run every experiment, save result tables to results/ and charts to figures/.

Usage:  python run_experiments.py
"""

from fractions import Fraction
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

from src.baseline import greedy_nearest
from src.data import make_instance
from src.evaluate import check_feasible, score
from src.model import solve

ROOT = Path(__file__).parent
RES, FIG = ROOT / "results", ROOT / "figures"
RES.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

SITE_FLOOR = 0.80      # every site gets at least 80% of its shifts filled
WORKER_CAP_KM = 280    # no worker drives more than 280 km in the week

PLANS = ["Manual rule", "Lowest travel", "+ Site equity", "+ Staff fairness"]
COLORS = {"Manual rule": "#8a8984", "Lowest travel": "#2a78d6",
          "+ Site equity": "#eb6834", "+ Staff fairness": "#1baf7a"}
INK, INK2, GRID, BG = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlecolor": INK, "figure.facecolor": BG, "axes.facecolor": BG,
    "savefig.facecolor": BG, "axes.spines.top": False, "axes.spines.right": False,
})


def run_plans(inst):
    rosters = {
        "Manual rule": greedy_nearest(inst),
        "Lowest travel": solve(inst)[0],
        "+ Site equity": solve(inst, min_fill=SITE_FLOOR)[0],
        "+ Staff fairness": solve(inst, min_fill=SITE_FLOOR,
                                  max_worker_km=WORKER_CAP_KM)[0],
    }
    summaries, sites, workers = [], [], []
    for name, rows in rosters.items():
        check_feasible(inst, rows)  # every roster must obey every hard rule
        s, site_df, wkm = score(inst, rows, name)
        summaries.append(s)
        sites.append(site_df)
        workers.append(wkm)
        slug = name.lower().replace("+ ", "").replace(" ", "_")
        pd.DataFrame(rows).sort_values(["day", "site", "worker"]).to_csv(
            RES / f"roster_{slug}.csv", index=False)
    return (pd.DataFrame(summaries), pd.concat(sites), pd.concat(workers, axis=1),
            rosters)


def sweep(inst):
    """Price of equity: how much extra travel each site floor costs."""
    rows = []
    for f in [0, .2, .3, .4, .5, .6, .7, .8, 5 / 6, .9]:
        r, status = solve(inst, min_fill=f)
        rows.append({"min_site_fill_pct": round(100 * f, 1), "status": status,
                     "total_travel_km": score(inst, r, "")[0]["total_travel_km"] if r else None})
    return pd.DataFrame(rows)


def cap_sweep(inst):
    """Staff fairness: shifts filled and total travel under the site floor,
    for a range of caps on any one worker's weekly travel."""
    rows = []
    for cap in range(180, 361, 10):
        r, status = solve(inst, min_fill=SITE_FLOOR, max_worker_km=cap)
        sm = score(inst, r, "")[0] if r else {}
        rows.append({"max_worker_km_cap": cap, "status": status,
                     "shifts_filled": sm.get("shifts_filled"),
                     "total_travel_km": sm.get("total_travel_km")})
    return pd.DataFrame(rows)


def other_regions(seeds=range(1, 31)):
    """Repeat the main comparison on 30 other random regions.

    Checks that the results are not an artefact of one lucky instance.
    """
    rows = []
    for seed in seeds:
        inst = make_instance(seed=seed)
        plans = {"manual": greedy_nearest(inst), "lowest": solve(inst)[0]}
        equity, status = solve(inst, min_fill=SITE_FLOOR)
        plans["equity"] = equity
        row = {"seed": seed, "equity_status": status}
        for name, r in plans.items():
            if r is None:
                continue
            check_feasible(inst, r)
            sm = score(inst, r, name)[0]
            row[f"{name}_filled"] = sm["shifts_filled"]
            row[f"{name}_km"] = sm["total_travel_km"]
            row[f"{name}_worst_site_pct"] = sm["lowest_site_fill_pct"]
        rows.append(row)
    return pd.DataFrame(rows)


# Every site's weekly need is 6, 10 or 15 shifts, so the floors that can make a
# difference are the fractions n/6, n/10 and n/15.
FLOORS = sorted({float(Fraction(n, m)) for m in (6, 10, 15) for n in range(1, m + 1)})


def highest_floor(inst, flex=False):
    """Highest coverage floor every site can meet, found by bisection."""
    lo, hi, best = 0, len(FLOORS) - 1, 0.0
    while lo <= hi:
        mid = (lo + hi) // 2
        rows, _ = solve(inst, min_fill=FLOORS[mid] - 1e-9, flex=flex)
        if rows:
            best, lo = FLOORS[mid], mid + 1
        else:
            hi = mid - 1
    return best


def flexibility(inst):
    """Strict skills vs letting clinical staff cover counselling shifts.

    Returns the main region's floor sweep under both rules, a staff-mix
    sensitivity (30 workers split differently between the two skills) and
    a check on 30 other regions.
    """
    floors = []
    for flex in (False, True):
        for f in [0.0, 0.6, SITE_FLOOR, 5 / 6]:
            r, status = solve(inst, min_fill=f, flex=flex)
            sm = score(inst, r, "")[0] if r else {}
            floors.append({"skills": "flexible" if flex else "strict",
                           "min_site_fill_pct": round(100 * f, 1), "status": status,
                           "shifts_filled": sm.get("shifts_filled"),
                           "total_travel_km": sm.get("total_travel_km")})

    mix = []
    for n_clin in (14, 16, 18, 20, 22):
        m = make_instance(n_clinical=n_clin, n_counsellors=30 - n_clin)
        for flex in (False, True):
            r, _ = solve(m, min_fill=SITE_FLOOR, flex=flex)
            check_feasible(m, r, flex=flex)
            sm = score(m, r, "")[0]
            mix.append({"clinical_staff": n_clin, "counselling_staff": 30 - n_clin,
                        "skills": "flexible" if flex else "strict",
                        "shifts_filled_80pct_floor": sm["shifts_filled"],
                        "total_travel_km_80pct_floor": sm["total_travel_km"],
                        "highest_feasible_floor_pct": round(100 * highest_floor(m, flex), 1)})

    regions = []
    for seed in range(1, 31):
        m = make_instance(seed=seed)
        row = {"seed": seed}
        for flex in (False, True):
            tag = "flexible" if flex else "strict"
            r, _ = solve(m, flex=flex)
            row[f"{tag}_filled"] = score(m, r, "")[0]["shifts_filled"]
            row[f"{tag}_highest_floor_pct"] = round(100 * highest_floor(m, flex), 1)
        regions.append(row)
    return pd.DataFrame(floors), pd.DataFrame(mix), pd.DataFrame(regions)


def fig_flexibility(mix):
    """Paired dots: strict skills vs flexible skills for each staff mix.

    Dots rather than bars, because the y-axes do not start at zero.
    """
    strict = mix[mix.skills == "strict"].set_index("clinical_staff")
    flex = mix[mix.skills == "flexible"].set_index("clinical_staff")
    xs = np.arange(len(strict))
    labels = [f"{c} / {30 - c}" for c in strict.index]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, col, title, fmt in [
            (a1, "shifts_filled_80pct_floor", "A. Shifts filled with an 80% site floor", "{:.0f}"),
            (a2, "highest_feasible_floor_pct", "B. Highest floor every site can meet (%)", "{:.1f}")]:
        for x, s_val, f_val in zip(xs, strict[col], flex[col]):
            ax.plot([x, x], [s_val, f_val], color=GRID, lw=3, zorder=1)
            ax.annotate(fmt.format(s_val), (x, s_val), xytext=(-8, 0),
                        textcoords="offset points", ha="right", va="center",
                        fontsize=8, color=INK2)
            if f_val != s_val:
                ax.annotate(fmt.format(f_val), (x, f_val), xytext=(8, 0),
                            textcoords="offset points", ha="left", va="center",
                            fontsize=8, color=INK)
        ax.scatter(xs, strict[col], s=60, color="#8a8984", zorder=2, label="Strict skills")
        ax.scatter(xs, flex[col], s=60, color="#2a78d6", zorder=3,
                   label="Clinical staff may cover counselling")
        vals = list(strict[col]) + list(flex[col])
        pad = 0.15 * (max(vals) - min(vals))
        ax.set_ylim(min(vals) - pad, max(vals) + pad)
        ax.set_xlim(-0.6, len(xs) - 0.4)
        ax.set_xticks(xs, labels)
        ax.set_xlabel("Clinical / counselling staff (30 in total; main region is 18 / 12)")
        ax.set_title(title, loc="left", fontweight="bold", fontsize=10)
        ax.grid(axis="y", color=GRID, lw=.6)
    a1.legend(frameon=False, loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "flexibility.png", dpi=160)
    plt.close(fig)


def fig_map(inst, roster):
    fig, ax = plt.subplots(figsize=(7, 6.4))
    W = inst.workers.set_index("id")
    S = inst.sites.set_index("id")
    mon = [r for r in roster if r["day"] == "Mon"]
    for r in mon:
        w, s = W.loc[r["worker"]], S.loc[r["site"]]
        ax.plot([w.x, s.x], [w.y, s.y], color="#1baf7a", lw=1, alpha=.7, zorder=1)
    for skill, marker in [("clinical", "o"), ("counselling", "^")]:
        sub = inst.workers[inst.workers.skill == skill]
        ax.scatter(sub.x, sub.y, marker=marker, s=34, color="#8a8984",
                   edgecolor=BG, lw=1, zorder=2, label=f"Worker ({skill})")
    shade = {"urban": INK, "peri-urban": "#52514e", "rural": "#a3a29c"}
    for t in ["urban", "peri-urban", "rural"]:
        sub = inst.sites[inst.sites.type == t]
        ax.scatter(sub.x, sub.y, marker="s", s=90, color=shade[t],
                   edgecolor=BG, lw=1.5, zorder=3, label=f"Site ({t})")
    # Hand-placed labels for the default region, where urban sites crowd together.
    nudge = {"U1": (9, -13), "U3": (-9, -12), "U5": (-9, 3)}
    for s in inst.sites.itertuples():
        dx, dy = nudge.get(s.id, (7, 6)) if inst.meta.get("seed") == 42 else (7, 6)
        ax.annotate(s.id, (s.x, s.y), xytext=(dx, dy), textcoords="offset points",
                    fontsize=9, color=INK, fontweight="bold", ha="left" if dx > 0 else "right",
                    bbox=dict(boxstyle="square,pad=0.1", fc=BG, ec="none", alpha=.8))
    ax.plot([], [], color="#1baf7a", lw=1, label="Monday trips (staff-fairness plan)")
    ax.set(xlim=(0, 100), ylim=(0, 100), xlabel="km", ylabel="km")
    ax.set_title(f"Synthetic region: {len(inst.workers)} workers, {len(inst.sites)} sites",
                 loc="left", fontweight="bold")
    ax.legend(loc="upper left", fontsize=8, frameon=False, bbox_to_anchor=(1.01, 1))
    ax.grid(color=GRID, lw=.6)
    fig.tight_layout()
    fig.savefig(FIG / "map.png", dpi=160)
    plt.close(fig)


def fig_comparison(summary, total_shifts):
    panels = [("total_travel_km", "Total travel (km)", "{:,.0f}"),
              ("lowest_site_fill_pct", "Worst-served site (% filled)", "{:.0f}%"),
              ("rural_fill_pct", "Rural sites (% filled)", "{:.0f}%"),
              ("max_worker_km", "Most km by one worker", "{:,.0f}")]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), sharey=False)
    y = np.arange(len(PLANS))[::-1]
    for ax, (col, title, fmt) in zip(axes, panels):
        vals = summary.set_index("plan").loc[PLANS, col].to_numpy()
        ax.barh(y, vals, height=.62, color=[COLORS[p] for p in PLANS])
        for yi, v in zip(y, vals):
            ax.text(v, yi, " " + fmt.format(v), va="center", fontsize=9, color=INK)
        ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
        ax.set_xlim(0, vals.max() * 1.28)
        ax.set_yticks(y, PLANS if ax is axes[0] else [""] * len(PLANS))
        ax.tick_params(axis="y", length=0)
        ax.xaxis.set_visible(False)
        ax.spines["bottom"].set_visible(False)
    filled = summary["shifts_filled"]
    note = (f"all plans fill {filled.iloc[0]} of {total_shifts} shifts" if filled.nunique() == 1
            else f"shifts filled out of {total_shifts}: " + ", ".join(map(str, filled)))
    fig.suptitle(f"Same staff, same week, four ways to roster them  ({note})",
                 x=.01, ha="left", fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "comparison.png", dpi=160)
    plt.close(fig)


def fig_tradeoff(sw, caps, baseline_km, baseline_worst, chosen_cap):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2))

    ok = sw[sw.status == "Optimal"]
    a1.plot(ok.min_site_fill_pct, ok.total_travel_km, color="#eb6834", lw=2,
            marker="o", ms=6, mec=BG, mew=1.5, label="Optimized plan")
    last = ok.iloc[-1]
    a1.annotate(f"{last.total_travel_km:,.0f} km at {last.min_site_fill_pct:.0f}%\n(highest feasible floor)",
                (last.min_site_fill_pct, last.total_travel_km), xytext=(-12, 10),
                textcoords="offset points", color=INK, fontsize=9, ha="right")
    a1.set(xlabel="Minimum % of shifts filled at every site")
    a1.set_title("A. Raising the coverage floor for every site", loc="left",
                 fontweight="bold", fontsize=10)

    okc = caps[caps.status == "Optimal"]
    full = okc[okc.shifts_filled == okc.shifts_filled.max()]
    short = okc[okc.shifts_filled < okc.shifts_filled.max()]
    a2.plot(full.max_worker_km_cap, full.total_travel_km, color="#1baf7a", lw=2,
            marker="o", ms=6, mec=BG, mew=1.5,
            label=f"Optimized plan, {int(full.shifts_filled.iloc[0])} shifts filled")
    if len(short):
        a2.plot(short.max_worker_km_cap, short.total_travel_km, color="#1baf7a", lw=0,
                marker="o", ms=6, mfc=BG, mew=1.5,
                label="Fewer shifts filled (cap too tight)")
    a2.axvline(chosen_cap, color=INK2, lw=1, ls=":")
    a2.annotate(f"cap used: {chosen_cap} km", (chosen_cap, 1), xycoords=("data", "axes fraction"),
                xytext=(4, -14), textcoords="offset points", color=INK2, fontsize=9)
    a2.set(xlabel="Most km any one worker may travel in the week")
    a2.set_title("B. Capping the travel of any one worker", loc="left",
                 fontweight="bold", fontsize=10)

    vals = list(ok.total_travel_km) + list(okc.total_travel_km) + [baseline_km]
    lo, hi = min(vals), max(vals)
    for ax in (a1, a2):
        ax.axhline(baseline_km, color="#8a8984", lw=2, ls="--", label="Manual rule")
        ax.set_ylim(lo - 0.06 * (hi - lo), hi + 0.12 * (hi - lo))
        ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
        ax.grid(axis="y", color=GRID, lw=.6)
    a1.legend(frameon=False, loc="lower right", fontsize=8)
    a2.legend(frameon=False, loc="lower left", fontsize=8)
    a1.set_ylabel("Total travel (km)")
    a1.annotate(f"Manual rule: {baseline_km:,.0f} km, worst site {baseline_worst:.0f}% filled",
                (100, baseline_km), xytext=(0, -14), textcoords="offset points",
                color=INK2, fontsize=8, ha="right")
    a1.set_xlim(right=100)
    fig.tight_layout()
    fig.savefig(FIG / "tradeoff.png", dpi=160)
    plt.close(fig)


def main():
    inst = make_instance()
    inst.sites.to_csv(RES / "sites.csv", index=False)
    inst.workers.to_csv(RES / "workers.csv", index=False)
    inst.demand.to_csv(RES / "demand.csv", index=False)

    summary, sites, workers, rosters = run_plans(inst)
    summary.to_csv(RES / "summary.csv", index=False)
    sites.to_csv(RES / "site_fill.csv", index=False)
    workers.round(1).to_csv(RES / "worker_km.csv")
    sw = sweep(inst)
    sw.to_csv(RES / "equity_sweep.csv", index=False)
    caps = cap_sweep(inst)
    caps.to_csv(RES / "cap_sweep.csv", index=False)
    rob = other_regions()
    rob.to_csv(RES / "other_regions.csv", index=False)
    flex_floors, flex_mix, flex_regions = flexibility(inst)
    flex_floors.to_csv(RES / "flexibility_floors.csv", index=False)
    flex_mix.to_csv(RES / "flexibility_staff_mix.csv", index=False)
    flex_regions.to_csv(RES / "flexibility_regions.csv", index=False)

    fig_map(inst, rosters["+ Staff fairness"])
    fig_comparison(summary, int(inst.demand["need"].sum()))
    manual = summary.set_index("plan").loc["Manual rule"]
    fig_tradeoff(sw, caps, manual["total_travel_km"], manual["lowest_site_fill_pct"],
                 WORKER_CAP_KM)
    fig_flexibility(flex_mix)

    pd.set_option("display.width", 160)
    print(summary.to_string(index=False))
    print(sw.to_string(index=False))
    print(caps.to_string(index=False))
    print(rob.describe().round(1).to_string())
    print(flex_floors.to_string(index=False))
    print(flex_mix.to_string(index=False))
    print(flex_regions.describe().round(1).to_string())


if __name__ == "__main__":
    main()
