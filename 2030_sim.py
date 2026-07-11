import pypsa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import logging

logging.getLogger("linopy").setLevel(logging.WARNING)
logging.getLogger("gurobipy").setLevel(logging.WARNING)

os.makedirs("networks", exist_ok=True)
os.makedirs("results",  exist_ok=True)

GUROBI_OPTIONS = {
    "NumericFocus": 3, "method": 2, "crossover": 0,
    "BarHomogeneous": 1, "BarConvTol": 1e-3,
    "FeasibilityTol": 1e-4, "OptimalityTol": 1e-4,
    "ObjScale": -0.5, "threads": 8, "Seed": 123, "LogToConsole": 0,
}

SEASONS = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4,
    "May": 5, "Jun": 6, "Jul": 7, "Aug": 8,
    "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}

MONTH_NAMES = list(SEASONS.keys())

RE_CARRIERS = ["solar", "onwind", "offwind-ac", "offwind-dc", "ror", "biomass", "geothermal"]

COLORS = {
    "solar":                       "#FFC107",
    "onwind":                      "#03A9F4",
    "offshore wind":               "#0D47A1",
    "solar + solar hsat":          "#FFC107",
    "geothermal + biomass + ror":  "#2CB332",
    "ocgt + ccgt":                 "#BE086C",
    "coal":                        "#212121",
    "lignite":                     "#4E342E",
    "oil":                         "#97099C",
    "nuclear":                     "#9b5de5",
    "waste":                       "#0C5C34",
    "BESS Discharge":              "#E65100",
    "BESS Charge":                 "#FFB74D",
}

def annuity(n_years, r):
    return r * (1 + r)**n_years / ((1 + r)**n_years - 1)


def calculate_capital_costs():
    # 2030 BESS
    cc = (annuity(10, 0.07) * 213.9279
          + annuity(25, 0.07) * 189.861 * 4
          + (0.3375 / 100) * 213.9279)
    # Baseline BESS
    cc_b = (annuity(10, 0.07) * 405.9268
            + annuity(22.5, 0.07) * 354.0137 * 4
            + (0.2512 / 100) * 405.9268)
    return cc, cc_b


def load_growth_factors(file_path):
    df = pd.read_csv(file_path)
    df["growth_factor"] = df["avg_2030_GW"] / df["capacity_2025_GW"]
    print("Growth Factors:\n", df["growth_factor"].head(20))
    return dict(zip(df["pypsa_carrier"], df["growth_factor"]))


# ─── CUSTOM CONSTRAINT ────────────────────────────────────────────────────────
def add_max_re_constraint(n_inner, snapshots):
    m = n_inner.model
    p_gen = m.variables["Generator-p"]
    gen_dim = [d for d in p_gen.dims if d != "snapshot"][0]

    re_idx = n_inner.generators.index[n_inner.generators.carrier.isin(RE_CARRIERS)]
    valid  = re_idx[re_idx.isin(p_gen.coords[gen_dim].values)]

    ts_gens     = valid[valid.isin(n_inner.generators_t.p_max_pu.columns)]
    static_gens = valid[~valid.isin(n_inner.generators_t.p_max_pu.columns)]

    ts_pot = (n_inner.generators_t.p_max_pu.loc[snapshots, ts_gens]
              * n_inner.generators.p_nom[ts_gens]).sum().sum()
    st_pot = (n_inner.generators.loc[static_gens, "p_max_pu"]
              * n_inner.generators.loc[static_gens, "p_nom"]).sum() * len(snapshots)

    m.add_constraints(p_gen.sel({gen_dim: valid}).sum() >= float(ts_pot + st_pot),
                      name="force_max_re")


# ─── SIMULATION ────────────────────────────────────────────────────────
def run_monthly_simulations(n_full, carrier_growth, capital_cost, capital_cost_b):
    for season_name, month in SEASONS.items():
        print(f"\n{'*'*55}\n  {season_name}\n{'*'*55}")
        snapshots = n_full.snapshots[n_full.snapshots.month == month]

        n      = n_full.copy()
        n_2030 = n_full.copy()
        n.set_snapshots(snapshots)
        n_2030.set_snapshots(snapshots)

        # Scale 2030 network
        # for carrier, factor in carrier_growth.items():
        #     if carrier == "load":
        #         n_2030.loads_t.p_set *= factor
        #     else:
        #         targets = (["offwind-ac", "offwind-dc"] if carrier == "offwind"
        #                    else ["OCGT", "CCGT"]        if carrier == "gas"
        #                    else [carrier])
        #         mask = n_2030.generators.carrier.isin(targets)
        #         n_2030.generators.loc[mask, "p_nom"] *= factor

        for carrier, factor in carrier_growth.items():
            if carrier == "load":
                n_2030.loads_t.p_set *= factor
                
            elif carrier in ["PHS", "hydro"]:
                # 1. Update StorageUnits (typically PHS and hydro reservoirs)
                storage_mask = n_2030.storage_units.carrier == carrier
                n_2030.storage_units.loc[storage_mask, "p_nom"] *= factor
            else:
                # Your existing logic for standard generators
                targets = (["offwind-ac", "offwind-dc"] if carrier == "offwind"
                        else ["OCGT", "CCGT"]        if carrier == "gas"
                        else [carrier])
                mask = n_2030.generators.carrier.isin(targets)
                n_2030.generators.loc[mask, "p_nom"] *= factor

        # Add BESS
        ac_buses = n_2030.buses[n_2030.buses.carrier == "AC"].index
        for bus in ac_buses:
            for net, cc in [(n_2030, capital_cost), (n, capital_cost_b)]:
                net.add("StorageUnit", f"BESS_{bus}", bus=bus, carrier="battery",
                        p_nom_extendable=True, capital_cost=cc,
                        max_hours=4, efficiency_store=0.95, efficiency_dispatch=0.95,
                        cyclic_state_of_charge=True)

        n_2030.generators.p_nom_extendable = False
        n.generators.p_nom_extendable = False
        n.lines.p_nom_extendable = False
        n_2030.lines.p_nom_extendable = False

        n.optimize(solver_name="gurobi", solver_options=GUROBI_OPTIONS)
        n_2030.optimize(solver_name="gurobi",
                        extra_functionality=add_max_re_constraint,
                        solver_options=GUROBI_OPTIONS)

        pfx = season_name.lower()
        n_2030.export_to_netcdf(f"networks/solved_2030_{pfx}.nc")
        n.export_to_netcdf(f"networks/base_solved_2030_{pfx}.nc")
        print(f"Saved: {pfx}")


def combine_monthly_networks(n_full):
    print("\nCombining monthly networks...")

    n_annual      = pypsa.Network("networks/base_solved_2030_jan.nc").copy()
    n_2030_annual = pypsa.Network("networks/solved_2030_jan.nc").copy()

    n_sample_2030 = pypsa.Network("networks/solved_2030_jan.nc")
    n_sample_base = pypsa.Network("networks/base_solved_2030_jan.nc")

    bess_mask_2030 = n_sample_2030.storage_units.index.str.contains("BESS", case=False)
    bess_mask_base = n_sample_base.storage_units.index.str.contains("BESS", case=False)

    for name, row in n_sample_2030.storage_units[bess_mask_2030].iterrows():
        n_2030_annual.add("StorageUnit", name, bus=row.bus, carrier=row.carrier,
                p_nom=row.p_nom_opt, p_nom_opt=row.p_nom_opt,
                p_nom_extendable=False, max_hours=row.max_hours,
                efficiency_store=row.efficiency_store,
                efficiency_dispatch=row.efficiency_dispatch,
                cyclic_state_of_charge=row.cyclic_state_of_charge)

    for name, row in n_sample_base.storage_units[bess_mask_base].iterrows():
        n_annual.add("StorageUnit", name, bus=row.bus, carrier=row.carrier,
                p_nom=row.p_nom_opt, p_nom_opt=row.p_nom_opt,
                p_nom_extendable=False, max_hours=row.max_hours,
                efficiency_store=row.efficiency_store,
                efficiency_dispatch=row.efficiency_dispatch,
                cyclic_state_of_charge=row.cyclic_state_of_charge)

    print(f"Added {bess_mask_2030.sum()} BESS units to 2030 annual network.")
    print(f"Added {bess_mask_base.sum()} BESS units to baseline annual network.")

    components = [
        ("generators",    ["p"]),
        ("storage_units", ["p", "state_of_charge"]),
        ("buses",         ["marginal_price"]),
        ("lines",         ["p0", "p1"]),
        ("loads",         ["p_set"]),
    ]
    for comp, attrs in components:
        for attr in attrs:
            frames_b, frames_2 = [], []
            for pfx in [s.lower() for s in SEASONS]:
                nb = pypsa.Network(f"networks/base_solved_2030_{pfx}.nc")
                n2 = pypsa.Network(f"networks/solved_2030_{pfx}.nc")
                db = getattr(getattr(nb, f"{comp}_t"), attr, pd.DataFrame())
                d2 = getattr(getattr(n2, f"{comp}_t"), attr, pd.DataFrame())
                if not db.empty: frames_b.append(db)
                if not d2.empty: frames_2.append(d2)
            if frames_b:
                setattr(getattr(n_annual,      f"{comp}_t"), attr, pd.concat(frames_b))
            if frames_2:
                setattr(getattr(n_2030_annual, f"{comp}_t"), attr, pd.concat(frames_2))

    n_annual.set_snapshots(n_full.snapshots)
    n_2030_annual.set_snapshots(n_full.snapshots)

    n_annual.export_to_netcdf("networks/combined_baseline_annual.nc")
    n_2030_annual.export_to_netcdf("networks/combined_2030_annual.nc")
    print("Saved annual networks.")


def c(name):
    return COLORS.get(name, "#cccccc")


def consolidated_dispatch(n):
    gen = n.generators_t.p.T.groupby(n.generators.carrier).sum().T.clip(lower=0)

    groups = {
        "ocgt + ccgt":                ["OCGT", "CCGT", "ocgt", "ccgt"],
        "solar + solar hsat":         ["solar", "solar hsat", "solar-hsat"],
        "offshore wind":              ["offwind-ac", "offwind-dc", "offwind-float"],
        "geothermal + biomass + ror": ["geothermal", "biomass", "ror"],
    }
    for name, carriers in groups.items():
        cols = [x for x in carriers if x in gen.columns]
        if cols:
            gen[name] = gen[cols].sum(axis=1)
            gen.drop(columns=cols, inplace=True)

    bess = n.storage_units.index[n.storage_units.index.str.contains("BESS|battery", case=False)]
    if len(bess):
        p = n.storage_units_t.p[bess].sum(axis=1)
        gen["BESS Discharge"] = p.clip(lower=0)
        gen["BESS Charge"]    = p.clip(upper=0)

    return gen.loc[:, (gen != 0).any()]


def re_share(n):
    re = n.generators.index[n.generators.carrier.isin(RE_CARRIERS)]
    return n.generators_t.p[re].sum().sum() / n.loads_t.p_set.sum().sum() * 100


def curtailment_pct(n):
    re        = n.generators.index[n.generators.carrier.isin(RE_CARRIERS)]
    p_nom_col = "p_nom_opt" if n.generators.p_nom_extendable.any() else "p_nom"
    potential = (n.generators_t.p_max_pu.reindex(columns=re).fillna(1.0)
                 * n.generators.loc[re, p_nom_col]).sum().sum()
    actual    = n.generators_t.p[re].sum().sum()
    return (potential - actual) / potential * 100 if potential > 0 else 0.0


def bess_stats(n):
    bess      = n.storage_units[n.storage_units.carrier == "battery"]
    cap_mw    = bess.p_nom_opt.sum()
    discharge = n.storage_units_t.p[bess.index].clip(lower=0).sum().sum()
    cycles    = discharge / (cap_mw * 4) if cap_mw > 0 else 0
    return cap_mw / 1e3, cap_mw * 4 / 1e3, discharge / 1e3, cycles


def print_metrics(label, n):
    cap_gw, cap_gwh, disch, cycles = bess_stats(n)
    print(f"\n{'═'*50}\n  {label}\n{'═'*50}")
    print(f"  RE share:           {re_share(n):.1f} %")
    print(f"  Curtailment:        {curtailment_pct(n):.2f} %")
    print(f"  BESS:               {cap_gw:.2f} GW / {cap_gwh:.2f} GWh")
    print(f"  BESS discharge:     {disch:,.0f} GWh  ({cycles:.0f} cycles)")
    print(f"  Avg LMP:            {n.buses_t.marginal_price.mean().mean():.2f} EUR/MWh")
    print(f"  Negative-price hrs: {int((n.buses_t.marginal_price < 0).sum().sum())}")
    print(f"  System cost:        {n.objective/1e6:.1f} M EUR")


def save_metrics(n_base, n_2030):
    rows = []
    for label, n in [("Baseline", n_base), ("2030 Projection", n_2030)]:
        cap_gw, cap_gwh, disch, cycles = bess_stats(n)
        rows.append({
            "Scenario":                   label,
            "RE Share of Load (%)":       round(re_share(n), 1),
            "BESS Capacity (GW)":         round(cap_gw, 2),
            "BESS Energy (GWh)":          round(cap_gwh, 2),
            "BESS Total Discharge (GWh)": round(disch, 1),
            "BESS Full Cycles":           round(cycles, 0),
            "Avg LMP (EUR/MWh)":          round(n.buses_t.marginal_price.mean().mean(), 2),
            "Negative Price Hours":       int((n.buses_t.marginal_price < 0).sum().sum()),
            "System Cost (MEUR)":         round(n.objective / 1e6, 1),
        })
    df = pd.DataFrame(rows).set_index("Scenario")
    df.to_csv("results/metrics_annual.csv")
    print("\nSaved: results/metrics_annual.csv")
    print(df.T.to_string())


def plot_dispatch(n_base, n_2030):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)

    gen_base = consolidated_dispatch(n_base).resample("D").sum() / 1e3
    gen_2030 = consolidated_dispatch(n_2030).resample("D").sum() / 1e3
    load_base = n_base.loads_t.p_set.sum(axis=1).resample("D").sum() / 1e3
    load_2030 = n_2030.loads_t.p_set.sum(axis=1).resample("D").sum() / 1e3

    max_p = max(
        gen_base[gen_base > 0].sum(axis=1).max(),
        gen_2030[gen_2030 > 0].sum(axis=1).max(),
        load_base.max(), load_2030.max()
    ) * 1.1
    min_p = min(
        gen_base[gen_base < 0].sum(axis=1).min() if (gen_base < 0).any().any() else 0,
        gen_2030[gen_2030 < 0].sum(axis=1).min() if (gen_2030 < 0).any().any() else 0
    ) * 1.1

    for ax, gen, load, title in [
        (ax1, gen_base, load_base, "Baseline — Annual Dispatch"),
        (ax2, gen_2030, load_2030, "2030 Projection — Annual Dispatch (RE priority)"),
    ]:
        pos = [x for x in gen.columns if x != "BESS Charge"]
        neg = [x for x in gen.columns if x == "BESS Charge"]

        gen[pos].plot.area(ax=ax, stacked=True, alpha=0.9,
                           color=[c(x) for x in pos], linewidth=0)
        if neg:
            gen[neg].plot.area(ax=ax, stacked=True, alpha=0.7,
                               color=[c("BESS Charge")], linewidth=0)
        load.plot(ax=ax, color="black", linewidth=1.5, linestyle="--",
                  label="Total Load", zorder=20)

        ax.set_ylim(min_p, max_p)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylabel("GWh / day")
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=9)

    plt.tight_layout()
    plt.savefig("results/annual_dispatch.png", dpi=150)
    plt.close()
    print("Saved: results/annual_dispatch.png")


def plot_lmp(n_base, n_2030):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)

    for ax, n, label, col in [
        (ax1, n_base, "Baseline",        "#3498db"),
        (ax2, n_2030, "2030 Projection", "#e67e22"),
    ]:
        lmp      = n.buses_t.marginal_price
        lmp_mean = lmp.mean(axis=1)

        ax.fill_between(lmp_mean.index, lmp.min(axis=1), lmp.max(axis=1),
                        alpha=0.15, color=col, label="Node range")
        ax.plot(lmp_mean.index, lmp_mean, color=col,
                linewidth=0.8, label=f"{label} avg LMP")
        ax.axhline(0, color="black", linewidth=0.5, linestyle=":")

        neg = lmp_mean < 0
        if neg.any():
            ax.fill_between(lmp_mean.index, lmp_mean, 0,
                            where=neg, color="#e71d36", alpha=0.4,
                            label="Negative prices")

        ax.set_ylabel("EUR/MWh")
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("results/annual_lmp.png", dpi=150)
    plt.close()
    print("Saved: results/annual_lmp.png")


def plot_lmp_distribution(n_base, n_2030):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, n, label, col in [
        (ax1, n_base, "Baseline",        "#3498db"),
        (ax2, n_2030, "2030 Projection", "#e67e22"),
    ]:
        vals = n.buses_t.marginal_price.values.flatten()
        vals = vals[~np.isnan(vals)]
        ax.hist(vals, bins=100, color=col, alpha=0.8, edgecolor="none")
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("EUR/MWh")
        ax.set_ylabel("Frequency" if ax == ax1 else "")
        ax.set_title(label)
        ax.text(0.98, 0.95, f"Negative: {(vals < 0).mean()*100:.1f}%",
                transform=ax.transAxes, ha="right", va="top", fontsize=9)

    plt.tight_layout()
    plt.savefig("results/annual_lmp_distribution.png", dpi=150)
    plt.close()
    print("Saved: results/annual_lmp_distribution.png")


def plot_bess(n_2030):
    bess = n_2030.storage_units.index[n_2030.storage_units.carrier == "battery"]
    if bess.empty:
        print("No BESS found, skipping.")
        return

    p   = n_2030.storage_units_t.p[bess].sum(axis=1).resample("D").mean()
    soc = n_2030.storage_units_t.state_of_charge[bess].sum(axis=1).resample("D").mean()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)

    ax1.bar(p.index, p.clip(lower=0), color=c("BESS Discharge"),
            alpha=0.85, width=0.8, label="Discharge")
    ax1.bar(p.index, p.clip(upper=0), color=c("BESS Charge"),
            alpha=0.85, width=0.8, label="Charge")
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.set_ylabel("MW")
    ax1.legend(fontsize=9)

    ax2.fill_between(soc.index, soc / 1e3, alpha=0.5, color="#4cc9f0")
    ax2.plot(soc.index, soc / 1e3, color="#4361ee", linewidth=1)
    ax2.set_ylabel("GWh")

    for ax in (ax1, ax2):
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

    plt.tight_layout()
    plt.savefig("results/annual_bess_dispatch.png", dpi=150)
    plt.close()
    print("Saved: results/annual_bess_dispatch.png")


def plot_line_loading(n_base, n_2030):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for ax, n, label, col in [
        (ax1, n_base, "Baseline",        "#95a5a6"),
        (ax2, n_2030, "2030 Projection", "#e67e22"),
    ]:
        vals = (n.lines_t.p0.abs() / n.lines.s_nom).values.flatten()
        vals = np.sort(vals[~np.isnan(vals)])[::-1]
        ax.plot(vals, color=col, linewidth=1)
        ax.axhline(1.0, color="#c0392b", linestyle="--", linewidth=1.2, label="s_nom limit")
        ax.axhline(0.9, color="#e67e22", linestyle=":",  linewidth=1.0, label=">90%")
        ax.text(0.98, 0.95,
                f">100%: {(vals > 1.0).mean()*100:.1f}%\n>90%: {(vals > 0.9).mean()*100:.1f}%",
                transform=ax.transAxes, ha="right", va="top", fontsize=9)
        ax.set_xlabel("Hours (sorted)")
        ax.set_ylabel("Loading (p.u.)")
        ax.set_title(label)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("results/annual_line_loading.png", dpi=150)
    plt.close()
    print("Saved: results/annual_line_loading.png")


def plot_re_share(n_base, n_2030):
    re_groups = {
        "solar + solar hsat":        ["solar", "solar hsat", "solar-hsat"],
        "onwind":                    ["onwind"],
        "offshore wind":             ["offwind-ac", "offwind-dc", "offwind-float"],
        "geothermal + biomass + ror":["geothermal", "biomass", "ror"],
    }
    x, w = np.arange(12), 0.35
    fig, ax = plt.subplots(figsize=(13, 5))

    for n, label, shift in [(n_base, "Baseline", -w/2), (n_2030, "2030 Proj.", w/2)]:
        bottom = np.zeros(12)
        for grp, carriers in re_groups.items():
            idx = n.generators.index[n.generators.carrier.isin(carriers)]
            if idx.empty:
                continue
            vals = np.array([
                n.generators_t.p.loc[n.snapshots.month == m, idx].sum().sum()
                / n.loads_t.p_set.loc[n.snapshots.month == m].sum().sum() * 100
                for m in range(1, 13)
            ])
            ax.bar(x + shift, vals, w, bottom=bottom,
                   color=c(grp), edgecolor="white",
                   label=grp if shift == -w/2 else "_nolegend_")
            bottom += vals
        for i, tot in enumerate(bottom):
            ax.text(x[i] + shift, tot + 0.5, f"{tot:.0f}%",
                    ha="center", va="bottom", fontsize=7)

    ax.axhline(80, color="#c0392b", linestyle="--", linewidth=1, label="80% target")
    ax.set_xticks(x)
    ax.set_xticklabels(MONTH_NAMES)
    ax.set_ylabel("RE Share of Load (%)")
    ax.set_ylim(0, 120)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("results/annual_re_monthly.png", dpi=150)
    plt.close()
    print("Saved: results/annual_re_monthly.png")


def plot_bess_monthly():
    path = "networks/bess_capacity_per_month.csv"
    if not os.path.exists(path):
        print(f"  {path} not found, skipping.")
        return
    df = pd.read_csv(path, index_col="month")
    x, w = np.arange(len(df)), 0.35
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x - w/2, df["baseline_GW"],  w, label="Baseline",        color="#95a5a6", edgecolor="white")
    ax.bar(x + w/2, df["proj_2030_GW"], w, label="2030 Projection", color="#2ec4b6", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(df.index)
    ax.set_ylabel("Optimal BESS Capacity (GW)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("results/annual_bess_monthly.png", dpi=150)
    plt.close()
    print("Saved: results/annual_bess_monthly.png")


def plot_capacity_mix(n_base, n_2030):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for ax, n, label in [(ax1, n_base, "Baseline"), (ax2, n_2030, "2030 Projection")]:
        cap = n.generators.groupby("carrier").p_nom_opt.sum() / 1e3
        cap = cap[cap > 0].sort_values()
        cap.plot.barh(ax=ax, color=[c(x) for x in cap.index], edgecolor="white")
        for i, v in enumerate(cap):
            ax.text(v + 0.1, i, f"{v:.1f}", va="center", fontsize=8)
        ax.set_xlabel("GW")
        ax.set_title(label)

    plt.tight_layout()
    plt.savefig("results/annual_capacity_mix.png", dpi=150)
    plt.close()
    print("Saved: results/annual_capacity_mix.png")


def save_metrics(n_base, n_2030):
    rows = []
    for label, n in [("Baseline", n_base), ("2030 Projection", n_2030)]:
        cap_gw, cap_gwh, disch, cycles = bess_stats(n)
        rows.append({
            "Scenario":                   label,
            "RE Share (%)":               round(re_share(n), 1),
            "Curtailment (%)":            round(curtailment_pct(n), 2),
            "BESS Capacity (GW)":         round(cap_gw, 2),
            "BESS Energy (GWh)":          round(cap_gwh, 2),
            "BESS Total Discharge (GWh)": round(disch, 1),
            "BESS Full Cycles":           round(cycles, 0),
            "Avg LMP (EUR/MWh)":          round(n.buses_t.marginal_price.mean().mean(), 2),
            "Negative Price Hours":       int((n.buses_t.marginal_price < 0).sum().sum()),
            "System Cost (MEUR)":         round(n.objective / 1e6, 1),
        })
    df = pd.DataFrame(rows).set_index("Scenario")
    df.to_csv("results/metrics_annual.csv")
    print("\nSaved: results/metrics_annual.csv")
    print(df.T.to_string())


# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1. Inputs
    carrier_growth = load_growth_factors("pypsa_capacity_2030(3).csv")
    capital_cost, capital_cost_b = calculate_capital_costs()
    n_full = pypsa.Network("networks/base_s_50_elec_.nc")

    # 2. Simulate
    run_monthly_simulations(n_full, carrier_growth, capital_cost, capital_cost_b)
    combine_monthly_networks(n_full)

    # 3. Analyse & plot
    print("\nLoading annual networks for analysis...")
    n_base = pypsa.Network("networks/combined_baseline_annual.nc")
    n_2030 = pypsa.Network("networks/combined_2030_annual.nc")

    print_metrics("Baseline",        n_base)
    print_metrics("2030 Projection", n_2030)

    plot_dispatch(n_base, n_2030)
    plot_lmp(n_base, n_2030)
    plot_lmp_distribution(n_base, n_2030)
    plot_bess(n_2030)
    plot_line_loading(n_base, n_2030)
    plot_re_share(n_base, n_2030)
    plot_bess_monthly()
    plot_capacity_mix(n_base, n_2030)
    save_metrics(n_base, n_2030)


    print("\nAll done.")