import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pmdarima import auto_arima
import warnings

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────
# 1. DATA INGESTION & CAPACITY PREPARATION
# ─────────────────────────────────────────────────────────────────────
def load_and_prepare_capacity(years=range(2019, 2026), base_path='net_generation_capacity_{year}.csv'):
    """Loads annual CSV files, filters for Germany (DE), and pivots values."""
    frames = []
    for year in years:
        try:
            df = pd.read_csv(base_path.format(year=year), sep=None, engine='python')
            df.columns = df.columns.str.strip()
            frames.append(df)
        except Exception as e:
            print(f"Skip {year}: {e}")

    data = pd.concat(frames, ignore_index=True)
    de_df = data[data['Country'] == 'DE'].copy()

    capacity = de_df.pivot_table(
        index='Year', columns='Category',
        values='ProvidedValue', aggfunc='sum'
    ).fillna(0)

    capacity.index = pd.to_datetime(capacity.index, format='%Y')
    capacity.sort_index(inplace=True)
    return capacity


# ─────────────────────────────────────────────────────────────────────
# 2. GENERATION CAPACITY FORECAST ENGINE
# ─────────────────────────────────────────────────────────────────────
def forecast_generation_capacity(capacity, n_steps=5):
    """Fits Holt-Winters and ARIMA models across generation technologies."""
    forecast_index = pd.date_range(
        start=capacity.index[-1] + pd.DateOffset(years=1),
        periods=n_steps,
        freq='YS'
    )
    results = {}

    for tech in capacity.columns:
        series = capacity[tech].copy()

        if series.max() == 0:
            results[tech] = {
                'history': series,
                'hw':      pd.Series(0.0, index=forecast_index),
                'arima':   pd.Series(0.0, index=forecast_index),
            }
            continue

        try:
            hw_fit = ExponentialSmoothing(
                series, trend='add', damped_trend=True, seasonal=None, initialization_method='estimated'
            ).fit(optimized=True)
            hw_fc = hw_fit.forecast(n_steps)
            hw_fc.index = forecast_index
            hw_fc = hw_fc.clip(lower=0)
        except Exception as e:
            print(f"  HW failed for {tech}: {e}")
            hw_fc = pd.Series(series.iloc[-1], index=forecast_index)

        try:
            arima_fit = auto_arima(
                series, d=1, seasonal=False, stepwise=True,
                suppress_warnings=True, error_action='ignore',
                max_p=3, max_q=3, max_d=2
            )
            arima_fc = arima_fit.predict(n_periods=n_steps)
            arima_fc = pd.Series(np.maximum(0, arima_fc), index=forecast_index)
        except Exception as e:
            print(f"  ARIMA failed for {tech}: {e}")
            arima_fc = pd.Series(series.iloc[-1], index=forecast_index)

        results[tech] = {
            'history': series,
            'hw':      hw_fc,
            'arima':   arima_fc,
        }
        print(f"  {tech:<40}  HW 2030: {hw_fc.iloc[-1]:.1f} GW  |  ARIMA 2030: {arima_fc.iloc[-1]:.1f} GW")

    return results, forecast_index


# ─────────────────────────────────────────────────────────────────────
# 3. ELECTRICITY DEMAND FORECAST ENGINE
# ─────────────────────────────────────────────────────────────────────
def forecast_electricity_demand(n_steps=5):
    """Extracts historical demand data and fits forecasting models using Code 1 config."""
    demand_data = {
        2015: 520.6, 2016: 548.4, 2017: 538.7, 2018: 538.1, 2019: 497.4,
        2020: 485.4, 2021: 504.5, 2022: 482.7, 2023: 458.0, 2024: 464.7,
        2025: 461.2,
    }

    series = pd.Series(demand_data)
    series.index = pd.to_datetime(series.index, format='%Y')
    series.name = 'DE Net Consumption (TWh)'

    hw_fit = ExponentialSmoothing(
        series, trend='add', damped_trend=True, seasonal=None, initialization_method='estimated'
    ).fit(optimized=True)

    demand_forecast_index = pd.date_range(
        start=series.index[-1] + pd.DateOffset(years=1),
        periods=n_steps, freq='YS'
    )
    hw_fc = hw_fit.forecast(n_steps)
    hw_fc.index = demand_forecast_index

    arima_fit = auto_arima(
        series, d=1, seasonal=False, stepwise=True,
        suppress_warnings=True, error_action='ignore',
        max_p=3, max_q=3, max_d=2
    )
    print(f"\nARIMA demand order selected: {arima_fit.order}")

    arima_fc = arima_fit.predict(n_periods=n_steps)
    arima_fc = pd.Series(arima_fc, index=demand_forecast_index)

    avg_fc = (hw_fc + arima_fc) / 2

    print(f"\n{'Year':<8} {'HW (TWh)':>10} {'ARIMA (TWh)':>12} {'Avg (TWh)':>11}")
    print("─" * 44)
    for ts in demand_forecast_index:
        print(f"  {ts.year:<6} {hw_fc[ts]:>10.1f} {arima_fc[ts]:>12.1f} {avg_fc[ts]:>11.1f}")

    print(f"\n  2030 HW forecast  : {hw_fc.iloc[-1]:.1f} TWh")
    print(f"  2030 ARIMA forecast: {arima_fc.iloc[-1]:.1f} TWh")
    print(f"  2030 Average       : {avg_fc.iloc[-1]:.1f} TWh")

    return series, hw_fc, arima_fc, avg_fc, demand_forecast_index


# ─────────────────────────────────────────────────────────────────────
# 4. PLOTTING VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
import numpy as np

def generate_plots(capacity_results, demand_series, demand_hw, demand_arima, demand_avg, cap_index, dem_index):
    SKIP = {'Energy storage', 'Geothermal', 'Fossil Coal-derived gas',
            'Other', 'Hydro Pumped Storage', 'Hydro Water Reservoir'}
    plot_techs = [t for t in capacity_results if t not in SKIP]

    NCOLS = 3
    nrows = (len(plot_techs) + NCOLS - 1) // NCOLS

    C_HIST  = '#2c3e50'
    C_HW    = '#27ae60'
    C_ARIMA = '#e74c3c'
    C_AVG   = '#8e44ad'

    ROW_H = 3.6
    fig, axes = plt.subplots(nrows, NCOLS,
                             figsize=(16, nrows * ROW_H),
                             constrained_layout=False)
    axes = axes.flatten()

    for i, tech in enumerate(plot_techs):
        ax = axes[i]
        res = capacity_results[tech]
        s, hw, ar = res['history'], res['hw'], res['arima']
        avg = (hw + ar) / 2

        ax.plot(s.index, s.values,
                color=C_HIST, lw=2, marker='o', ms=3, label='Historical')

        for series, color, label in [(hw, C_HW, 'Holt-Winters'),
                                     (ar, C_ARIMA, 'ARIMA'),
                                     (avg, C_AVG, 'Average')]:
            ax.plot(series.index, series.values,
                    color=color, lw=2, ls='--', marker=None, label=label)
            ax.plot([s.index[-1], series.index[0]],
                    [s.values[-1], series.values[0]],
                    color=color, lw=1.1, ls='--', alpha=0.3)

        ax.plot(avg.index[-1], avg.values[-1],
                marker='*', ms=14, color=C_AVG, zorder=5)

        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        sub_handles = [
            Line2D([0], [0], color=C_HIST,  lw=2, marker='o', ms=5, label='Historical'),
            Line2D([0], [0], color=C_HW,    lw=2, ls='--',
                   label=f'HW  {hw.values[-1]:.0f} MW'),
            Line2D([0], [0], color=C_ARIMA, lw=2, ls='--',
                   label=f'ARIMA  {ar.values[-1]:.0f} MW'),
            Line2D([0], [0], color=C_AVG,   lw=2, ls='--',
                   label=f'Avg  {avg.values[-1]:.0f} MW'),
        ]
        ax.legend(handles=sub_handles,
                  fontsize=9, frameon=True,
                  facecolor='white', edgecolor='#dfe6e9', framealpha=0.9,
                  loc='best', handlelength=1.6)

        ax.axvspan(hw.index[0], hw.index[-1], alpha=0.04, color='#7f8c8d')

        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.tick_params(axis='x', labelsize=11, rotation=45)
        ax.tick_params(axis='y', labelsize=11)
        ax.set_title(f'{tech}  (MW)', fontsize=13, fontweight='bold', pad=5)
        ax.grid(axis='both', ls=':', alpha=0.45, color='#b2bec3')
        ax.spines[['top', 'right']].set_visible(False)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)


    plt.subplots_adjust(wspace=0.14, hspace=0.35,
                        bottom=0.04, top=0.97,
                        left=0.04, right=0.97)
    save_path = 'germany_capacity_hw_arima_2030.png'
    plt.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_path}")

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(demand_series.index, demand_series.values, color='#2c3e50', lw=2.5, marker='o', ms=6, label='Historical (ENTSO-E)')

    for fc, color, label, marker in [(demand_hw, '#27ae60', 'Holt-Winters', 's'), (demand_arima, '#e74c3c', 'ARIMA', '^'), (demand_avg, '#8e44ad', 'Average', 'D')]:
        ax.plot(fc.index, fc.values, color=color, lw=2, marker=marker, ms=5, ls='--', label=label)
        ax.plot([demand_series.index[-1], fc.index[0]], [demand_series.values[-1], fc.values[0]], color=color, lw=1.5, ls='--', alpha=0.4)

        ax.annotate(f'{avg.values[-1]:.0f}',
            xy=(avg.index[-1], avg.values[-1]),
            xytext=(-30, 0), textcoords='offset points',  
            color=C_AVG, fontsize=10, fontweight='bold',
            va='center', ha='right', clip_on=False)

    ax.axvspan(dem_index[0], dem_index[-1], alpha=0.06, color='grey')
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    ax.tick_params(axis='x', rotation=45, labelsize=12)
    ax.tick_params(axis='y', labelsize=12)

    ax.set_ylabel('TWh', fontsize=14)

    ax.set_ylim(400, 600)
    ax.grid(axis='both', ls=':', alpha=0.4)

    ax.legend(fontsize=12)

    plt.tight_layout()
    plt.show()
    print(f"\n  Use this in your 80% calculation: DEMAND_2030_TWH = {demand_avg.iloc[-1]:.1f}")


# ─────────────────────────────────────────────────────────────────────
# 5. DATA EXPORT FRAMEWORK
# ─────────────────────────────────────────────────────────────────────
def export_pypsa_results(capacity_results, demand_series, demand_hw, demand_arima, demand_avg):
    """Combines generation capacity matrices and demand data into the PyPSA CSV structure."""
    PYPSA_CARRIER_MAP = {
        'Solar':                               'solar',
        'Wind Onshore':                        'onwind',
        'Wind Offshore':                       'offwind',
        'Fossil Gas':                          'gas',
        'Fossil Brown coal/Lignite':           'lignite',
        'Fossil Hard coal':                    'coal',
        'Fossil Oil':                          'oil',
        'Nuclear':                             'nuclear',
        'Hydro Run-of-river and poundage':     'ror',
        'Hydro Water Reservoir':               'hydro',
        'Hydro Pumped Storage':                'PHS',
        'Biomass':                             'biomass',
        'Waste':                             'waste',
    }

    rows = []
    # Process capacities
    for entsoe_name, pypsa_carrier in PYPSA_CARRIER_MAP.items():
        if entsoe_name not in capacity_results:
            print(f"  ✗ {entsoe_name} not found in results — skipping")
            continue

        res = capacity_results[entsoe_name]
        hw_val = float(res['hw'].iloc[-1])
        ar_val = float(res['arima'].iloc[-1])
        avg = (hw_val + ar_val) / 2

        rows.append({
            'pypsa_carrier':     pypsa_carrier,
            'entsoe_category':   entsoe_name,
            'capacity_2025_GW':  round(float(res['history'].iloc[-1])/1000, 2),
            'hw_2030_GW':        round(hw_val/1000, 2),
            'arima_2030_GW':     round(ar_val/1000, 2),
            'avg_2030_GW':       round(avg/1000, 2),
            'hw_2030_MW':        round(hw_val, 1),
            'arima_2030_MW':     round(ar_val, 1),
            'avg_2030_MW':       round(avg, 1),
        })

    rows.append({
        'pypsa_carrier':     'load',
        'entsoe_category':   'Electricity Net Consumption',
        'capacity_2025_GW':  round(float(demand_series.iloc[-1]), 1),
        'hw_2030_GW':        round(float(demand_hw.iloc[-1]), 1),
        'arima_2030_GW':     round(float(demand_arima.iloc[-1]), 1),
        'avg_2030_GW':       round(float(demand_avg.iloc[-1]), 1),
        'hw_2030_MW':        round(float(demand_hw.iloc[-1]), 1),
        'arima_2030_MW':     round(float(demand_arima.iloc[-1]), 1),
        'avg_2030_MW':       round(float(demand_avg.iloc[-1]), 1),
    })

    pypsa_df = pd.DataFrame(rows).set_index('pypsa_carrier')
    pypsa_df.to_csv('pypsa_capacity_2030.csv')

    print("\n" + "="*65)
    print("PyPSA 2030 capacity predictions saved (including predicted load row)")
    print("="*65)
    print(pypsa_df[['capacity_2025_GW', 'hw_2030_GW', 'arima_2030_GW', 'avg_2030_GW']].to_string())


if __name__ == "__main__":

    # 1. Ingest Data
    capacity_df = load_and_prepare_capacity()

    # 2. Run Predictions
    cap_results, cap_index = forecast_generation_capacity(capacity_df)
    dem_series, dem_hw, dem_arima, dem_avg, dem_index = forecast_electricity_demand()

    # 3. Handle Plotting Output
    generate_plots(cap_results, dem_series, dem_hw, dem_arima, dem_avg, cap_index, dem_index)

    # 4. Save and Export Structured CSV Matrix
    export_pypsa_results(cap_results, dem_series, dem_hw, dem_arima, dem_avg)
