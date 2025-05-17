

# -*- coding: utf-8 -*-

"""
Created on Sat May 17 09:41:14 2025
@author: SuPAR Group (updated)
"""



import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import pydeck as pdk
from scipy.stats import gaussian_kde



st.set_page_config(page_title="Paver Temperature Analysis", layout="wide")
st.title("Paver Temperature Analysis Dashboard")

# Sidebar inputs
uploaded_file = st.file_uploader("Upload your Paving file.xlsx file", type=["xlsx"])
thresholds = st.sidebar.slider("Choose Paving Width (m)", -6.5, 6.5, (-2.0, 2.0), 0.25)
lower_threshold, upper_threshold = thresholds
st.sidebar.markdown(f"**Selected thresholds:** {lower_threshold} m to {upper_threshold} m")
show_cold = st.sidebar.checkbox("Show Cold Spots (<120°C)", value=True)
show_risk = st.sidebar.checkbox("Show Risk Spots (<90% avg)", value=True)

@st.cache_data
def load_excel(file):
    return pd.read_excel(file, header=0)

@st.cache_data
def clean_and_prepare(df, lower_threshold, upper_threshold):
    fixed_cols = ["Time", "Moving distance", "Latitude", "Longitude",
                  "Signal type", "width left", "width right"]
    right = [round(x, 2) for x in np.arange(6.5, -0.001, -0.25)]
    left_neg = [-round(x, 2) for x in np.arange(0.25, 6.251, 0.25)]
    new_header = fixed_cols + right + left_neg
    df.columns = new_header

    df['Time_clean'] = df['Time'].str.replace(r'\s*UTC\s*\+\s*', '+', regex=True)
    df['Time'] = pd.to_datetime(df['Time_clean'], dayfirst=True, utc=True).dt.strftime('%H:%M:%S')
    df.drop(columns='Time_clean', inplace=True)

    widths = sorted(c for c in df.columns if isinstance(c, float))
    mean_temps = df[widths].mean()
    zero_widths = mean_temps[mean_temps == 0.0].index.tolist()
    df.drop(columns=zero_widths, inplace=True)
    widths = [w for w in widths if w not in zero_widths]

    to_drop = [w for w in widths if w >= upper_threshold or w <= lower_threshold]
    df_trimmed = df.drop(columns=to_drop)
    widths_2 = [w for w in widths if w not in to_drop]
    return df_trimmed, widths_2, to_drop

if uploaded_file is not None:
    df = load_excel(uploaded_file)
    df_trimmed, widths_2, dropped = clean_and_prepare(df, lower_threshold, upper_threshold)
    #st.write(f"Dropped width columns based on thresholds: {dropped}")

    df_trimmed['Time_dt'] = pd.to_datetime(df_trimmed['Time'], format='%H:%M:%S')
    df_trimmed['lat_diff'] = df_trimmed['Latitude'].diff().fillna(0)
    df_trimmed['lon_diff'] = df_trimmed['Longitude'].diff().fillna(0)
    df_trimmed['is_stop'] = (df_trimmed['lat_diff'] == 0) & (df_trimmed['lon_diff'] == 0)
    df_trimmed['stop_group'] = (df_trimmed['is_stop'] != df_trimmed['is_stop'].shift()).cumsum()
    stop_groups = df_trimmed[df_trimmed['is_stop']].groupby('stop_group')
    stops = stop_groups.agg(
        start=('Time_dt', 'first'),
        end=('Time_dt', 'last'),
        moving_dist=('Moving distance', 'first')
    )
    stops['duration'] = stops['end'] - stops['start']
    total_stop_time = stops['duration'].sum()

    Y2 = df_trimmed['Moving distance'].values
    Z2 = df_trimmed[widths_2].values

    
    st.subheader("Temperature Heatmap with Stop Indicators from Temperature Drops")

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    pcm = ax1.pcolormesh(widths_2, Y2, Z2, shading='auto')
    ax1.set_xlabel("Paving width (m)")
    ax1.set_ylabel("Moving distance (m)")
    
    #  Compute average temperature per row (across paving width)
    avg_temp_row = np.mean(Z2, axis=1)
    
    #  Define drop threshold (e.g., large sudden temperature dips)
    drop_threshold = 30  # adjust as needed
    drop_indices = np.where(np.diff(avg_temp_row) < -drop_threshold)[0]
    
    #  Plot vertical stop lines based on detected drops
    for i in drop_indices:
        y_pos = Y2[i]
        ax1.axhline(y=y_pos, color='red', linestyle='--', linewidth=1.5, label='Detected Stop' if i == drop_indices[0] else "")
    
    
    # Set plot title and colorbar
    ax1.set_title(f"Temperature Map\nTotal Stop Time: {total_stop_time}")
    fig1.colorbar(pcm, ax=ax1, label="Temperature [°C]")
    ax1.legend(loc='upper right')
    st.pyplot(fig1)

    
    #st.subheader("Temperature Heatmap with Stop Lines")
    #fig1, ax1 = plt.subplots(figsize=(10, 6))
    #pcm = ax1.pcolormesh(widths_2, Y2, Z2, shading='auto')
    #ax1.set_xlabel("Paving width (m)")
    #ax1.set_ylabel("Moving distance (m)")
#for _, row in stops.iterrows():
    #y = row['moving_dist']
    #duration_str = str(row['duration'])
    #ax1.text(widths_2[-1], y, f"{duration_str}", va='center', fontsize=6, color='red')
    ##for md in stops['moving_dist']:
        ##ax1.hlines(md, widths_2[0], widths_2[-1], linestyles='--', linewidth=5)
    #ax1.set_title(f"Temperature Map\nTotal Stop Time: {total_stop_time}")
    #fig1.colorbar(pcm, ax=ax1, label="Temperature [°C]")
    #st.pyplot(fig1)

    Z3 = Z2.copy()
    if show_cold:
        st.subheader("Cold Spots (T < 120°C)")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        rows, cols = np.where(Z3 < 120)
        cold_x = [widths_2[c] for c in cols]
        cold_y = [Y2[r] for r in rows]
        pcm2 = ax2.pcolormesh(widths_2, Y2, Z3, shading='auto')
        ax2.scatter(cold_x, cold_y, marker='o', label='Cold spots (T < 120°C)')
        ax2.set_title("Paving Temperature Map with Cold Spots")
        ax2.set_xlabel("Paving width (m)")
        ax2.set_ylabel("Moving distance (m)")
        ax2.legend()
        fig2.colorbar(pcm2, ax=ax2, label="Temperature [°C]")
        st.pyplot(fig2)

    if show_risk:
        st.subheader("Risk Areas (T < 90% of Avg)")
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        avg_temp = Z3.mean()
        risk_temp = 0.9 * avg_temp
        rows_r, cols_r = np.where(Z3 < risk_temp)
        risk_x = [widths_2[c] for c in cols_r]
        risk_y = [Y2[r] for r in rows_r]
        pcm3 = ax3.pcolormesh(widths_2, Y2, Z3, shading='auto')
        ax3.scatter(risk_x, risk_y, marker='x', label=f'Risk spots (T < {risk_temp:.1f}°C)')
        ax3.set_title(f"Paving Temperature Map with Risk Areas")
        ax3.set_xlabel("Paving width (m)")
        ax3.set_ylabel("Moving distance (m)")
        ax3.legend()
        fig3.colorbar(pcm3, ax=ax3, label="Temperature [°C]")
        st.pyplot(fig3)

    st.subheader("Temperature & TSI Profiles")
    temps = df_trimmed[widths_2].values
    df_trimmed['TSI_C'] = temps.max(axis=1) - temps.mean(axis=1)
    avg_tsi = df_trimmed['TSI_C'].mean()
    tsi_cat = ('Low' if avg_tsi <= 5 else
               'Moderate' if avg_tsi <= 20 else 'High')
    fig4, (ax4, ax5) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    pcm4 = ax4.pcolormesh(widths_2, Y2, temps, shading='auto')
    ax4.set_title("Temperature [°C]")
    ax4.set_xlabel("Paving width (m)")
    ax4.set_ylabel("Moving distance (m)")
    fig4.colorbar(pcm4, ax=ax4)
    Z_tsi = np.tile(df_trimmed['TSI_C'].values[:, None], (1, len(widths_2)))
    pcm5 = ax5.pcolormesh(widths_2, Y2, Z_tsi, shading='auto', cmap='inferno')
    ax5.set_title("TSI Profile [°C]")
    ax5.set_xlabel("Paving width (m)")
    fig4.suptitle(f"Average TSI: {avg_tsi:.2f}°C → {tsi_cat} Segregation")
    fig4.colorbar(pcm5, ax=ax5)
    st.pyplot(fig4)

    st.subheader("Differential Range Statistics (DRS)")
    temps_all = temps.flatten()
    T10_C = np.percentile(temps_all, 10)
    T90_C = np.percentile(temps_all, 90)
    DRS_C = T90_C - T10_C
    DRS_F = DRS_C * 9/5 + 32
    drs_severity = ('Low' if DRS_C <= 5 else
                    'Moderate' if DRS_C <= 10 else 'High')
    summary = pd.DataFrame({
        'Statistic': ['T10 (°C)', 'T90 (°C)', 'DRS (°C)', 'DRS (°F)', 'Severity'],
        'Value': [T10_C, T90_C, DRS_C, DRS_F, drs_severity]
    })
    st.text(summary.to_string(index=False))

    st.subheader("Distribution of Differential Range Statistics (Row-Wise)")
    row_drs = df_trimmed[widths_2].apply(lambda row: np.percentile(row, 90) - np.percentile(row, 10), axis=1)
    df_trimmed['DRS_row'] = row_drs
    

    fig_drs_hist, ax_drs_hist = plt.subplots(figsize=(10, 5))
    
    # Histogram
    counts, bins, _ = ax_drs_hist.hist(row_drs, bins=30, color='skyblue', edgecolor='black', density=True, alpha=0.6)
    
    # KDE line
    kde = gaussian_kde(row_drs)
    x_vals = np.linspace(min(row_drs), max(row_drs), 500)
    ax_drs_hist.plot(x_vals, kde(x_vals), color='darkred', linewidth=2, label='Density Curve')
    
    ax_drs_hist.set_title("Histogram of Row-Wise DRS Values with KDE")
    ax_drs_hist.set_xlabel("DRS per Row (°C)")
    ax_drs_hist.set_ylabel("Density")
    ax_drs_hist.legend()
    st.pyplot(fig_drs_hist)

    st.subheader("Average Temperature Along the Paving Width per Moving Distance")

    # Compute average temperature per row (i.e., across paving width)
    df_trimmed["Avg_Temp_Row"] = df_trimmed[widths_2].mean(axis=1)
    
    # Prepare dataframe for plotting
    avg_temp_moving = df_trimmed[["Moving distance", "Avg_Temp_Row"]].copy()
    
    # Sort by moving distance
    avg_temp_moving = avg_temp_moving.sort_values("Moving distance")
    
    # Plot line chart using Plotly
    fig_avg_moving = px.line(
        avg_temp_moving,
        x="Moving distance",
        y="Avg_Temp_Row",
        title="Average Temperature Along Paving Width (per Moving Distance)",
        labels={"Moving distance": "Moving Distance (m)", "Avg_Temp_Row": "Average Temperature (°C)"},
    )
    
    st.plotly_chart(fig_avg_moving, use_container_width=True)


    

    
    
    # avg_temp_per_width = df_trimmed[widths_2].mean().reset_index()
    # avg_temp_per_width.columns = ["Paving Width (m)", "Average Temperature (°C)"]
    
    # # Sort widths numerically for better plot layout
    # avg_temp_per_width = avg_temp_per_width.sort_values("Paving Width (m)")
    
    # # Plot bar chart using Plotly
    # fig_avg_width = px.bar(
    #     avg_temp_per_width,
    #     x="Paving Width (m)",
    #     y="Average Temperature (°C)",
    #     title="Average Temperature Across Paving Width",
    #     text_auto=".2f",
    #     labels={"Average Temperature (°C)": "Avg Temp (°C)"},
    # )
    
    # st.plotly_chart(fig_avg_width, use_container_width=True)


    st.subheader("Interactive GPS Track with Tooltips")

    # Sample and clean GPS data
    gps_data = df_trimmed[['Latitude', 'Longitude', 'Time']].dropna().iloc[::10]
    gps_data = gps_data[(gps_data['Latitude'] != 0) & (gps_data['Longitude'] != 0)]
    
    if not gps_data.empty and len(gps_data) >= 2:
        # Create coordinates column for tooltips and path
        gps_data = gps_data.copy()
        gps_data["coordinates"] = gps_data[["Longitude", "Latitude"]].values.tolist()
    
        # Line path for the full route
        gps_path_df = pd.DataFrame({
            "path": [gps_data["coordinates"].tolist()],
            "color": [[255, 0, 0]]  # Red line
        })
    
        # Points layer for interactive tooltips
        point_layer = pdk.Layer(
            "ScatterplotLayer",
            data=gps_data,
            get_position="coordinates",
            get_color=[0, 0, 255],
            get_radius=10,
            pickable=True,
            tooltip=True
        )
    
        # Line layer showing the paver path
        line_layer = pdk.Layer(
            "LineLayer",
            data=gps_path_df,
            get_path="path",
            get_width=4,
            get_color="color",
            pickable=False
        )
    
        # View settings
        view_state = pdk.ViewState(
            latitude=gps_data["Latitude"].mean(),
            longitude=gps_data["Longitude"].mean(),
            zoom=15,
            pitch=0
        )
    
        # Tooltip configuration
        tooltip = {
            "html": "<b>Time:</b> {Time}<br><b>Lat:</b> {Latitude}<br><b>Lon:</b> {Longitude}",
            "style": {"backgroundColor": "steelblue", "color": "white"}
        }
    
        # Render map with both layers
        st.pydeck_chart(
            pdk.Deck(
                layers=[line_layer, point_layer],
                initial_view_state=view_state,
                tooltip=tooltip
            )
        )
    else:
        st.info("No valid GPS data to display or not enough points for a path.")




    # =========================
    # Script-style Results Summary
    # =========================
    st.subheader("📄 Script Summary of Analysis Results")

    summary_lines = [
        "COLD & RISK SPOTS ",
        f"- Cold Spots Enabled: {'Yes' if show_cold else 'No'}",
        f"- Risk Areas Enabled: {'Yes' if show_risk else 'No'}",
        "",
        "TSI (Thermal Segregation Index)",
        f"- Average TSI: {avg_tsi:.2f} °C",
        f"- TSI Severity Category: {tsi_cat}",
        "",
        "DRS (Differential Range Statistics)",
        f"- T10 (°C): {T10_C:.2f}",
        f"- T90 (°C): {T90_C:.2f}",
        f"- DRS (°C): {DRS_C:.2f}",
        f"- DRS (°F): {DRS_F:.2f}",
        f"- DRS Severity: {drs_severity}",
        "",
        "GPS TRACK ",
        f"- Valid GPS Points: {len(gps_data)}" if not gps_data.empty else "- No valid GPS data available",
        "",
        ]

# Display as formatted text
    st.text("\n".join(summary_lines))
    
    
    summary_text = "\n".join(summary_lines)
    st.download_button("📥 Download Summary as Text", summary_text.encode(), file_name="paver_analysis_summary.txt", mime="text/plain")

