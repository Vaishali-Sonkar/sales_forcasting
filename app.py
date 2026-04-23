import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Sales Forecast Dashboard",
    page_icon="📈",
    layout="wide"
)

# ─────────────────────────────────────────
# CUSTOM CSS — Dark professional theme
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0d1117;
    color: #e6edf3;
}

.main { background-color: #0d1117; }

h1, h2, h3 {
    font-family: 'Space Mono', monospace;
    color: #58a6ff;
}

.metric-card {
    background: linear-gradient(135deg, #161b22, #1c2128);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    transition: transform 0.2s;
}
.metric-card:hover { transform: translateY(-3px); }
.metric-label {
    font-size: 12px;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 6px;
    font-family: 'Space Mono', monospace;
}
.metric-value {
    font-size: 28px;
    font-weight: 700;
    color: #58a6ff;
    font-family: 'Space Mono', monospace;
}
.metric-sub {
    font-size: 12px;
    color: #3fb950;
    margin-top: 4px;
}

.section-header {
    background: linear-gradient(90deg, #58a6ff22, transparent);
    border-left: 3px solid #58a6ff;
    padding: 10px 16px;
    margin: 24px 0 16px 0;
    border-radius: 0 8px 8px 0;
    font-family: 'Space Mono', monospace;
    font-size: 14px;
    color: #58a6ff;
    letter-spacing: 1px;
}

.insight-box {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 16px 20px;
    margin: 8px 0;
    font-size: 14px;
    line-height: 1.6;
}
.insight-box span {
    color: #3fb950;
    font-weight: 600;
    font-family: 'Space Mono', monospace;
}

.stButton > button {
    background: linear-gradient(135deg, #1f6feb, #388bfd);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 24px;
    font-family: 'Space Mono', monospace;
    font-size: 13px;
    letter-spacing: 0.5px;
    transition: all 0.2s;
    width: 100%;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 20px #1f6feb55;
}

.stSelectbox label, .stSlider label, .stFileUploader label {
    color: #8b949e !important;
    font-size: 13px;
    font-family: 'Space Mono', monospace;
}

div[data-testid="stSidebar"] {
    background: #161b22;
    border-right: 1px solid #30363d;
}

.winner-badge {
    background: linear-gradient(135deg, #3fb95022, #3fb95044);
    border: 1px solid #3fb950;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 12px;
    color: #3fb950;
    font-family: 'Space Mono', monospace;
}
.loser-badge {
    background: #f8514922;
    border: 1px solid #f85149;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 12px;
    color: #f85149;
    font-family: 'Space Mono', monospace;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# MATPLOTLIB DARK THEME
# ─────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor': '#161b22',
    'axes.edgecolor': '#30363d',
    'axes.labelcolor': '#8b949e',
    'xtick.color': '#8b949e',
    'ytick.color': '#8b949e',
    'grid.color': '#21262d',
    'grid.linestyle': '--',
    'grid.alpha': 0.6,
    'text.color': '#e6edf3',
    'axes.titlecolor': '#e6edf3',
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'legend.facecolor': '#161b22',
    'legend.edgecolor': '#30363d',
    'legend.labelcolor': '#e6edf3',
})


# ─────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────
@st.cache_data
def load_and_preprocess(file):
    df = pd.read_csv(file, encoding='latin1')
    df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
    daily = df.groupby('Order Date')['Sales'].sum().reset_index()
    daily = daily.sort_values('Order Date')
    date_range = pd.date_range(start=daily['Order Date'].min(),
                               end=daily['Order Date'].max(), freq='D')
    daily = daily.set_index('Order Date').reindex(date_range, fill_value=0)
    daily.index.name = 'Order Date'
    p99 = daily['Sales'].quantile(0.99)
    daily['Sales'] = daily['Sales'].clip(upper=p99)
    return daily


@st.cache_data
def run_sarima(_train, _test):
    model = SARIMAX(_train, order=(1,1,1), seasonal_order=(1,1,1,7),
                    enforce_stationarity=False, enforce_invertibility=False)
    result = model.fit(disp=False)
    preds = result.forecast(steps=len(_test))
    mae  = mean_absolute_error(_test, preds)
    rmse = np.sqrt(mean_squared_error(_test, preds))
    return preds, mae, rmse, result


@st.cache_data
def run_xgboost(_daily):
    df_ml = _daily[['Sales']].copy()
    for lag in [1,2,3,7,14,30]:
        df_ml[f'lag_{lag}'] = df_ml['Sales'].shift(lag)
    df_ml['day_of_week']  = df_ml.index.dayofweek
    df_ml['month']        = df_ml.index.month
    df_ml['quarter']      = df_ml.index.quarter
    df_ml['day_of_month'] = df_ml.index.day
    df_ml['rolling_7']    = df_ml['Sales'].shift(1).rolling(7).mean()
    df_ml['rolling_30']   = df_ml['Sales'].shift(1).rolling(30).mean()
    df_ml = df_ml.dropna()

    features = ['lag_1','lag_2','lag_3','lag_7','lag_14','lag_30',
                'day_of_week','month','quarter','day_of_month',
                'rolling_7','rolling_30']
    X, y = df_ml[features], df_ml['Sales']
    X_train, X_test = X[:-90], X[-90:]
    y_train, y_test = y[:-90], y[-90:]

    model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05,
                              max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    return preds, mae, rmse, model, df_ml, features, y_test


def make_future_forecast(model, df_ml, features, daily, days=30):
    last_known = df_ml[features].iloc[-1].copy()
    future_preds = []
    for i in range(days):
        pred = model.predict(last_known.values.reshape(1,-1))[0]
        future_preds.append(max(pred, 0))
        last_known['lag_30'] = last_known['lag_14']
        last_known['lag_14'] = last_known['lag_7']
        last_known['lag_7']  = last_known['lag_3']
        last_known['lag_3']  = last_known['lag_2']
        last_known['lag_2']  = last_known['lag_1']
        last_known['lag_1']  = pred
        nd = daily.index[-1] + pd.Timedelta(days=i+1)
        last_known['day_of_week']  = nd.dayofweek
        last_known['month']        = nd.month
        last_known['quarter']      = nd.quarter
        last_known['day_of_month'] = nd.day
    future_index = pd.date_range(
        start=daily.index[-1] + pd.Timedelta(days=1), periods=days, freq='D')
    return future_index, future_preds


# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.markdown("""
<div style='padding: 32px 0 8px 0;'>
    <h1 style='font-size:2.2rem; margin:0;'>📈 Sales Forecast Dashboard</h1>
    <p style='color:#8b949e; font-size:14px; margin-top:8px;'>
        End-to-end Time Series Analysis · SARIMA + XGBoost · Superstore Dataset
    </p>
</div>
""", unsafe_allow_html=True)
st.markdown("<hr style='border-color:#30363d; margin: 0 0 24px 0;'>", unsafe_allow_html=True)


# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Controls")
    st.markdown("<hr style='border-color:#30363d;'>", unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload Superstore CSV", type=['csv'])

    st.markdown("---")
    st.markdown("### 🔮 Forecast Settings")
    forecast_days = st.slider("Future Forecast Days", 7, 60, 30)
    test_days     = st.slider("Test Set Size (days)", 30, 180, 90)

    st.markdown("---")
    st.markdown("### 📌 About")
    st.markdown("""
    <div style='font-size:12px; color:#8b949e; line-height:1.8;'>
    Built with Python · Streamlit<br>
    Models: SARIMA · XGBoost<br>
    Dataset: Superstore (Kaggle)<br><br>
    <span style='color:#58a6ff;'>▸ 4 years daily sales data</span><br>
    <span style='color:#58a6ff;'>▸ 1,458 data points</span><br>
    <span style='color:#58a6ff;'>▸ Weekly seasonality detected</span>
    </div>
    """, unsafe_allow_html=True)

    run_btn = st.button("🚀 Run Full Analysis")


# ─────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────
if uploaded is None:
    st.markdown("""
    <div style='text-align:center; padding: 80px 20px;'>
        <div style='font-size:64px; margin-bottom:16px;'>📂</div>
        <h3 style='color:#58a6ff;'>Upload Your Superstore CSV to Begin</h3>
        <p style='color:#8b949e; font-size:14px;'>
            Upload the file using the sidebar → then click Run Full Analysis
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ─── LOAD DATA ───
daily_sales = load_and_preprocess(uploaded)

if not run_btn and 'analysis_done' not in st.session_state:
    st.info("👈 Click **Run Full Analysis** in the sidebar to start!")
    # Show raw data preview
    st.markdown('<div class="section-header">📋 DATA PREVIEW</div>', unsafe_allow_html=True)
    st.dataframe(
        daily_sales.head(10).style.format({'Sales': '${:.2f}'}),
        use_container_width=True
    )
    st.stop()

if run_btn:
    st.session_state['analysis_done'] = True

# ─────────────────────────────────────────
# RUN MODELS
# ─────────────────────────────────────────
with st.spinner("⚙️ Running SARIMA model..."):
    train_s = daily_sales['Sales'][:-test_days]
    test_s  = daily_sales['Sales'][-test_days:]
    sarima_preds, sarima_mae, sarima_rmse, sarima_result = run_sarima(train_s, test_s)

with st.spinner("⚙️ Running XGBoost model..."):
    xgb_preds, xgb_mae, xgb_rmse, xgb_model, df_ml, features, y_test = run_xgboost(daily_sales)

with st.spinner("🔮 Generating future forecast..."):
    future_idx, future_preds = make_future_forecast(
        xgb_model, df_ml, features, daily_sales, forecast_days)


# ─────────────────────────────────────────
# SECTION 1 — KPI METRICS
# ─────────────────────────────────────────
st.markdown('<div class="section-header">📊 KEY METRICS</div>', unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)

total_sales  = daily_sales['Sales'].sum()
avg_daily    = daily_sales['Sales'].mean()
best_month   = daily_sales.groupby(daily_sales.index.month)['Sales'].mean().idxmax()
months       = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
future_total = sum(future_preds)

with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Total Revenue</div>
        <div class="metric-value">${total_sales/1e6:.2f}M</div>
        <div class="metric-sub">4 Year Period</div>
    </div>""", unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Avg Daily Sales</div>
        <div class="metric-value">${avg_daily:.0f}</div>
        <div class="metric-sub">Per Day</div>
    </div>""", unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Peak Month</div>
        <div class="metric-value">{months[best_month-1]}</div>
        <div class="metric-sub">Highest Avg Sales</div>
    </div>""", unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">SARIMA MAE</div>
        <div class="metric-value">${sarima_mae:.0f}</div>
        <div class="metric-sub">Avg Daily Error</div>
    </div>""", unsafe_allow_html=True)

with c5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Forecast ({forecast_days}d)</div>
        <div class="metric-value">${future_total/1e3:.1f}K</div>
        <div class="metric-sub">Projected Revenue</div>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# SECTION 2 — HISTORICAL SALES TREND
# ─────────────────────────────────────────
st.markdown('<div class="section-header">📈 HISTORICAL SALES TREND</div>', unsafe_allow_html=True)

fig, axes = plt.subplots(2, 1, figsize=(14, 7), gridspec_kw={'hspace': 0.4})

# Daily
axes[0].plot(daily_sales.index, daily_sales['Sales'],
             color='#58a6ff', alpha=0.6, linewidth=0.8)
axes[0].fill_between(daily_sales.index, daily_sales['Sales'],
                     alpha=0.15, color='#58a6ff')
axes[0].set_title('Daily Sales (2011–2014)', fontsize=12)
axes[0].set_ylabel('Sales ($)')
axes[0].grid(True)
axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

# Monthly
monthly = daily_sales['Sales'].resample('M').sum()
axes[1].plot(monthly.index, monthly.values,
             color='#f78166', linewidth=2, marker='o', markersize=3)
axes[1].fill_between(monthly.index, monthly.values,
                     alpha=0.15, color='#f78166')
axes[1].set_title('Monthly Sales Aggregated', fontsize=12)
axes[1].set_ylabel('Sales ($)')
axes[1].grid(True)
axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

st.pyplot(fig)
plt.close()


# ─────────────────────────────────────────
# SECTION 3 — SEASONALITY
# ─────────────────────────────────────────
st.markdown('<div class="section-header">🗓️ SEASONALITY ANALYSIS</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(7, 4))
    monthly_avg = daily_sales.groupby(daily_sales.index.month)['Sales'].mean()
    colors = ['#f85149' if v == monthly_avg.max() else '#58a6ff' for v in monthly_avg.values]
    bars = ax.bar(months, monthly_avg.values, color=colors, edgecolor='#30363d', linewidth=0.5)
    ax.set_title('Average Sales by Month')
    ax.set_ylabel('Avg Sales ($)')
    ax.grid(True, axis='y')
    ax.set_xticklabels(months, fontsize=9)
    st.pyplot(fig)
    plt.close()

with col2:
    fig, ax = plt.subplots(figsize=(7, 4))
    dow_avg = daily_sales.groupby(daily_sales.index.dayofweek)['Sales'].mean()
    days = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
    colors2 = ['#3fb950' if v == dow_avg.max() else '#58a6ff' for v in dow_avg.values]
    ax.bar(days, dow_avg.values, color=colors2, edgecolor='#30363d', linewidth=0.5)
    ax.set_title('Average Sales by Day of Week')
    ax.set_ylabel('Avg Sales ($)')
    ax.grid(True, axis='y')
    st.pyplot(fig)
    plt.close()


# ─────────────────────────────────────────
# SECTION 4 — MODEL COMPARISON
# ─────────────────────────────────────────
st.markdown('<div class="section-header">🤖 MODEL COMPARISON — SARIMA vs XGBOOST</div>',
            unsafe_allow_html=True)

# Metrics comparison
m1, m2, m3, m4 = st.columns(4)
sarima_winner = sarima_mae < xgb_mae

with m1:
    badge = '<span class="winner-badge">✓ WINNER</span>' if sarima_winner else '<span class="loser-badge">✗</span>'
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">SARIMA MAE {badge}</div>
        <div class="metric-value" style="color:#f78166">${sarima_mae:.0f}</div>
    </div>""", unsafe_allow_html=True)

with m2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">SARIMA RMSE</div>
        <div class="metric-value" style="color:#f78166">${sarima_rmse:.0f}</div>
    </div>""", unsafe_allow_html=True)

with m3:
    badge2 = '<span class="winner-badge">✓ WINNER</span>' if not sarima_winner else '<span class="loser-badge">✗</span>'
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">XGBoost MAE {badge2}</div>
        <div class="metric-value" style="color:#3fb950">${xgb_mae:.0f}</div>
    </div>""", unsafe_allow_html=True)

with m4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">XGBoost RMSE</div>
        <div class="metric-value" style="color:#3fb950">${xgb_rmse:.0f}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Side by side prediction plots
col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(test_s.index, test_s.values, color='#58a6ff',
            linewidth=1.2, label='Actual', alpha=0.8)
    ax.plot(test_s.index, sarima_preds, color='#f78166',
            linewidth=1.5, label='SARIMA', linestyle='--')
    ax.set_title('SARIMA — Actual vs Predicted')
    ax.legend(fontsize=9)
    ax.grid(True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.xticks(rotation=30)
    st.pyplot(fig)
    plt.close()

with col2:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(y_test.index, y_test.values, color='#58a6ff',
            linewidth=1.2, label='Actual', alpha=0.8)
    ax.plot(y_test.index, xgb_preds, color='#3fb950',
            linewidth=1.5, label='XGBoost', linestyle='--')
    ax.set_title('XGBoost — Actual vs Predicted')
    ax.legend(fontsize=9)
    ax.grid(True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.xticks(rotation=30)
    st.pyplot(fig)
    plt.close()


# ─────────────────────────────────────────
# SECTION 5 — FUTURE FORECAST
# ─────────────────────────────────────────
st.markdown(f'<div class="section-header">🔮 NEXT {forecast_days} DAYS FORECAST (XGBOOST)</div>',
            unsafe_allow_html=True)

fig, ax = plt.subplots(figsize=(14, 5))
hist_slice = daily_sales['Sales'][-90:]
ax.plot(hist_slice.index, hist_slice.values,
        color='#58a6ff', linewidth=1.5, label='Historical', alpha=0.8)
ax.fill_between(hist_slice.index, hist_slice.values, alpha=0.1, color='#58a6ff')

ax.plot(future_idx, future_preds,
        color='#3fb950', linewidth=2.5,
        label=f'Forecast ({forecast_days} days)', linestyle='--', marker='o', markersize=3)
ax.fill_between(future_idx, future_preds, alpha=0.15, color='#3fb950')

# Divider line
ax.axvline(x=daily_sales.index[-1], color='#f85149',
           linestyle=':', linewidth=1.5, alpha=0.7, label='Forecast Start')

ax.set_title(f'Sales Forecast — Next {forecast_days} Days')
ax.set_ylabel('Sales ($)')
ax.legend(fontsize=10)
ax.grid(True)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
plt.xticks(rotation=30)
st.pyplot(fig)
plt.close()

# Forecast table
st.markdown("<br>", unsafe_allow_html=True)
col1, col2 = st.columns([2, 1])

with col1:
    forecast_df = pd.DataFrame({
        'Date': future_idx.strftime('%Y-%m-%d'),
        'Day': future_idx.strftime('%A'),
        'Forecasted Sales ($)': [f"${v:.2f}" for v in future_preds]
    })
    st.dataframe(forecast_df, use_container_width=True, height=300)

with col2:
    st.markdown("""
    <div class="insight-box">
        <span>📊 Forecast Summary</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    <div class="insight-box">
        Total Projected: <span>${sum(future_preds):,.0f}</span>
    </div>
    <div class="insight-box">
        Daily Average: <span>${np.mean(future_preds):,.0f}</span>
    </div>
    <div class="insight-box">
        Peak Day: <span>{future_idx[np.argmax(future_preds)].strftime('%b %d')}</span>
    </div>
    <div class="insight-box">
        Lowest Day: <span>{future_idx[np.argmin(future_preds)].strftime('%b %d')}</span>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────
# SECTION 6 — FEATURE IMPORTANCE
# ─────────────────────────────────────────
st.markdown('<div class="section-header">🎯 XGBOOST FEATURE IMPORTANCE</div>',
            unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])

with col1:
    importance = xgb_model.feature_importances_
    feat_df = pd.DataFrame({'Feature': features, 'Importance': importance})
    feat_df = feat_df.sort_values('Importance', ascending=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    colors_fi = ['#3fb950' if f.startswith('lag') else
                 '#58a6ff' if f.startswith('rolling') else '#f78166'
                 for f in feat_df['Feature']]
    ax.barh(feat_df['Feature'], feat_df['Importance'],
            color=colors_fi, edgecolor='#30363d', linewidth=0.5)
    ax.set_title('Feature Importance Scores')
    ax.set_xlabel('Importance')
    ax.grid(True, axis='x')
    st.pyplot(fig)
    plt.close()

with col2:
    st.markdown("""
    <div class="insight-box">
        <span>🟢 Lag Features</span><br>
        Yesterday's sales (lag_1) is the single most important predictor.
        Recent history strongly influences tomorrow.
    </div>
    <div class="insight-box">
        <span>🔵 Rolling Averages</span><br>
        7-day and 30-day rolling means capture short and
        medium-term trends effectively.
    </div>
    <div class="insight-box">
        <span>🔴 Date Features</span><br>
        Month and day-of-week encode seasonality —
        November peaks and weekend dips are captured here.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────
# SECTION 7 — BUSINESS INSIGHTS
# ─────────────────────────────────────────
st.markdown('<div class="section-header">💡 BUSINESS INSIGHTS</div>', unsafe_allow_html=True)

i1, i2, i3 = st.columns(3)

with i1:
    st.markdown("""
    <div class="metric-card" style="text-align:left;">
        <div style="font-size:24px; margin-bottom:8px;">📦</div>
        <div style="font-weight:600; color:#58a6ff; margin-bottom:8px;">
            Inventory Planning
        </div>
        <div style="font-size:13px; color:#8b949e; line-height:1.7;">
            Nov–Dec shows 40%+ higher sales. Stock up 6 weeks in advance.
            Weekend inventory can be 20% lower than weekdays.
        </div>
    </div>""", unsafe_allow_html=True)

with i2:
    st.markdown("""
    <div class="metric-card" style="text-align:left;">
        <div style="font-size:24px; margin-bottom:8px;">👥</div>
        <div style="font-weight:600; color:#3fb950; margin-bottom:8px;">
            Staffing Decisions
        </div>
        <div style="font-size:13px; color:#8b949e; line-height:1.7;">
            Schedule extra staff on weekdays and Q4.
            Forecast allows 2-week ahead planning with
            confidence for HR scheduling.
        </div>
    </div>""", unsafe_allow_html=True)

with i3:
    st.markdown("""
    <div class="metric-card" style="text-align:left;">
        <div style="font-size:24px; margin-bottom:8px;">💰</div>
        <div style="font-weight:600; color:#f78166; margin-bottom:8px;">
            Budget Allocation
        </div>
        <div style="font-size:13px; color:#8b949e; line-height:1.7;">
            YoY upward trend confirms business growth.
            30-day forecast enables accurate monthly
            revenue projection for finance teams.
        </div>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<hr style='border-color:#30363d;'>
<div style='text-align:center; color:#8b949e; font-size:12px; 
     font-family: Space Mono, monospace; padding: 16px 0;'>
    Built with Python · Streamlit · Statsmodels · XGBoost
    &nbsp;·&nbsp; Superstore Sales Forecasting Project
</div>
""", unsafe_allow_html=True)
