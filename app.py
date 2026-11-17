import io
import json
import os
import re
import urllib.parse
import urllib.request
from collections import OrderedDict
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
import streamlit as st
import streamlit.components.v1 as components

plt.rcParams["font.sans-serif"] = [
    "Heiti TC",
    "Arial Unicode MS",
    "Noto Sans CJK TC",
    "Noto Sans CJK HK",
    "Noto Sans CJK SC",
    "AppleGothic",
    "sans-serif",
]
plt.rcParams["axes.unicode_minus"] = False

st.set_page_config(page_title="港股 K 線與恒指對比分析系統", layout="wide")

# 內置常用港股名稱對照表
BUILTIN_HK_MAP = {
    "騰訊控股": "0700.HK",
    "小米集團": "1810.HK",
    "滙豐控股": "0005.HK",
    "比亞迪股份": "1211.HK",
}

MAP_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hk_stock_map.json")


def load_hk_stock_map() -> dict:
    """載入 hk_stock_map.json 對照表（支援大小寫及前後空白標準化）"""
    merged = {k.strip(): v for k, v in BUILTIN_HK_MAP.items()}
    if os.path.exists(MAP_FILE_PATH):
        try:
            with open(MAP_FILE_PATH, "r", encoding="utf-8") as f:
                custom_map = json.load(f)
            for k, v in custom_map.items():
                merged[k.strip()] = v.strip()
        except Exception:
            pass
    return merged


HK_STOCK_MAP = load_hk_stock_map()
HK_STOCK_MAP_LOWER = {k.lower(): v for k, v in HK_STOCK_MAP.items()}


@st.cache_data(ttl=86400)
def resolve_hk_ticker(name_or_code: str) -> str:
    """將中文股票名稱或代號轉換為 Yahoo Finance 港股代號 (先查本地對照表，搵唔到再經網上接口搜尋)"""
    clean = name_or_code.strip()
    if clean in HK_STOCK_MAP:
        return HK_STOCK_MAP[clean]
    if clean.lower() in HK_STOCK_MAP_LOWER:
        return HK_STOCK_MAP_LOWER[clean.lower()]

    if clean.upper().endswith(".HK"):
        num = clean[:-3].zfill(4)
        return f"{num}.HK"
    if clean.isdigit():
        num = clean.lstrip("0").zfill(4)
        return f"{num}.HK"

    try:
        q = urllib.parse.quote(clean)
        url = f"https://smartbox.gtimg.cn/s3/?v=2&q={q}&t=hk"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
        match = re.search(r"hk~(\d{5})~", raw)
        if match:
            code5 = match.group(1)
            code4 = code5[1:] if code5.startswith("0") else code5
            return f"{code4}.HK"
    except Exception:
        pass

    return clean


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """計算 10EMA, 20EMA, KDJ(9,3,3) 及單日升跌幅"""
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df["Prev_Close"] = df["Close"].shift(1)
    df["Pct_Change"] = (df["Close"] - df["Prev_Close"]) / df["Prev_Close"] * 100

    # 10 EMA & 20 EMA
    df["EMA10"] = df["Close"].ewm(span=10, adjust=False).mean()
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()

    # KDJ (9, 3, 3)
    low_9 = df["Low"].rolling(window=9, min_periods=1).min()
    high_9 = df["High"].rolling(window=9, min_periods=1).max()
    rsv = (df["Close"] - low_9) / (high_9 - low_9).replace(0, np.nan) * 100
    rsv = rsv.fillna(50)

    k_list, d_list = [], []
    k_val, d_val = 50.0, 50.0
    for r in rsv:
        k_val = (2 / 3) * k_val + (1 / 3) * r
        d_val = (2 / 3) * d_val + (1 / 3) * k_val
        k_list.append(k_val)
        d_list.append(d_val)

    df["K"] = k_list
    df["D"] = d_list
    df["J"] = 3 * df["K"] - 2 * df["D"]

    return df


@st.cache_data(ttl=3600)
def fetch_stock_window(ticker: str, target_dt: datetime, post_days: int = 10):
    """抓取目標日期前 3 個月至後 10 個交易日數據（自動提前抓取預熱期以確保指標準確）"""
    warmup_start = target_dt - relativedelta(months=6)
    window_start = target_dt - relativedelta(months=3)
    fetch_end = target_dt + timedelta(days=max(post_days * 3, 30))

    df = yf.download(
        ticker,
        start=warmup_start.strftime("%Y-%m-%d"),
        end=fetch_end.strftime("%Y-%m-%d"),
        progress=False,
        auto_adjust=False,
    )
    if df is None or df.empty:
        return pd.DataFrame(), None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index).tz_localize(None)

    df = calculate_indicators(df)

    target_ts = pd.Timestamp(target_dt)
    df_from_start = df[df.index >= pd.Timestamp(window_start)]
    before_or_on = df_from_start[df_from_start.index <= target_ts]
    after_target = df_from_start[df_from_start.index > target_ts].iloc[:post_days]
    df_window = pd.concat([before_or_on, after_target])

    if df_window.empty:
        return pd.DataFrame(), None

    after_or_on = df_window[df_window.index >= target_ts]
    marked_date = after_or_on.index[0] if not after_or_on.empty else df_window.index[-1]

    return df_window, marked_date


def build_kline_figure(df: pd.DataFrame, title: str, marked_date: pd.Timestamp, input_date_str: str) -> go.Figure:
    """建立包含陰陽燭、10/20EMA、成交量、KDJ 嘅互動圖表（綠升紅跌）"""
    date_strs = df.index.strftime("%Y-%m-%d")

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.58, 0.20, 0.22],
        subplot_titles=(f"{title} (陰陽燭 + 10/20 EMA)", "成交量 (Volume)", "KDJ (9, 3, 3)"),
    )

    # 1. 陰陽燭 (綠升紅跌)
    fig.add_trace(
        go.Candlestick(
            x=date_strs,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            increasing_line_color="#26a69a",
            increasing_fillcolor="#26a69a",
            decreasing_line_color="#ef5350",
            decreasing_fillcolor="#ef5350",
            hoverinfo="none",
            name="陰陽燭",
        ),
        row=1,
        col=1,
    )

    # 10 EMA & 20 EMA
    fig.add_trace(
        go.Scatter(
            x=date_strs,
            y=df["EMA10"],
            mode="lines",
            line=dict(color="#ff9800", width=1.6),
            name="10 EMA",
            hoverinfo="none",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=date_strs,
            y=df["EMA20"],
            mode="lines",
            line=dict(color="#29b6f6", width=1.6),
            name="20 EMA",
            hoverinfo="none",
        ),
        row=1,
        col=1,
    )

    # 2. 成交量 (綠升紅跌)
    vol_colors = np.where(df["Close"] >= df["Open"], "#26a69a", "#ef5350")
    fig.add_trace(
        go.Bar(
            x=date_strs,
            y=df["Volume"],
            marker_color=vol_colors,
            name="成交量",
            hoverinfo="none",
        ),
        row=2,
        col=1,
    )

    # 3. KDJ
    fig.add_trace(
        go.Scatter(
            x=date_strs,
            y=df["K"],
            mode="lines",
            line=dict(color="#ffeb3b", width=1.4),
            name="K",
            hoverinfo="none",
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=date_strs,
            y=df["D"],
            mode="lines",
            line=dict(color="#29b6f6", width=1.4),
            name="D",
            hoverinfo="none",
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=date_strs,
            y=df["J"],
            mode="lines",
            line=dict(color="#ff4081", width=1.4),
            name="J",
            hoverinfo="none",
        ),
        row=3,
        col=1,
    )

    # 標示輸入嘅基準日期垂直虛線
    if marked_date is not None:
        marked_str = marked_date.strftime("%Y-%m-%d")
        for r in range(1, 4):
            fig.add_vline(
                x=marked_str,
                line_width=1.5,
                line_dash="dash",
                line_color="#ffd54f",
                row=r,
                col=1,
            )
        fig.add_annotation(
            x=marked_str,
            y=df["High"].max(),
            text=f"基準日 ({input_date_str})",
            showarrow=True,
            arrowhead=2,
            bgcolor="#ffd54f",
            font=dict(color="#000000", size=11),
            row=1,
            col=1,
        )

    # 滑鼠同步對比垂直虛線（預設隱藏，懸停時全部圖表同步顯示同一日垂直虛線）
    init_x = marked_date.strftime("%Y-%m-%d") if marked_date is not None else date_strs[0]
    fig.add_shape(
        type="line",
        x0=init_x,
        x1=init_x,
        y0=0,
        y1=1,
        xref="x",
        yref="paper",
        line=dict(color="#64b5f6", width=1.5, dash="dash"),
        visible=False,
    )

    fig.update_layout(
        template="plotly_dark",
        height=650,
        margin=dict(l=45, r=18, t=42, b=28),
        xaxis_rangeslider_visible=False,
        hovermode="x",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(
        type="category",
        nticks=10,
        showspikes=False,
    )
    return fig


def df_to_lookup(df: pd.DataFrame) -> dict:
    """整理每個交易日嘅現價同升跌幅供前端同步顯示"""
    lookup = {}
    for dt_s, row in zip(df.index.strftime("%Y-%m-%d"), df.to_dict("records")):
        pct = row["Pct_Change"]
        lookup[dt_s] = {
            "close": round(float(row["Close"]), 2),
            "pct": round(float(pct), 2) if pd.notna(pct) else None,
        }
    return lookup


def render_pinned_hsi_row(
    hsi_entry: dict,
    stock_entries: list,
    default_date_str: str,
    card_width: int,
    uid: str,
):
    """
    左手邊固定顯示恒生指數，右手邊橫向排列同日所有個股圖表。
    滑鼠移到任何一張圖表時，全部圖表同步顯示同一日垂直虛線及頂部現價與升跌幅。
    """
    all_items = [hsi_entry] + stock_entries
    is_single_pair = len(stock_entries) == 1

    cards_html = []
    js_items = []
    for idx, item in enumerate(all_items):
        is_hsi = idx == 0
        card_cls = "card hsi-pinned" if is_hsi else "card stock-card"
        chart_id = f"chart_{uid}_{idx}"
        hud_id = f"hud_{uid}_{idx}"
        pin_badge = '<span class="pin-badge">📌 固定對比</span>' if is_hsi else ""

        cards_html.append(
            f"""
            <div class="{card_cls}">
              <div class="hud {'hud-hsi' if is_hsi else ''}">
                <span class="hud-title">{pin_badge}{item['title']}</span>
                <div class="hud-metrics" id="{hud_id}"></div>
              </div>
              <div id="{chart_id}"></div>
            </div>
            """
        )
        js_items.append(
            {
                "chartId": chart_id,
                "hudId": hud_id,
                "fig": json.loads(item["fig"].to_json()),
                "map": df_to_lookup(item["df"]),
            }
        )

    if is_single_pair:
        row_style = "display: grid; grid-template-columns: 1fr 1fr; gap: 14px; width: 100%;"
        card_flex_style = "width: 100%; min-width: 0;"
    else:
        row_style = (
            "display: flex; flex-direction: row; align-items: flex-start; "
            "gap: 14px; width: 100%; overflow-x: auto; padding-bottom: 10px;"
        )
        card_flex_style = f"flex: 0 0 {card_width}px; width: {card_width}px;"

    items_json = json.dumps(js_items, ensure_ascii=False)
    total_height = 760

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8" />
      <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
      <style>
        body {{
          margin: 0;
          padding: 0;
          background: transparent;
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
          color: #e0e0e0;
        }}
        .row-wrap {{
          {row_style}
        }}
        .row-wrap::-webkit-scrollbar {{
          height: 10px;
        }}
        .row-wrap::-webkit-scrollbar-track {{
          background: #1a1a1a;
          border-radius: 5px;
        }}
        .row-wrap::-webkit-scrollbar-thumb {{
          background: #444b5a;
          border-radius: 5px;
        }}
        .card {{
          {card_flex_style}
          background: #111111;
          border: 1px solid #2b2b2b;
          border-radius: 10px;
          padding: 10px 12px 6px 12px;
          box-sizing: border-box;
        }}
        .hsi-pinned {{
          position: sticky;
          left: 0;
          z-index: 20;
          border-color: #ffd54f;
          box-shadow: 6px 0 18px rgba(0, 0, 0, 0.78);
        }}
        .hud {{
          display: flex;
          align-items: center;
          justify-content: space-between;
          flex-wrap: wrap;
          gap: 8px;
          padding: 6px 10px;
          margin-bottom: 6px;
          background: #1b1e26;
          border-radius: 6px;
          font-size: 13.5px;
          border-left: 4px solid #64b5f6;
        }}
        .hud-hsi {{
          border-left: 4px solid #ffd54f;
        }}
        .hud-title {{
          font-weight: 700;
          color: #ffffff;
          display: flex;
          align-items: center;
          gap: 6px;
        }}
        .pin-badge {{
          font-size: 11px;
          background: #ffd54f;
          color: #000000;
          padding: 1px 6px;
          border-radius: 4px;
          font-weight: 700;
        }}
        .hud-metrics {{
          display: flex;
          gap: 12px;
          align-items: center;
          font-variant-numeric: tabular-nums;
        }}
        .badge-up {{
          color: #26a69a;
          font-weight: 700;
        }}
        .badge-down {{
          color: #ef5350;
          font-weight: 700;
        }}
        .badge-flat {{
          color: #b0bec5;
          font-weight: 700;
        }}
      </style>
    </head>
    <body>
      <div class="row-wrap">
        {''.join(cards_html)}
      </div>

      <script>
        const chartItems = {items_json};
        const defaultDate = {json.dumps(default_date_str)};
        const plotConfig = {{ responsive: true, displayModeBar: true, displaylogo: false }};

        function formatHud(el, dateStr, map) {{
          const info = map[dateStr];
          if (!info) {{
            el.innerHTML = `<span>日期: ${{dateStr}}</span><span>無交易數據</span>`;
            return;
          }}
          let pctHtml = '<span class="badge-flat">升跌幅: N/A</span>';
          if (info.pct !== null && info.pct !== undefined) {{
            const sign = info.pct >= 0 ? "+" : "";
            const cls = info.pct >= 0 ? "badge-up" : "badge-down";
            pctHtml = `<span class="${{cls}}">升跌幅: ${{sign}}${{info.pct.toFixed(2)}}%</span>`;
          }}
          el.innerHTML = `<span>📅 ${{dateStr}}</span><span><b>現價: ${{info.close.toFixed(2)}}</b></span>${{pctHtml}}`;
        }}

        const instances = chartItems.map((item) => {{
          const div = document.getElementById(item.chartId);
          const hud = document.getElementById(item.hudId);
          const hoverIdx = (item.fig.layout.shapes ? item.fig.layout.shapes.length : 1) - 1;
          return {{ div, hud, fig: item.fig, map: item.map, hoverIdx }};
        }});

        let lastHoverDate = null;
        let rafPending = false;
        let pendingDate = null;
        let unhoverTimer = null;

        function applySyncHover(dateStr) {{
          instances.forEach((inst) => {{
            formatHud(inst.hud, dateStr, inst.map);
            if (inst.map[dateStr]) {{
              Plotly.relayout(inst.div, {{
                [`shapes[${{inst.hoverIdx}}].x0`]: dateStr,
                [`shapes[${{inst.hoverIdx}}].x1`]: dateStr,
                [`shapes[${{inst.hoverIdx}}].visible`]: true,
              }});
            }} else {{
              Plotly.relayout(inst.div, {{ [`shapes[${{inst.hoverIdx}}].visible`]: false }});
            }}
          }});
        }}

        function syncHover(dateStr) {{
          if (unhoverTimer) {{ clearTimeout(unhoverTimer); unhoverTimer = null; }}
          if (dateStr === lastHoverDate) return;
          lastHoverDate = dateStr;
          pendingDate = dateStr;
          if (!rafPending) {{
            rafPending = true;
            requestAnimationFrame(() => {{
              rafPending = false;
              if (pendingDate) applySyncHover(pendingDate);
            }});
          }}
        }}

        function scheduleUnhover() {{
          if (unhoverTimer) clearTimeout(unhoverTimer);
          unhoverTimer = setTimeout(() => {{
            lastHoverDate = null;
            pendingDate = null;
            instances.forEach((inst) => {{
              Plotly.relayout(inst.div, {{ [`shapes[${{inst.hoverIdx}}].visible`]: false }});
              formatHud(inst.hud, defaultDate, inst.map);
            }});
          }}, 120);
        }}

        Promise.all(
          instances.map((inst) => Plotly.newPlot(inst.div, inst.fig.data, inst.fig.layout, plotConfig))
        ).then(() => {{
          instances.forEach((inst) => {{
            formatHud(inst.hud, defaultDate, inst.map);
            inst.div.on("plotly_hover", (ev) => {{
              if (ev && ev.points && ev.points.length > 0) {{
                syncHover(String(ev.points[0].x));
              }}
            }});
            inst.div.on("plotly_unhover", scheduleUnhover);
          }});
        }});
      </script>
    </body>
    </html>
    """
    components.html(html, height=total_height, scrolling=False)


def parse_batch_input(raw_text: str):
    """解析多行『YYYY,MM,DD  股票名稱』格式輸入（亦支援用分號 ; 分隔多筆，方便手機輸入）"""
    entries = []
    normalized_text = raw_text.replace("；", "\n").replace(";", "\n")
    for line in normalized_text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = re.split(r"[\t\s]+", line, maxsplit=1)
        if len(parts) < 2:
            continue
        raw_date, stock_name = parts[0].strip(), parts[1].strip()
        norm_date = re.sub(r"[.,/]", "-", raw_date)
        try:
            dt = datetime.strptime(norm_date, "%Y-%m-%d")
            entries.append({"date": dt, "date_str": dt.strftime("%Y-%m-%d"), "name": stock_name})
        except ValueError:
            st.warning(f"無法解析日期格式：{line}")
    return entries


def _draw_mpl_kline(fig, outer_slot, df: pd.DataFrame, title: str, marked_date: pd.Timestamp, input_date_str: str):
    """喺 Matplotlib 畫出單張包含陰陽燭、10/20EMA、成交量同 KDJ 嘅圖表（白底清晰版，供 PDF 匯出）"""
    inner_gs = gridspec.GridSpecFromSubplotSpec(
        3, 1, subplot_spec=outer_slot, height_ratios=[0.56, 0.20, 0.24], hspace=0.14
    )
    ax1 = fig.add_subplot(inner_gs[0])
    ax2 = fig.add_subplot(inner_gs[1], sharex=ax1)
    ax3 = fig.add_subplot(inner_gs[2], sharex=ax1)

    x = np.arange(len(df))
    opens = df["Open"].values
    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values
    vols = df["Volume"].values
    date_strs = df.index.strftime("%Y-%m-%d")

    up_color = "#16a34a"
    down_color = "#dc2626"
    price_span = max(float(np.nanmax(highs) - np.nanmin(lows)), 1e-6)
    min_body = price_span * 0.008

    # 1. 陰陽燭 + 10/20 EMA
    for i in range(len(df)):
        c = up_color if closes[i] >= opens[i] else down_color
        ax1.plot([i, i], [lows[i], highs[i]], color=c, linewidth=0.65, zorder=2)
        body_h = max(abs(closes[i] - opens[i]), min_body)
        body_y = min(opens[i], closes[i])
        ax1.add_patch(Rectangle((i - 0.32, body_y), 0.64, body_h, facecolor=c, edgecolor=c, linewidth=0.4, zorder=3))

    ax1.plot(x, df["EMA10"].values, color="#f59e0b", linewidth=0.95, label="10 EMA", zorder=4)
    ax1.plot(x, df["EMA20"].values, color="#0284c7", linewidth=0.95, label="20 EMA", zorder=4)

    # 顯示基準日現價同升跌幅喺標題
    ref_idx = None
    if marked_date is not None and marked_date in df.index:
        ref_idx = df.index.get_loc(marked_date)
    if ref_idx is not None:
        ref_close = closes[ref_idx]
        ref_pct = df["Pct_Change"].iloc[ref_idx]
        pct_txt = f"{'+' if ref_pct >= 0 else ''}{ref_pct:.2f}%" if pd.notna(ref_pct) else "N/A"
        subtitle = f"{title}  |  {date_strs[ref_idx]} 現價: {ref_close:.2f} ({pct_txt})"
    else:
        subtitle = title
    ax1.set_title(subtitle, fontsize=8.5, fontweight="bold", loc="left", pad=4)
    ax1.legend(loc="upper left", fontsize=6.5, frameon=False, ncol=2)
    ax1.grid(True, linestyle=":", linewidth=0.4, alpha=0.6)
    ax1.tick_params(axis="y", labelsize=6.5)
    ax1.tick_params(axis="x", labelbottom=False)

    # 2. 成交量
    vol_colors = [up_color if closes[i] >= opens[i] else down_color for i in range(len(df))]
    ax2.bar(x, vols, color=vol_colors, width=0.65, alpha=0.85)
    ax2.set_ylabel("Vol", fontsize=6.5)
    ax2.grid(True, linestyle=":", linewidth=0.4, alpha=0.6)
    ax2.tick_params(axis="y", labelsize=6)
    ax2.tick_params(axis="x", labelbottom=False)

    # 3. KDJ
    ax3.plot(x, df["K"].values, color="#ca8a04", linewidth=0.85, label="K")
    ax3.plot(x, df["D"].values, color="#0284c7", linewidth=0.85, label="D")
    ax3.plot(x, df["J"].values, color="#db2777", linewidth=0.85, label="J")
    ax3.legend(loc="upper left", fontsize=6, frameon=False, ncol=3)
    ax3.grid(True, linestyle=":", linewidth=0.4, alpha=0.6)
    ax3.tick_params(axis="y", labelsize=6)

    # 基準日垂直虛線
    if ref_idx is not None:
        for ax in (ax1, ax2, ax3):
            ax.axvline(ref_idx, color="#d97706", linestyle="--", linewidth=0.95, zorder=5)

    # X 軸日期標籤
    tick_count = min(6, len(df))
    if tick_count > 0:
        tick_pos = np.linspace(0, len(df) - 1, tick_count, dtype=int)
        ax3.set_xticks(tick_pos)
        ax3.set_xticklabels([date_strs[p][2:] for p in tick_pos], fontsize=6.5)
    ax3.set_xlim(-0.8, len(df) - 0.2)


def generate_vertical_pdf(pdf_pairs: list) -> bytes:
    """產生打直睇版面嘅 PDF 報告（每頁最多 2 組，每組左邊恒指、右邊個股）"""
    buf = io.BytesIO()
    pairs_per_page = 2
    with PdfPages(buf) as pdf:
        for page_start in range(0, len(pdf_pairs), pairs_per_page):
            chunk = pdf_pairs[page_start : page_start + pairs_per_page]
            fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
            outer_gs = gridspec.GridSpec(
                pairs_per_page, 2, figure=fig, left=0.06, right=0.97, top=0.94, bottom=0.05, wspace=0.18, hspace=0.22
            )
            fig.suptitle("港股 K 線與恒生指數對比報告（打直版面）", fontsize=12, fontweight="bold", y=0.975)

            for row_idx, pair in enumerate(chunk):
                _draw_mpl_kline(
                    fig,
                    outer_gs[row_idx, 0],
                    pair["hsi_df"],
                    "恒生指數 (^HSI)",
                    pair["hsi_marked"],
                    pair["date_str"],
                )
                _draw_mpl_kline(
                    fig,
                    outer_gs[row_idx, 1],
                    pair["stock_df"],
                    pair["stock_title"],
                    pair["stock_marked"],
                    pair["date_str"],
                )

            pdf.savefig(fig)
            plt.close(fig)

    buf.seek(0)
    return buf.getvalue()


# ================= 介面佈局 =================
st.title("📈 批次股票 K 線與恒生指數對比系統")

default_input = "2024,03,18\t中遠海能\n2024,03,18\t中金公司\n2024,03,18\t中國中鐵"
if "user_input_text" not in st.session_state:
    st.session_state["user_input_text"] = default_input

with st.sidebar:
    st.header("1. 批次輸入股票與日期")
    with st.expander("📱 手機快速加入一隻股票", expanded=True):
        q_col1, q_col2 = st.columns(2)
        with q_col1:
            quick_date = st.date_input("基準日期", value=datetime(2024, 3, 18))
        with q_col2:
            quick_name = st.text_input("股票名／代號", placeholder="例如：騰訊")
        b_col1, b_col2 = st.columns(2)
        with b_col1:
            if st.button("➕ 加入清單", use_container_width=True):
                clean_q = quick_name.strip()
                if clean_q:
                    new_line = f"{quick_date.strftime('%Y,%m,%d')}\t{clean_q}"
                    cur = st.session_state["user_input_text"].strip()
                    st.session_state["user_input_text"] = f"{cur}\n{new_line}" if cur else new_line
                    st.rerun()
        with b_col2:
            if st.button("🗑️ 清空清單", use_container_width=True):
                st.session_state["user_input_text"] = ""
                st.rerun()

    user_input = st.text_area(
        "批次清單（每行一隻，手機可用 ; 分隔）：",
        key="user_input_text",
        height=160,
    )
    st.button("🔄 套用並更新圖表", use_container_width=True)

    st.header("2. 圖表顯示設定")
    display_layout = st.radio(
        "排列方式：",
        options=["橫向並列（左邊固定恒指）", "打直逐隻睇（左恒指、右個股）"],
        index=0,
    )
    if display_layout == "橫向並列（左邊固定恒指）":
        card_width = st.slider(
            "橫向多股對比時單張圖表闊度 (px)：",
            min_value=420,
            max_value=760,
            value=540,
            step=20,
        )
    else:
        card_width = 540
    show_all_dates = st.checkbox("同時顯示全部日期組別", value=True)

    st.header("3. 匯出 PDF 報告")
    export_clicked = st.button("📄 生成打直睇 PDF 報告", use_container_width=True)
    pdf_slot = st.empty()

entries = parse_batch_input(user_input)

if export_clicked:
    if not entries:
        st.sidebar.warning("請先輸入股票同日期。")
    else:
        with st.sidebar:
            with st.spinner("正在生成打直版面 PDF 報告..."):
                pdf_pairs = []
                hsi_cache = {}
                for it in entries:
                    d_str = it["date_str"]
                    if d_str not in hsi_cache:
                        hsi_cache[d_str] = fetch_stock_window("^HSI", it["date"], 10)
                    df_hsi_p, md_hsi_p = hsi_cache[d_str]
                    ticker_p = resolve_hk_ticker(it["name"])
                    df_s_p, md_s_p = fetch_stock_window(ticker_p, it["date"], 10)
                    if not df_hsi_p.empty and not df_s_p.empty:
                        pdf_pairs.append(
                            {
                                "date_str": d_str,
                                "hsi_df": df_hsi_p,
                                "hsi_marked": md_hsi_p,
                                "stock_title": f"{it['name']} ({ticker_p})",
                                "stock_df": df_s_p,
                                "stock_marked": md_s_p,
                            }
                        )
                if pdf_pairs:
                    st.session_state["pdf_bytes"] = generate_vertical_pdf(pdf_pairs)
                    st.session_state["pdf_filename"] = f"stock_kline_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
                else:
                    st.error("未能載入任何有效股票數據以生成 PDF。")

if st.session_state.get("pdf_bytes"):
    with pdf_slot:
        st.download_button(
            label="⬇️ 下載 PDF 報告",
            data=st.session_state["pdf_bytes"],
            file_name=st.session_state.get("pdf_filename", "stock_kline_report.pdf"),
            mime="application/pdf",
            use_container_width=True,
        )

if not entries:
    st.info("請喺左側輸入至少一行「日期 + 股票名稱」資料。")
else:
    # 按日期分組（同一日嘅股票可橫向排成一列，或者打直逐隻對比）
    grouped = OrderedDict()
    for e in entries:
        grouped.setdefault(e["date_str"], {"date": e["date"], "items": []})["items"].append(e)

    date_keys = list(grouped.keys())
    if show_all_dates or len(date_keys) == 1:
        selected_dates = date_keys
    else:
        date_options = [f"{d}（{len(grouped[d]['items'])} 隻股票）" for d in date_keys]
        chosen_label = st.selectbox("選擇要檢視嘅基準日期組別：", date_options)
        selected_dates = [date_keys[date_options.index(chosen_label)]]

    for d_idx, date_str in enumerate(selected_dates):
        group = grouped[date_str]
        target_dt = group["date"]
        items = group["items"]
        names_preview = "、".join([it["name"] for it in items])
        st.subheader(f"📅 基準日：{date_str}（前 3 個月至後 10 個交易日）｜ {names_preview}")

        with st.spinner(f"正在載入 {date_str} 恒生指數 (^HSI) 及 {len(items)} 隻個股數據..."):
            df_hsi, marked_hsi = fetch_stock_window("^HSI", target_dt, 10)
            stock_results = []
            for it in items:
                ticker = resolve_hk_ticker(it["name"])
                df_s, marked_s = fetch_stock_window(ticker, target_dt, 10)
                stock_results.append((it, ticker, df_s, marked_s))

        if df_hsi.empty:
            st.error(f"暫時無法載入 {date_str} 嘅恒生指數 (^HSI) 數據。")
            continue

        fig_hsi = build_kline_figure(df_hsi, "恒生指數 (^HSI)", marked_hsi, date_str)
        hsi_entry = {"title": "恒生指數 (^HSI)", "fig": fig_hsi, "df": df_hsi}

        valid_stocks = []
        for it, ticker, df_s, marked_s in stock_results:
            if df_s.empty:
                st.warning(f"搵唔到「{it['name']} ({ticker})」喺 {date_str} 嘅行情數據，已略過。")
                continue
            fig_s = build_kline_figure(df_s, f"{it['name']} ({ticker})", marked_s, date_str)
            valid_stocks.append({"title": f"{it['name']} ({ticker})", "fig": fig_s, "df": df_s})

        if not valid_stocks:
            continue

        default_dt_str = marked_hsi.strftime("%Y-%m-%d") if marked_hsi is not None else date_str
        if display_layout == "橫向並列（左邊固定恒指）":
            render_pinned_hsi_row(
                hsi_entry=hsi_entry,
                stock_entries=valid_stocks,
                default_date_str=default_dt_str,
                card_width=card_width,
                uid=f"row_{d_idx}",
            )
        else:
            for s_idx, s_entry in enumerate(valid_stocks):
                render_pinned_hsi_row(
                    hsi_entry=hsi_entry,
                    stock_entries=[s_entry],
                    default_date_str=default_dt_str,
                    card_width=card_width,
                    uid=f"row_{d_idx}_{s_idx}",
                )
