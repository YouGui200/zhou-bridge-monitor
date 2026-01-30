"""
================================================================================
州桥结构健康监测系统 - 云端部署版 (Strict Standardized Version)
================================================================================
核心修复：
1. 强制列映射：无论输入CSV表头为何，系统会自动将其映射为标准传感器名称。
   - 应变模式 -> strain_S-01_micro ... strain_S-04_micro
   - 加速度模式 -> accel_A-01_ms2 ...
2. 严格对应算法库定义的 4/2/1/1 通道数量。
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
import time
import io
from datetime import datetime
import traceback

# =============================================================================
# 1. 核心配置与路径系统
# =============================================================================

st.set_page_config(
    page_title="州桥监测系统",
    page_icon="🌉",
    layout="wide",
    initial_sidebar_state="expanded"
)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)
DATA_PATH = CURRENT_DIR

try:
    from preprocessing_lib import (
        MissingValueHandler, NoiseFilter, AnomalyDetector, PerformanceMetrics
    )
    ALGO_STATUS = True
except ImportError:
    ALGO_STATUS = False

# -----------------------------------------------------------------------------
# 传感器配置 (严格定义标准通道名)
# -----------------------------------------------------------------------------
SENSORS = {
    'strain': {
        'name': '应变传感器', 
        'icon': '🔴', 
        'color': '#F44336', 
        'file': 'raw_data_strain.csv', 
        'unit': 'με',
        # 定义标准通道名列表
        'channels': [
            'strain_S-01_micro', 
            'strain_S-02_micro', 
            'strain_S-03_micro', 
            'strain_S-04_micro'
        ],
        'desc': '监测拱顶/拱脚受力 (4通道: S-01~S-04)'
    },
    'accel': {
        'name': '加速度传感器', 
        'icon': '🔵', 
        'color': '#2196F3', 
        'file': 'raw_data_acceleration.csv', 
        'unit': 'm/s²', 
        'channels': [
            'accel_A-01_ms2', 
            'accel_A-02_ms2'
        ],
        'desc': '监测桥面振动 (2通道: A-01~A-02)'
    },
    'temp': {
        'name': '温度传感器', 
        'icon': '🟢', 
        'color': '#4CAF50', 
        'file': 'raw_data_temperature.csv', 
        'unit': '°C', 
        'channels': [
            'temperature_T-01_C'
        ],
        'desc': '监测环境温度 (1通道: T-01)'
    },
    'disp': {
        'name': '位移传感器', 
        'icon': '🟣', 
        'color': '#9C27B0', 
        'file': 'raw_data_displacement.csv', 
        'unit': 'mm', 
        'channels': [
            'displacement_D-01_mm'
        ],
        'desc': '监测桥墩沉降 (1通道: D-01)'
    }
}

# =============================================================================
# 2. 核心逻辑工具函数
# =============================================================================

def apply_style():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Times+New+Roman&display=swap');
        .stApp { font-family: "Times New Roman", sans-serif; background-color: #f8f9fa; }
        [data-testid="stSidebar"] { background-color: #1e272e; }
        [data-testid="stSidebar"] * { color: #dcdde1 !important; font-family: Arial, sans-serif; }
        .card { background: white; padding: 20px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); margin-bottom: 15px; border-top: 3px solid #e1e1e1; }
    </style>
    """, unsafe_allow_html=True)

def standardize_columns(df, sensor_type):
    """
    核心修复逻辑：
    强制将数据列重命名为系统预设的标准名称 (如 strain_S-01_micro)，
    确保下拉框显示的永远是标准名称。
    """
    if df is None: return None
    
    # 1. 识别时间列
    cols = df.columns.tolist()
    time_col = None
    data_cols = []
    
    exclude_keywords = ['time', 'date', 'timestamp', 'unnamed', 'id', 'index']
    
    for c in cols:
        if any(k in c.lower() for k in exclude_keywords):
            time_col = c
        else:
            data_cols.append(c)
    
    # 2. 获取该传感器类型应有的标准列名
    expected_channels = SENSORS[sensor_type]['channels']
    
    # 3. 建立重命名映射
    rename_map = {}
    
    # 如果找到了时间列，保留它
    if time_col:
        # 确保时间列名统一，方便后续处理（可选，这里保持原样）
        pass
    
    # 强制映射数据列
    # 如果数据列数量 <= 标准通道数，按顺序赋予标准名
    # 如果数据列数量 > 标准通道数，只取前N个
    count = min(len(data_cols), len(expected_channels))
    
    for i in range(count):
        original_col = data_cols[i]
        new_name = expected_channels[i]
        rename_map[original_col] = new_name
        
    # 执行重命名
    new_df = df.rename(columns=rename_map)
    
    # 提示信息 (仅调试用)
    # print(f"Renamed {rename_map}")
    
    return new_df

def get_display_columns(df):
    """获取用于显示的列（排除时间列）"""
    if df is None: return []
    cols = df.columns.tolist()
    # 只要不包含time/date/timestamp字样，且在我们的标准命名列表里（或者是为了兼容原始数据）
    return [c for c in cols if not any(x in c.lower() for x in ['time', 'date', 'timestamp'])]

# =============================================================================
# 3. 状态管理
# =============================================================================

if 'sensor' not in st.session_state: st.session_state.sensor = 'strain'
if 'page' not in st.session_state: st.session_state.page = 'home'
if 'data_map' not in st.session_state: 
    st.session_state.data_map = {k: {'data': None, 'processed': None, 'meta': None} for k in SENSORS.keys()}

def get_current_data(): return st.session_state.data_map[st.session_state.sensor]
def set_current_data(data=None, processed=None, meta=None):
    if data is not None: st.session_state.data_map[st.session_state.sensor]['data'] = data
    if processed is not None: st.session_state.data_map[st.session_state.sensor]['processed'] = processed
    if meta is not None: st.session_state.data_map[st.session_state.sensor]['meta'] = meta

# =============================================================================
# 4. 绘图逻辑
# =============================================================================

@st.cache_data(show_spinner=False)
def load_csv_data(path):
    return pd.read_csv(path)

def plot_paper_chart(df, col, color, title):
    fig = go.Figure()
    step = max(1, len(df) // 5000)
    
    time_col = None
    for c in df.columns:
        if any(x in c.lower() for x in ['time', 'date', 'timestamp']):
            time_col = c
            break
            
    x_data = df[time_col][::step] if time_col else df.index[::step]
    
    fig.add_trace(go.Scattergl(
        x=x_data, 
        y=df[col][::step], 
        mode='lines', 
        name=col,
        line=dict(color=color, width=1)
    ))
    fig.update_layout(title=f"{title} - {col}", height=350, margin=dict(l=40,r=20,t=40,b=30), plot_bgcolor='white', 
                     xaxis=dict(showgrid=True, gridcolor='#eee'),
                     yaxis=dict(showgrid=True, gridcolor='#eee'))
    return fig

def plot_comparison(orig, proc, color, col_name):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, 
                        subplot_titles=(f"Original: {col_name}", "Processed"))
    step = max(1, len(orig) // 5000)
    fig.add_trace(go.Scattergl(y=orig[::step], name='Raw', line=dict(color='#999', width=0.8)), row=1, col=1)
    fig.add_trace(go.Scattergl(y=proc[::step], name='Clean', line=dict(color=color, width=1.2)), row=2, col=1)
    fig.update_layout(height=500, margin=dict(l=40,r=20,t=40,b=20), plot_bgcolor='white', showlegend=False)
    fig.update_xaxes(showgrid=True, gridcolor='#eee')
    fig.update_yaxes(showgrid=True, gridcolor='#eee')
    return fig

# =============================================================================
# 5. 侧边栏与页面
# =============================================================================

def render_sidebar():
    with st.sidebar:
        st.markdown("<h3 style='text-align:center;'>ZHOU BRIDGE SHM</h3>", unsafe_allow_html=True)
        st.caption("SENSOR SELECTION")
        opts = list(SENSORS.keys())
        labels = [f"{SENSORS[k]['name']}" for k in opts]
        idx = st.radio("Sensor", range(len(opts)), format_func=lambda x: labels[x], label_visibility="collapsed")
        
        key = opts[idx]
        if key != st.session_state.sensor:
            st.session_state.sensor = key
            st.toast(f"已切换至 {SENSORS[key]['name']}", icon="🔄")
            time.sleep(0.3)
            st.rerun()
            
        cur = SENSORS[key]
        st.info(f"**类型:** {cur['name']}\n**标准通道数:** {len(cur['channels'])}")
        st.markdown("---")
        st.caption("MODULES")
        nav = {'home': '🏠 系统概览', 'data': '📊 数据管理', 'process': '⚡ 智能处理', 'export': '📥 成果导出'}
        page = st.radio("Nav", list(nav.keys()), format_func=lambda x: nav[x], label_visibility="collapsed")
        return page

def page_home():
    st.title("🏠 系统概览")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("传感器类型", "4 类")
    c2.metric("监测通道总数", "8 个", help="S-01~04, A-01~02, T-01, D-01")
    c3.metric("算法引擎", "Ready" if ALGO_STATUS else "Missing")
    total = sum([len(v['data']) if v['data'] is not None else 0 for v in st.session_state.data_map.values()])
    c4.metric("总数据量", f"{total:,}")
    
    st.markdown("---")
    cols = st.columns(4)
    for i, (k, s) in enumerate(SENSORS.items()):
        with cols[i]:
            has_data = st.session_state.data_map[k]['data'] is not None
            st.markdown(f"""
            <div class="card" style="border-top-color:{s['color']}; text-align:center;">
                <h1 style="font-size: 3em; margin: 0;">{s['icon']}</h1>
                <h4 style="margin: 10px 0;">{s['name']}</h4>
                <p style="font-size: 0.85em; color: #666;">{len(s['channels'])} 个通道</p>
                <p style="color:{'#2ecc71' if has_data else '#95a5a6'}; font-weight: bold;">● {'在线' if has_data else '离线'}</p>
            </div>
            """, unsafe_allow_html=True)

def page_data():
    sensor_key = st.session_state.sensor
    s = SENSORS[sensor_key]
    store = get_current_data()
    st.title(f"📊 数据管理 - {s['name']}")
    
    c1, c2 = st.columns([1, 2])
    
    with c1:
        st.markdown("### 📥 数据加载")
        
        tab_auto, tab_manual = st.tabs(["📂 文件 / 演示", "✍️ 手动输入"])
        
        with tab_auto:
            st.caption(f"预期加载文件: {s['file']}")
            if st.button("🚀 加载演示数据", type="primary", use_container_width=True):
                path = os.path.join(DATA_PATH, s['file'])
                if os.path.exists(path):
                    with st.spinner("读取并标准化..."):
                        raw_df = load_csv_data(path)
                        # 核心步骤：标准化列名
                        std_df = standardize_columns(raw_df, sensor_key)
                        set_current_data(data=std_df, processed=None)
                        st.success(f"已加载并映射 {len(std_df)} 行数据")
                        time.sleep(0.5)
                        st.rerun()
                else:
                    st.error(f"未找到文件: {s['file']}")

            st.markdown("---")
            uploaded = st.file_uploader("上传 CSV", type=['csv'])
            if uploaded:
                try:
                    raw_df = pd.read_csv(uploaded)
                    std_df = standardize_columns(raw_df, sensor_key)
                    set_current_data(data=std_df)
                    st.success("上传并标准化成功")
                    st.rerun()
                except Exception as e:
                    st.error(f"解析失败: {e}")

        with tab_manual:
            st.info("请输入 CSV 数据 (表头名称不重要，系统将自动按顺序映射为标准通道名)")
            
            # 动态生成符合当前传感器通道数量的默认文本
            if sensor_key == 'strain':
                # 4通道
                example = "timestamp,CH1,CH2,CH3,CH4\n2023-01-01,10.1,10.2,10.3,10.4\n2023-01-02,11.1,11.2,11.3,11.4"
            elif sensor_key == 'accel':
                # 2通道
                example = "timestamp,CH1,CH2\n2023-01-01,0.01,0.02\n2023-01-02,0.03,0.01"
            else:
                # 1通道
                example = "timestamp,CH1\n2023-01-01,25.5\n2023-01-02,26.1"
                
            manual_text = st.text_area("数据输入区", height=200, value=example)
            
            if st.button("解析并标准化", use_container_width=True):
                if manual_text.strip():
                    try:
                        raw_df = pd.read_csv(io.StringIO(manual_text))
                        std_df = standardize_columns(raw_df, sensor_key)
                        set_current_data(data=std_df)
                        st.toast("数据加载成功 (列名已重置)", icon="✅")
                        st.rerun()
                    except Exception as e:
                        st.error(f"格式错误: {e}")
                else:
                    st.warning("输入为空")

    with c2:
        if store['data'] is not None:
            df = store['data']
            st.markdown("### 📈 数据预览")
            st.dataframe(df.head(10), use_container_width=True)
            
            # --- 核心：这里的 cols 一定是标准化后的 (strain_S-01_micro 等) ---
            sensor_cols = get_display_columns(df)
            # 再次按名称排序，确保 S-01, S-02 顺序
            sensor_cols.sort()
            
            if len(sensor_cols) > 0:
                col = st.selectbox("选择传感器通道", sensor_cols)
                if col:
                    try:
                        st.plotly_chart(plot_paper_chart(df, col, s['color'], s['name']), use_container_width=True)
                    except Exception as e:
                        st.error(f"绘图错误: {e}")
            else:
                st.warning("数据中未找到有效的数据列")
        else:
            st.info("👈 请先从左侧加载数据")

def page_process():
    s = SENSORS[st.session_state.sensor]
    store = get_current_data()
    st.title(f"⚡ 智能处理 - {s['name']}")
    if store['data'] is None:
        st.warning("⚠️ 请先加载数据")
        return
    
    df = store['data']
    sensor_cols = get_display_columns(df)
    sensor_cols.sort()

    c1, c2 = st.columns([1, 2.5])
    with c1:
        st.markdown("### ⚙️ 算法配置")
        if not sensor_cols:
            st.error("无可处理通道")
            return
        
        target = st.selectbox("1. 目标通道", sensor_cols)
        st.markdown("---")
        fill = st.selectbox("2. 缺失值处理", ['spline', 'linear', 'polynomial', 'nearest'])
        anom = st.selectbox("3. 异常检测", ['sigma', 'iqr', 'mad', 'isolation_forest'])
        
        if anom == 'sigma': thresh = st.slider("阈值 (n_sigma)", 1.0, 5.0, 3.0)
        elif anom == 'iqr': thresh = st.slider("阈值 (k)", 1.0, 3.0, 1.5)
        else: thresh = st.slider("阈值", 2.0, 5.0, 3.5)
        
        filt = st.selectbox("4. 滤波算法", ['wavelet', 'moving_average', 'gaussian', 'savgol'])
        st.markdown("---")
        
        if st.button("🚀 运行处理", type="primary", use_container_width=True):
            if not ALGO_STATUS:
                st.error("算法库 preprocessing_lib.py 缺失")
                return

            bar = st.progress(0, text="初始化...")
            
            try:
                raw = pd.to_numeric(df[target], errors='coerce').values
                
                # Step 1
                bar.progress(30, text=f"填补 ({fill})...")
                time.sleep(0.2)
                h = MissingValueHandler()
                s1 = h.fill_missing(raw, fill)
                
                # Step 2
                bar.progress(60, text=f"检测异常 ({anom})...")
                time.sleep(0.2)
                d = AnomalyDetector()
                kw = {}
                if anom == 'sigma': kw['n_sigma'] = thresh
                elif anom == 'iqr': kw['k'] = thresh
                else: kw['threshold'] = thresh
                
                _, idx = d.detect_anomalies(s1, anom, **kw)
                s2 = d.replace_anomalies(s1, anom, 'interpolation', **kw)
                
                # Step 3
                bar.progress(85, text=f"去噪 ({filt})...")
                time.sleep(0.2)
                f = NoiseFilter()
                s3 = f.filter_signal(s2, filt)
                snr = PerformanceMetrics.calculate_snr(s2, s3)
                
                bar.progress(100, text="完成")
                time.sleep(0.5)
                bar.empty()
                
                meta = {
                    'col': target,
                    'params': {'fill': fill, 'anom': anom, 'filt': filt},
                    'stats': {'idx': len(idx), 'snr': snr},
                    'original': raw
                }
                set_current_data(processed=s3, meta=meta)
                st.toast("处理成功", icon="✅")
                
            except Exception as e:
                st.error(f"错误: {e}")
                st.code(traceback.format_exc())

    with c2:
        if store['processed'] is not None:
            res = store['meta']
            if res.get('col') != target:
                st.warning("⚠️ 结果未更新，请点击运行")
            
            proc = store['processed']
            orig = res['original']
            
            st.markdown("### 📈 结果分析")
            k1, k2, k3 = st.columns(3)
            k1.metric("异常点", f"{res['stats']['idx']}", delta="Detected")
            k2.metric("SNR", f"{res['stats']['snr']:.2f} dB", delta="Quality")
            k3.metric("通道", target)
            
            st.plotly_chart(plot_comparison(orig, proc, s['color'], target), use_container_width=True)
        else:
            st.info("👈 请运行算法")

def page_export():
    s_info = SENSORS[st.session_state.sensor]
    store = get_current_data()
    st.title(f"📥 成果导出 - {s_info['name']}")
    if store['processed'] is None:
        st.warning("⚠️ 无处理结果")
        return
        
    res = store['meta']
    proc = store['processed']
    col_name = res.get('col', 'data')
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 💾 导出 CSV")
        df_out = pd.DataFrame({
            f'Raw_{col_name}': res['original'], 
            f'Clean_{col_name}': proc
        })
        csv = df_out.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📥 下载数据 (CSV)",
            data=csv,
            file_name=f"Result_{col_name}.csv",
            mime="text/csv",
            type="primary"
        )
    with c2:
        st.markdown("### 📄 导出报告")
        rpt = f"""监测报告\n通道: {col_name}\n异常点: {res['stats']['idx']}\nSNR: {res['stats']['snr']:.2f} dB\n结论: 正常"""
        st.text_area("预览", rpt, height=200)
        st.download_button("📥 下载报告 (TXT)", rpt, f"Report_{col_name}.txt")

def main():
    apply_style()
    page = render_sidebar()
    if page == 'home': page_home()
    elif page == 'data': page_data()
    elif page == 'process': page_process()
    elif page == 'export': page_export()

if __name__ == "__main__":
    main()
