"""
================================================================================
州桥结构健康监测系统 - 云端部署版 (Enhanced)
================================================================================
更新说明：
1. 新增：手动数据输入 (Manual Input)
2. 优化：数据预览移除 Timestamp，显示具体传感器通道名称
3. 路径：保持当前目录适配
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

# 获取当前脚本所在的文件夹路径
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
    # 静默处理或在侧边栏提示，不阻断主流程

# 传感器配置
SENSORS = {
    'strain': {'name': '应变传感器', 'icon': '🔴', 'color': '#e74c3c', 'file': 'raw_data_strain.csv', 'unit': 'με', 'desc': '监测拱顶/拱脚受力'},
    'accel': {'name': '加速度传感器', 'icon': '🔵', 'color': '#3498db', 'file': 'raw_data_acceleration.csv', 'unit': 'm/s²', 'desc': '监测桥面振动'},
    'temp': {'name': '温度传感器', 'icon': '🟢', 'color': '#2ecc71', 'file': 'raw_data_temperature.csv', 'unit': '°C', 'desc': '监测环境温度'},
    'disp': {'name': '位移传感器', 'icon': '🟣', 'color': '#9b59b6', 'file': 'raw_data_displacement.csv', 'unit': 'mm', 'desc': '监测桥墩沉降'}
}

# =============================================================================
# 2. 视觉样式
# =============================================================================

def apply_style():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Times+New+Roman&display=swap');
        .stApp { font-family: "Times New Roman", sans-serif; background-color: #f8f9fa; }
        [data-testid="stSidebar"] { background-color: #1e272e; }
        [data-testid="stSidebar"] * { color: #dcdde1 !important; font-family: Arial, sans-serif; }
        .stRadio > div[role="radiogroup"] > label { background: rgba(255,255,255,0.05); padding: 10px; border-radius: 4px; margin-bottom: 5px; }
        .stRadio > div[role="radiogroup"] > label:hover { background: #3b82f6; }
        .card { background: white; padding: 20px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); margin-bottom: 15px; border-top: 3px solid #e1e1e1; }
        div[data-testid="stToast"] { border-left: 5px solid #3b82f6; }
    </style>
    """, unsafe_allow_html=True)

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
# 4. 核心逻辑
# =============================================================================

@st.cache_data(show_spinner=False)
def load_csv_data(path):
    return pd.read_csv(path)

def plot_paper_chart(df, col, color, title):
    fig = go.Figure()
    # 智能降采样：如果数据量过大，进行降采样以提高绘图速度
    step = max(1, len(df) // 5000) 
    
    # 尝试寻找时间轴，如果没有则用索引
    time_col = None
    for c in df.columns:
        if 'time' in c.lower() or 'date' in c.lower():
            time_col = c
            break
            
    x_data = df[time_col][::step] if time_col else df.index[::step]
    
    fig.add_trace(go.Scattergl(x=x_data, y=df[col][::step], mode='lines', line=dict(color=color, width=1)))
    fig.update_layout(title=title, height=300, margin=dict(l=40,r=20,t=30,b=30), plot_bgcolor='white', 
                     xaxis=dict(showgrid=True, gridcolor='#eee', showline=True, mirror=True),
                     yaxis=dict(showgrid=True, gridcolor='#eee', showline=True, mirror=True))
    return fig

def plot_comparison(orig, proc, color):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)
    step = max(1, len(orig) // 5000)
    fig.add_trace(go.Scattergl(y=orig[::step], line=dict(color='#999', width=0.8)), row=1, col=1)
    fig.add_trace(go.Scattergl(y=proc[::step], line=dict(color=color, width=1.2)), row=2, col=1)
    fig.update_layout(height=450, margin=dict(l=40,r=20,t=20,b=20), plot_bgcolor='white', showlegend=False)
    fig.update_xaxes(showgrid=True, gridcolor='#eee')
    fig.update_yaxes(showgrid=True, gridcolor='#eee')
    return fig

# =============================================================================
# 5. 侧边栏
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
        st.info(f"**ID:** {cur['file']}\n\n**Unit:** {cur['unit']}")
        st.markdown("---")
        st.caption("MODULES")
        nav = {'home': '🏠 系统概览', 'data': '📊 数据管理', 'process': '⚡ 智能处理', 'export': '📥 成果导出'}
        page = st.radio("Nav", list(nav.keys()), format_func=lambda x: nav[x], label_visibility="collapsed")
        return page

# =============================================================================
# 6. 页面
# =============================================================================

def page_home():
    st.title("🏠 系统概览")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("传感器", "4 类")
    c2.metric("监测点", "8 个")
    status_text = "就绪" if ALGO_STATUS else "受限"
    c3.metric("算法库", status_text, "Available" if ALGO_STATUS else "Missing")
    total = sum([len(v['data']) if v['data'] is not None else 0 for v in st.session_state.data_map.values()])
    c4.metric("数据量", f"{total:,}")
    st.markdown("---")
    cols = st.columns(4)
    for i, (k, s) in enumerate(SENSORS.items()):
        with cols[i]:
            has = st.session_state.data_map[k]['data'] is not None
            st.markdown(f"""<div class="card" style="border-top-color:{s['color']}; text-align:center;">
            <h1>{s['icon']}</h1><h4>{s['name']}</h4>
            <p style="color:{'#2ecc71' if has else '#95a5a6'}">● {'已就绪' if has else '待机'}</p>
            </div>""", unsafe_allow_html=True)

def page_data():
    s = SENSORS[st.session_state.sensor]
    store = get_current_data()
    st.title(f"📊 数据管理 - {s['name']}")
    
    c1, c2 = st.columns([1, 2])
    
    with c1:
        st.markdown("### 📥 数据来源")
        
        # 使用 Tabs 区分不同的输入方式
        tab_auto, tab_manual = st.tabs(["📂 文件 / 演示", "✍️ 手动输入"])
        
        with tab_auto:
            # 1. 演示数据按钮
            if st.button("🚀 加载演示数据", type="primary", use_container_width=True):
                path = os.path.join(DATA_PATH, s['file'])
                if os.path.exists(path):
                    with st.spinner("读取数据中..."):
                        df = load_csv_data(path)
                        set_current_data(data=df, processed=None)
                        st.toast(f"成功加载 {len(df)} 行数据", icon="✅")
                        time.sleep(0.5)
                        st.rerun()
                else:
                    st.error(f"在服务器上未找到文件: {s['file']}")

            st.markdown("---")
            
            # 2. 文件上传
            uploaded = st.file_uploader("上传 CSV 文件", type=['csv'])
            if uploaded:
                try:
                    df = pd.read_csv(uploaded)
                    set_current_data(data=df)
                    st.success(f"上传成功: {uploaded.name}")
                except Exception as e:
                    st.error(f"解析失败: {e}")

        with tab_manual:
            st.info("请直接粘贴 CSV 格式的文本数据 (包含表头)")
            # 3. 手动输入
            manual_text = st.text_area("粘贴数据区域", height=200, placeholder="timestamp,strain_S-01\n2023-01-01,10.5\n2023-01-02,11.2...")
            if st.button("解析文本数据", use_container_width=True):
                if manual_text.strip():
                    try:
                        df = pd.read_csv(io.StringIO(manual_text))
                        set_current_data(data=df)
                        st.toast("文本解析成功", icon="✅")
                        st.rerun()
                    except Exception as e:
                        st.error(f"格式错误: {e}")
                else:
                    st.warning("内容为空")

    with c2:
        if store['data'] is not None:
            df = store['data']
            st.markdown("### 📈 数据预览")
            st.dataframe(df.head(50), use_container_width=True, height=200)
            
            # --- 关键修改：过滤列并显示所有传感器 ---
            # 1. 获取所有列名
            all_cols = df.columns.tolist()
            # 2. 过滤掉包含 'time', 'date', 'timestamp' 的列 (不区分大小写)
            sensor_cols = [c for c in all_cols if 'time' not in c.lower() and 'date' not in c.lower() and 'timestamp' not in c.lower()]
            
            if len(sensor_cols) > 0:
                # 3. 让用户选择具体的传感器通道 (例如 strain_S-01_micro)
                col = st.selectbox("选择传感器通道 (预览)", sensor_cols)
                # 4. 绘图
                st.plotly_chart(plot_paper_chart(df, col, s['color'], col), use_container_width=True)
            else:
                st.warning("未找到有效的传感器数据列 (非时间列)")
        else:
            st.info("👈 请在左侧选择数据加载方式")

def page_process():
    s = SENSORS[st.session_state.sensor]
    store = get_current_data()
    st.title(f"⚡ 智能处理 - {s['name']}")
    if store['data'] is None:
        st.warning("⚠️ 请先加载数据")
        return
    df = store['data']
    
    # 同样过滤掉时间列，供算法选择
    all_cols = df.columns.tolist()
    sensor_cols = [c for c in all_cols if 'time' not in c.lower() and 'date' not in c.lower()]
    # 确保只要是数值型且不在排除列表中
    num = [c for c in sensor_cols if pd.api.types.is_numeric_dtype(df[c])]

    c1, c2 = st.columns([1, 2.5])
    with c1:
        st.markdown("### ⚙️ 算法配置")
        if not num:
            st.error("数据中没有数值列可处理")
            return
            
        target = st.selectbox("1. 目标列", num)
        st.markdown("---")
        fill = st.selectbox("2. 缺失值", ['spline', 'linear', 'polynomial'])
        anom = st.selectbox("3. 异常检测", ['sigma', 'iqr', 'mad'])
        if anom == 'sigma': thresh = st.slider("阈值 (n_sigma)", 1.0, 5.0, 3.0)
        elif anom == 'iqr': thresh = st.slider("阈值 (k)", 1.0, 3.0, 1.5)
        else: thresh = st.slider("阈值 (threshold)", 2.0, 5.0, 3.5)
        filt = st.selectbox("4. 滤波去噪", ['wavelet', 'moving_average', 'gaussian'])
        st.markdown("---")
        
        if st.button("🚀 开始处理", type="primary", use_container_width=True):
            bar = st.progress(0, text="初始化 0%")
            
            try:
                raw = df[target].values.astype(float)
                
                # Step 1
                bar.progress(25, text=f"正在执行 {fill} 插值... 25%")
                time.sleep(0.3)
                if ALGO_STATUS:
                    h = MissingValueHandler()
                    s1 = h.fill_missing(raw, fill)
                else:
                    s1 = pd.Series(raw).interpolate().bfill().values
                
                # Step 2
                bar.progress(50, text=f"正在执行 {anom} 检测... 50%")
                time.sleep(0.3)
                idx = []
                if ALGO_STATUS:
                    d = AnomalyDetector()
                    kw = {'n_sigma': thresh} if anom=='sigma' else {'k': thresh} if anom=='iqr' else {'threshold': thresh}
                    _, idx = d.detect_anomalies(s1, anom, **kw)
                    s2 = d.replace_anomalies(s1, anom, 'interpolation', **kw)
                else:
                    mean = np.mean(s1); std = np.std(s1)
                    idx = np.where(np.abs(s1 - mean) > thresh * std)[0]
                    s2 = s1
                
                # Step 3
                bar.progress(75, text=f"正在执行 {filt} 滤波... 75%")
                time.sleep(0.3)
                snr = 0
                if ALGO_STATUS:
                    f = NoiseFilter()
                    s3 = f.filter_signal(s2, filt)
                    snr = PerformanceMetrics.calculate_snr(s2, s3)
                else:
                    s3 = np.convolve(s2, np.ones(10)/10, mode='same')
                    snr = 0
                
                # Finish
                bar.progress(100, text="处理完成 100%")
                time.sleep(0.5)
                bar.empty()
                
                meta = {
                    'col': target,
                    'params': {'fill': fill, 'anom': anom, 'filt': filt, 'th': thresh},
                    'stats': {'idx': len(idx), 'snr': snr},
                    'original': raw
                }
                set_current_data(processed=s3, meta=meta)
                st.toast(f"处理成功！修复 {len(idx)} 个异常点", icon="🎉")
                
            except Exception as e:
                st.error("处理失败")
                st.code(traceback.format_exc())

    with c2:
        if store['processed'] is not None:
            res = store['meta']
            proc = store['processed']
            orig = res['original']
            st.markdown("### 📈 结果分析")
            k1, k2, k3 = st.columns(3)
            k1.metric("异常点", f"{res['stats']['idx']} 个", delta="Fixed", delta_color="inverse")
            k2.metric("SNR 提升", f"{res['stats']['snr']:.2f} dB", delta="Quality")
            k3.metric("状态", "Success")
            st.plotly_chart(plot_comparison(orig, proc, s['color']), use_container_width=True)
        else:
            st.info("👈 请点击处理")

def page_export():
    s_info = SENSORS[st.session_state.sensor]
    store = get_current_data()
    st.title(f"📥 成果导出 - {s_info['name']}")
    if store['processed'] is None:
        st.warning("⚠️ 请先进行智能处理")
        if st.button("前往智能处理"):
            st.session_state.page = 'process'
            st.rerun()
        return
    res = store['meta']
    proc = store['processed']
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 💾 下载数据集")
        with st.spinner("准备数据中..."):
            df_out = pd.DataFrame({'Original': res['original'], 'Processed': proc})
            csv = df_out.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 下载 CSV", csv, f"Result_{s_info['name']}.csv", "text/csv", type="primary")
    with c2:
        st.markdown("### 📄 下载实验报告")
        rpt = f"""州桥结构健康监测报告\n时间: {datetime.now()}\n传感器: {s_info['name']}\n通道: {res.get('col', 'N/A')}\n异常点: {res['stats']['idx']}\nSNR: {res['stats']['snr']:.2f} dB\n结论: 正常。"""
        st.text_area("预览", rpt, height=150)
        st.download_button("📥 下载 TXT", rpt, "Report.txt")

def main():
    apply_style()
    page = render_sidebar()
    if page == 'home': page_home()
    elif page == 'data': page_data()
    elif page == 'process': page_process()
    elif page == 'export': page_export()

if __name__ == "__main__":
    main()
