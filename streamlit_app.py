"""
================================================================================
州桥结构健康监测系统 - 云端部署版 (Final Fixed V2)
================================================================================
更新内容：
1. 下拉框修复：确保完整显示 strain_S-01_micro ~ S-04 等所有通道。
2. 手动输入优化：默认示例直接展示4个应变通道格式，方便测试。
3. 自动识别：算法自动读取CSV表头中的所有非时间列。
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

# 尝试导入算法库
try:
    from preprocessing_lib import (
        MissingValueHandler, NoiseFilter, AnomalyDetector, PerformanceMetrics
    )
    ALGO_STATUS = True
except ImportError:
    ALGO_STATUS = False

# -----------------------------------------------------------------------------
# 传感器配置
# -----------------------------------------------------------------------------
SENSORS = {
    'strain': {
        'name': '应变传感器', 
        'icon': '🔴', 
        'color': '#F44336', 
        'file': 'raw_data_strain.csv', 
        'unit': 'με', 
        'desc': '监测拱顶/拱脚受力 (4通道: S-01~S-04)'
    },
    'accel': {
        'name': '加速度传感器', 
        'icon': '🔵', 
        'color': '#2196F3', 
        'file': 'raw_data_acceleration.csv', 
        'unit': 'm/s²', 
        'desc': '监测桥面振动 (2通道: A-01~A-02)'
    },
    'temp': {
        'name': '温度传感器', 
        'icon': '🟢', 
        'color': '#4CAF50', 
        'file': 'raw_data_temperature.csv', 
        'unit': '°C', 
        'desc': '监测环境温度 (1通道: T-01)'
    },
    'disp': {
        'name': '位移传感器', 
        'icon': '🟣', 
        'color': '#9C27B0', 
        'file': 'raw_data_displacement.csv', 
        'unit': 'mm', 
        'desc': '监测桥墩沉降 (1通道: D-01)'
    }
}

# =============================================================================
# 2. 核心工具函数
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

def get_sensor_columns(df):
    """
    获取传感器数据列：
    自动过滤 time/date/timestamp/index/id 等无关列
    """
    if df is None: return []
    
    # 清理列名空格
    df.columns = df.columns.str.strip()
    
    cols = df.columns.tolist()
    # 排除关键词
    exclude_keywords = ['time', 'date', 'timestamp', 'unnamed', 'id', 'index']
    
    sensor_cols = []
    for c in cols:
        c_lower = c.lower()
        if not any(k in c_lower for k in exclude_keywords):
            sensor_cols.append(c)
            
    # 简单的字母数字排序
    sensor_cols.sort()
    return sensor_cols

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
    step = max(1, len(df) // 5000) # 智能降采样
    
    # 尝试找时间轴
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
    fig.update_layout(title=f"{title} - 通道: {col}", height=350, margin=dict(l=40,r=20,t=40,b=30), plot_bgcolor='white', 
                     xaxis=dict(showgrid=True, gridcolor='#eee'),
                     yaxis=dict(showgrid=True, gridcolor='#eee'))
    return fig

def plot_comparison(orig, proc, color, col_name):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, 
                        subplot_titles=(f"原始信号 ({col_name})", "预处理后信号"))
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
        # 显示中文名
        labels = [f"{SENSORS[k]['name']}" for k in opts]
        idx = st.radio("Sensor", range(len(opts)), format_func=lambda x: labels[x], label_visibility="collapsed")
        
        key = opts[idx]
        if key != st.session_state.sensor:
            st.session_state.sensor = key
            st.toast(f"已切换至 {SENSORS[key]['name']}", icon="🔄")
            time.sleep(0.3)
            st.rerun()
            
        cur = SENSORS[key]
        st.info(f"**类型:** {cur['name']}\n**单位:** {cur['unit']}")
        st.markdown("---")
        st.caption("MODULES")
        nav = {'home': '🏠 系统概览', 'data': '📊 数据管理', 'process': '⚡ 智能处理', 'export': '📥 成果导出'}
        page = st.radio("Nav", list(nav.keys()), format_func=lambda x: nav[x], label_visibility="collapsed")
        return page

def page_home():
    st.title("🏠 系统概览")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("传感器类型", "4 类")
    c2.metric("监测通道", "8 个", help="Strain(4) + Accel(2) + Temp(1) + Disp(1)")
    
    algo_text = "正常" if ALGO_STATUS else "未检测到库"
    c3.metric("算法引擎", algo_text, delta="Ready" if ALGO_STATUS else "Error")
    
    total = sum([len(v['data']) if v['data'] is not None else 0 for v in st.session_state.data_map.values()])
    c4.metric("总数据行数", f"{total:,}")
    
    st.markdown("---")
    st.caption("各传感器节点状态")
    
    cols = st.columns(4)
    for i, (k, s) in enumerate(SENSORS.items()):
        with cols[i]:
            has_data = st.session_state.data_map[k]['data'] is not None
            status_color = "#2ecc71" if has_data else "#95a5a6"
            status_text = "数据已加载" if has_data else "等待数据"
            
            st.markdown(f"""
            <div class="card" style="border-top-color:{s['color']}; text-align:center;">
                <h1 style="font-size: 3em; margin: 0;">{s['icon']}</h1>
                <h4 style="margin: 10px 0;">{s['name']}</h4>
                <p style="font-size: 0.85em; color: #666; height: 40px;">{s['desc']}</p>
                <p style="color:{status_color}; font-weight: bold;">● {status_text}</p>
            </div>
            """, unsafe_allow_html=True)

def page_data():
    s = SENSORS[st.session_state.sensor]
    store = get_current_data()
    st.title(f"📊 数据管理 - {s['name']}")
    
    c1, c2 = st.columns([1, 2])
    
    with c1:
        st.markdown("### 📥 数据加载")
        
        tab_auto, tab_manual = st.tabs(["📂 文件 / 演示", "✍️ 手动输入"])
        
        with tab_auto:
            st.caption(f"默认读取: {s['file']}")
            if st.button("🚀 加载演示数据", type="primary", use_container_width=True):
                path = os.path.join(DATA_PATH, s['file'])
                if os.path.exists(path):
                    with st.spinner("读取中..."):
                        df = load_csv_data(path)
                        set_current_data(data=df, processed=None)
                        st.success(f"已加载 {len(df)} 行数据")
                        time.sleep(0.5)
                        st.rerun()
                else:
                    st.error(f"未找到文件: {s['file']}")

            st.markdown("---")
            uploaded = st.file_uploader("上传 CSV 文件", type=['csv'])
            if uploaded:
                try:
                    df = pd.read_csv(uploaded)
                    set_current_data(data=df)
                    st.success(f"上传成功")
                    st.rerun()
                except Exception as e:
                    st.error(f"解析失败: {e}")

        with tab_manual:
            st.info("请粘贴 CSV 文本 (包含表头)")
            
            # --- 核心修改：针对不同传感器提供对应的默认示例 ---
            if st.session_state.sensor == 'strain':
                example = "timestamp,strain_S-01_micro,strain_S-02_micro,strain_S-03_micro,strain_S-04_micro\n2023-01-01,10.5,12.1,11.2,10.9\n2023-01-02,10.8,12.3,11.5,11.1"
            elif st.session_state.sensor == 'accel':
                example = "timestamp,accel_A-01,accel_A-02\n2023-01-01,0.01,0.02\n2023-01-02,0.03,0.01"
            else:
                example = "timestamp,value_1\n2023-01-01,10.5\n2023-01-02,11.2"
                
            manual_text = st.text_area("数据输入区", height=200, value=example, help="修改此处文本以测试不同通道")
            
            if st.button("解析文本数据", use_container_width=True):
                if manual_text.strip():
                    try:
                        df = pd.read_csv(io.StringIO(manual_text))
                        set_current_data(data=df)
                        st.toast("手动数据加载成功", icon="✅")
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
            
            # --- 获取所有传感器列 ---
            sensor_cols = get_sensor_columns(df)
            
            if len(sensor_cols) > 0:
                # 下拉框：内容完全取决于 CSV 表头 (Manual Input 的表头决定了这里显示什么)
                col = st.selectbox("选择传感器通道", sensor_cols)
                
                if col:
                    try:
                        st.plotly_chart(plot_paper_chart(df, col, s['color'], s['name']), use_container_width=True)
                    except Exception as e:
                        st.error(f"无法绘图: {e}")
            else:
                st.warning("未检测到有效的数据列 (表头需包含 S-01, A-01 等标识)")
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
    sensor_cols = get_sensor_columns(df)

    c1, c2 = st.columns([1, 2.5])
    with c1:
        st.markdown("### ⚙️ 算法配置")
        if not sensor_cols:
            st.error("没有可处理的数据列")
            return
        
        target = st.selectbox("1. 目标通道", sensor_cols)
        st.markdown("---")
        fill = st.selectbox("2. 缺失值处理", ['spline', 'linear', 'polynomial', 'nearest'])
        anom = st.selectbox("3. 异常检测", ['sigma', 'iqr', 'mad', 'isolation_forest'])
        
        if anom == 'sigma': thresh = st.slider("阈值 (n_sigma)", 1.0, 5.0, 3.0)
        elif anom == 'iqr': thresh = st.slider("阈值 (k)", 1.0, 3.0, 1.5)
        else: thresh = st.slider("阈值 (threshold)", 2.0, 5.0, 3.5)
        
        filt = st.selectbox("4. 滤波算法", ['wavelet', 'moving_average', 'gaussian', 'savgol'])
        st.markdown("---")
        
        if st.button("🚀 运行处理", type="primary", use_container_width=True):
            if not ALGO_STATUS:
                st.error("找不到 preprocessing_lib.py")
                return

            bar = st.progress(0, text="初始化...")
            
            try:
                # 预处理：转数值
                raw = pd.to_numeric(df[target], errors='coerce').values
                
                # Step 1
                bar.progress(30, text=f"填补缺失值 ({fill})...")
                time.sleep(0.2)
                h = MissingValueHandler()
                s1 = h.fill_missing(raw, fill)
                
                # Step 2
                bar.progress(60, text=f"检测异常值 ({anom})...")
                time.sleep(0.2)
                d = AnomalyDetector()
                kw = {}
                if anom == 'sigma': kw['n_sigma'] = thresh
                elif anom == 'iqr': kw['k'] = thresh
                else: kw['threshold'] = thresh
                
                _, idx = d.detect_anomalies(s1, anom, **kw)
                s2 = d.replace_anomalies(s1, anom, 'interpolation', **kw)
                
                # Step 3
                bar.progress(85, text=f"信号降噪 ({filt})...")
                time.sleep(0.2)
                f = NoiseFilter()
                s3 = f.filter_signal(s2, filt)
                snr = PerformanceMetrics.calculate_snr(s2, s3)
                
                bar.progress(100, text="完成")
                time.sleep(0.5)
                bar.empty()
                
                meta = {
                    'col': target,
                    'params': {'fill': fill, 'anom': anom, 'filt': filt, 'th': thresh},
                    'stats': {'idx': len(idx), 'snr': snr},
                    'original': raw
                }
                set_current_data(processed=s3, meta=meta)
                st.toast("处理成功", icon="✅")
                
            except Exception as e:
                st.error(f"运行出错: {e}")
                st.code(traceback.format_exc())

    with c2:
        if store['processed'] is not None:
            res = store['meta']
            if res.get('col') != target:
                st.warning(f"⚠️ 显示结果为通道 {res.get('col')}，请重新运行以更新")
            
            proc = store['processed']
            orig = res['original']
            
            st.markdown("### 📈 结果分析")
            k1, k2, k3 = st.columns(3)
            k1.metric("异常点数", f"{res['stats']['idx']}", delta="Detected")
            k2.metric("信噪比 (SNR)", f"{res['stats']['snr']:.2f} dB", delta="Quality")
            k3.metric("当前通道", target)
            
            st.plotly_chart(plot_comparison(orig, proc, s['color'], target), use_container_width=True)
        else:
            st.info("👈 请在左侧配置并运行")

def page_export():
    s_info = SENSORS[st.session_state.sensor]
    store = get_current_data()
    st.title(f"📥 成果导出 - {s_info['name']}")
    if store['processed'] is None:
        st.warning("⚠️ 请先进行智能处理")
        return
        
    res = store['meta']
    proc = store['processed']
    col_name = res.get('col', 'data')
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 💾 导出 CSV")
        df_out = pd.DataFrame({
            f'Original_{col_name}': res['original'], 
            f'Processed_{col_name}': proc
        })
        csv = df_out.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📥 下载数据 (CSV)",
            data=csv,
            file_name=f"Processed_{col_name}.csv",
            mime="text/csv",
            type="primary"
        )
    with c2:
        st.markdown("### 📄 导出报告")
        rpt = f"""州桥结构健康监测报告
-----------------------
时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
传感器: {s_info['name']}
通道: {col_name}
参数: {res['params']}
异常点: {res['stats']['idx']}
SNR提升: {res['stats']['snr']:.2f} dB
结论: 数据预处理完毕，质量符合要求。
"""
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
