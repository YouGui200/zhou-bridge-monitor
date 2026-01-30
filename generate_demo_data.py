# generate_demo_data.py (云端演示专用 - 修正版)
# 功能：生成与 App 传感器配置严格对应的演示数据
# 对应关系：应变(4路), 加速度(2路), 温度(1路), 位移(1路)

import numpy as np
import pandas as pd
import os

# 配置：只生成 1 小时数据，采样率适中，确保文件 < 5MB 且加载流畅
DURATION_HOURS = 1
HIGH_FS = 50  # 高频采样 (应变/加速度)
LOW_FS = 1    # 低频采样 (温度/位移)

def generate():
    print("正在生成符合传感器通道配置的演示数据...")
    
    # =========================================================================
    # 1. Strain (应变传感器) - 4个通道
    # =========================================================================
    # 时间轴
    t_high = np.linspace(0, 3600, 3600 * HIGH_FS)
    
    # 模拟不同位置的受力情况
    # S-01: 主受力，正弦波
    s1 = np.sin(t_high * 0.1) * 20 + 500
    # S-02: 略有相位差
    s2 = np.sin(t_high * 0.1 + 0.5) * 18 + 520
    # S-03: 包含更多高频噪声
    s3 = np.sin(t_high * 0.05) * 15 + 480 + np.random.normal(0, 2, len(t_high))
    # S-04: 平稳但有漂移
    s4 = np.linspace(450, 460, len(t_high)) + np.random.normal(0, 0.5, len(t_high))

    # --- 注入人工异常 (以便演示算法效果) ---
    # 在 S-01 中加入几个尖峰异常
    s1[1000:1005] = 800  
    s1[5000:5005] = 0
    # 在 S-02 中加入缺失值
    s2[2000:2050] = np.nan 

    df_strain = pd.DataFrame({
        'timestamp': t_high,
        'strain_S-01_micro': s1,
        'strain_S-02_micro': s2,
        'strain_S-03_micro': s3,
        'strain_S-04_micro': s4
    })
    # 保留一位小数，减小体积
    df_strain = df_strain.round(2)
    df_strain.to_csv('raw_data_strain.csv', index=False)
    print(f"  -> 生成 raw_data_strain.csv (4通道): {len(df_strain)} 行")

    # =========================================================================
    # 2. Accel (加速度传感器) - 2个通道
    # =========================================================================
    # A-01: 垂直振动
    a1 = np.sin(t_high * 5) * 0.2 + np.random.normal(0, 0.05, len(t_high))
    # A-02: 横向振动 (较小)
    a2 = np.sin(t_high * 3) * 0.1 + np.random.normal(0, 0.05, len(t_high))
    
    df_accel = pd.DataFrame({
        'timestamp': t_high,
        'accel_A-01_ms2': a1,
        'accel_A-02_ms2': a2
    })
    df_accel = df_accel.round(3)
    df_accel.to_csv('raw_data_acceleration.csv', index=False)
    print(f"  -> 生成 raw_data_acceleration.csv (2通道): {len(df_accel)} 行")

    # =========================================================================
    # 3. Temp (温度传感器) - 1个通道
    # =========================================================================
    t_low = np.linspace(0, 3600, 3600 * LOW_FS)
    # T-01: 缓慢上升
    temp = 25 + t_low * 0.0005 + np.random.normal(0, 0.02, len(t_low))
    
    df_temp = pd.DataFrame({
        'timestamp': t_low,
        'temperature_T-01_C': temp
    })
    df_temp = df_temp.round(2)
    df_temp.to_csv('raw_data_temperature.csv', index=False)
    print(f"  -> 生成 raw_data_temperature.csv (1通道): {len(df_temp)} 行")

    # =========================================================================
    # 4. Disp (位移传感器) - 1个通道
    # =========================================================================
    # D-01: 周期性变化
    disp = np.sin(t_low * 0.01) * 2 + 10 + np.random.normal(0, 0.05, len(t_low))
    
    df_disp = pd.DataFrame({
        'timestamp': t_low,
        'displacement_D-01_mm': disp
    })
    df_disp = df_disp.round(2)
    df_disp.to_csv('raw_data_displacement.csv', index=False)
    print(f"  -> 生成 raw_data_displacement.csv (1通道): {len(df_disp)} 行")
    
    print("\n✅ 所有演示数据生成完毕！请将生成的 CSV 文件上传至 GitHub。")

if __name__ == '__main__':
    generate()
