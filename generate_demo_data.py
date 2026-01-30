# generate_demo_data.py (云端演示专用)
import numpy as np
import pandas as pd
import os

# 配置：只生成 1 小时数据，采样率降低，确保文件 < 5MB
DURATION_HOURS = 1
HIGH_FS = 50  # 降采样率 (原100)
LOW_FS = 1

def generate():
    print("正在生成云端演示专用轻量数据...")
    
    # 1. Strain (应变)
    t = np.linspace(0, 3600, 3600*HIGH_FS)
    strain = np.sin(t*0.1) * 10 + np.random.normal(0, 0.5, len(t))
    df_strain = pd.DataFrame({'timestamp': t, 'strain_S-01_micro': strain})
    df_strain.to_csv('raw_data_strain.csv', index=False)
    
    # 2. Accel (加速度)
    accel = np.sin(t*5) * 0.5 + np.random.normal(0, 0.1, len(t))
    df_accel = pd.DataFrame({'timestamp': t, 'accel_A-01_m_s2': accel})
    df_accel.to_csv('raw_data_acceleration.csv', index=False)
    
    # 3. Temp (温度)
    t_low = np.linspace(0, 3600, 3600*LOW_FS)
    temp = 25 + t_low * 0.001 + np.random.normal(0, 0.05, len(t_low))
    df_temp = pd.DataFrame({'timestamp': t_low, 'temperature_T01_C': temp})
    df_temp.to_csv('raw_data_temperature.csv', index=False)
    
    # 4. Disp (位移)
    disp = np.sin(t_low*0.01) * 2 + np.random.normal(0, 0.1, len(t_low))
    df_disp = pd.DataFrame({'timestamp': t_low, 'displacement_D01_mm': disp})
    df_disp.to_csv('raw_data_displacement.csv', index=False)
    
    print("✅ 演示数据生成完毕！文件大小已优化。")

if __name__ == '__main__':
    generate()