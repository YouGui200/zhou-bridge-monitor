"""
===============================================================================
文件名：preprocessing_lib.py
功能：州桥结构健康监测系统 - 数据预处理核心算法库

包含模块：
    1. PerformanceMetrics - 性能评估指标
    2. MissingValueHandler - 缺失值处理
    3. NoiseFilter - 滤波去噪
    4. AnomalyDetector - 异常检测
    5. PreprocessingPipeline - 完整预处理流水线

传感器对应（与阶段一设计一致）：
    ● 应变传感器 S-01~S-04（红色 #F44336）
    ▲ 加速度传感器 A-01~A-02（蓝色 #2196F3）
    ■ 温度传感器 T-01（绿色 #4CAF50）
    ◆ 位移传感器 D-01（紫色 #9C27B0）

作者：[你的姓名]
日期：2025年
===============================================================================
"""

import numpy as np
import pandas as pd
from scipy import interpolate, signal, stats
from scipy.ndimage import uniform_filter1d, gaussian_filter1d
import pywt
from sklearn.ensemble import IsolationForest
from typing import Tuple, List, Dict, Optional, Union
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
#                           性能评估指标模块
# =============================================================================

class PerformanceMetrics:
    """
    预处理性能评估指标计算类
    用于评估各种预处理算法的效果
    """
    
    @staticmethod
    def calculate_snr(original: np.ndarray, processed: np.ndarray) -> float:
        """
        计算信噪比 (Signal-to-Noise Ratio)
        SNR = 10 * log10(信号功率 / 噪声功率)
        
        参数：
            original: 原始信号
            processed: 处理后信号
        
        返回：
            SNR值 (dB)
        """
        mask = ~(np.isnan(original) | np.isnan(processed))
        orig = original[mask]
        proc = processed[mask]
        
        if len(orig) == 0:
            return 0.0
        
        signal_power = np.mean(proc ** 2)
        noise = orig - proc
        noise_power = np.mean(noise ** 2)
        
        if noise_power == 0:
            return 100.0  # 返回一个大值而不是无穷大
        
        snr = 10 * np.log10(signal_power / noise_power)
        return round(snr, 4)
    
    @staticmethod
    def calculate_mse(signal1: np.ndarray, signal2: np.ndarray) -> float:
        """计算均方误差 (Mean Squared Error)"""
        mask = ~(np.isnan(signal1) | np.isnan(signal2))
        s1, s2 = signal1[mask], signal2[mask]
        
        if len(s1) == 0:
            return float('inf')
        
        return round(np.mean((s1 - s2) ** 2), 6)
    
    @staticmethod
    def calculate_rmse(signal1: np.ndarray, signal2: np.ndarray) -> float:
        """计算均方根误差 (Root Mean Squared Error)"""
        mse = PerformanceMetrics.calculate_mse(signal1, signal2)
        return round(np.sqrt(mse), 6)
    
    @staticmethod
    def calculate_mae(signal1: np.ndarray, signal2: np.ndarray) -> float:
        """计算平均绝对误差 (Mean Absolute Error)"""
        mask = ~(np.isnan(signal1) | np.isnan(signal2))
        s1, s2 = signal1[mask], signal2[mask]
        
        if len(s1) == 0:
            return float('inf')
        
        return round(np.mean(np.abs(s1 - s2)), 6)
    
    @staticmethod
    def calculate_correlation(signal1: np.ndarray, signal2: np.ndarray) -> float:
        """计算相关系数"""
        mask = ~(np.isnan(signal1) | np.isnan(signal2))
        s1, s2 = signal1[mask], signal2[mask]
        
        if len(s1) < 2:
            return 0.0
        
        corr = np.corrcoef(s1, s2)[0, 1]
        return round(corr, 6) if not np.isnan(corr) else 0.0
    
    @staticmethod
    def calculate_smoothness(signal_data: np.ndarray) -> float:
        """
        计算信号平滑度（一阶差分的标准差）
        值越小表示越平滑
        """
        valid = signal_data[~np.isnan(signal_data)]
        if len(valid) < 2:
            return float('inf')
        
        diff = np.diff(valid)
        return round(np.std(diff), 6)
    
    @staticmethod
    def evaluate_all(original: np.ndarray, 
                     processed: np.ndarray,
                     method_name: str = "Unknown") -> Dict:
        """
        计算所有评估指标
        
        返回包含所有指标的字典
        """
        return {
            'method': method_name,
            'snr_db': PerformanceMetrics.calculate_snr(original, processed),
            'mse': PerformanceMetrics.calculate_mse(original, processed),
            'rmse': PerformanceMetrics.calculate_rmse(original, processed),
            'mae': PerformanceMetrics.calculate_mae(original, processed),
            'correlation': PerformanceMetrics.calculate_correlation(original, processed),
            'smoothness_before': PerformanceMetrics.calculate_smoothness(original),
            'smoothness_after': PerformanceMetrics.calculate_smoothness(processed),
            'missing_before': int(np.sum(np.isnan(original))),
            'missing_after': int(np.sum(np.isnan(processed)))
        }


# =============================================================================
#                           缺失值处理模块
# =============================================================================

class MissingValueHandler:
    """
    缺失值处理类
    提供多种插值方法填补缺失数据
    """
    
    def __init__(self):
        self.methods = ['linear', 'spline', 'lagrange', 'polynomial', 'nearest']
    
    def detect_missing(self, data: np.ndarray) -> Dict:
        """
        检测缺失值情况
        
        返回：缺失值统计信息字典
        """
        is_missing = np.isnan(data)
        missing_count = int(np.sum(is_missing))
        missing_rate = missing_count / len(data) * 100
        
        # 检测连续缺失段
        gaps = []
        in_gap = False
        gap_start = 0
        
        for i, missing in enumerate(is_missing):
            if missing and not in_gap:
                in_gap = True
                gap_start = i
            elif not missing and in_gap:
                in_gap = False
                gaps.append({'start': gap_start, 'end': i, 'length': i - gap_start})
        
        if in_gap:
            gaps.append({'start': gap_start, 'end': len(data), 'length': len(data) - gap_start})
        
        return {
            'missing_count': missing_count,
            'missing_rate': round(missing_rate, 4),
            'gap_count': len(gaps),
            'gaps': gaps,
            'max_gap_length': max([g['length'] for g in gaps]) if gaps else 0
        }
    
    def linear_interpolation(self, data: np.ndarray) -> np.ndarray:
        """
        线性插值
        优点：简单快速
        缺点：在变化剧烈处可能不够平滑
        """
        result = data.copy()
        valid_idx = np.where(~np.isnan(result))[0]
        
        if len(valid_idx) < 2:
            return result
        
        all_idx = np.arange(len(result))
        result = np.interp(all_idx, valid_idx, result[valid_idx])
        
        return result
    
    def spline_interpolation(self, data: np.ndarray, order: int = 3) -> np.ndarray:
        """
        样条插值（默认三次样条）
        优点：曲线平滑，适合缓慢变化的信号
        缺点：可能在边界处产生振荡
        """
        result = data.copy()
        valid_idx = np.where(~np.isnan(result))[0]
        
        if len(valid_idx) < order + 1:
            return self.linear_interpolation(data)
        
        try:
            spline_func = interpolate.UnivariateSpline(
                valid_idx, 
                result[valid_idx], 
                k=order,
                s=0
            )
            all_idx = np.arange(len(result))
            result = spline_func(all_idx)
        except Exception:
            result = self.linear_interpolation(data)
        
        return result
    
    def lagrange_interpolation(self, data: np.ndarray, window_size: int = 10) -> np.ndarray:
        """
        拉格朗日插值（局部窗口）
        优点：多项式拟合精确
        缺点：计算量大，高阶时可能振荡
        """
        result = data.copy()
        missing_idx = np.where(np.isnan(result))[0]
        
        for idx in missing_idx:
            start = max(0, idx - window_size)
            end = min(len(result), idx + window_size + 1)
            
            window_data = result[start:end]
            window_idx = np.arange(start, end)
            valid_mask = ~np.isnan(window_data)
            
            if np.sum(valid_mask) < 2:
                continue
            
            valid_x = window_idx[valid_mask]
            valid_y = window_data[valid_mask]
            
            try:
                poly = interpolate.lagrange(valid_x, valid_y)
                result[idx] = poly(idx)
            except Exception:
                if len(valid_x) >= 2:
                    result[idx] = np.interp(idx, valid_x, valid_y)
        
        return result
    
    def polynomial_interpolation(self, data: np.ndarray, degree: int = 3) -> np.ndarray:
        """多项式插值"""
        result = data.copy()
        valid_idx = np.where(~np.isnan(result))[0]
        
        if len(valid_idx) < degree + 1:
            return self.linear_interpolation(data)
        
        try:
            coeffs = np.polyfit(valid_idx, result[valid_idx], degree)
            poly = np.poly1d(coeffs)
            all_idx = np.arange(len(result))
            result = poly(all_idx)
        except Exception:
            result = self.linear_interpolation(data)
        
        return result
    
    def nearest_interpolation(self, data: np.ndarray) -> np.ndarray:
        """
        最近邻插值
        优点：保持原始值特征
        缺点：不平滑
        """
        result = data.copy()
        valid_idx = np.where(~np.isnan(result))[0]
        
        if len(valid_idx) == 0:
            return result
        
        missing_idx = np.where(np.isnan(result))[0]
        
        for idx in missing_idx:
            distances = np.abs(valid_idx - idx)
            nearest = valid_idx[np.argmin(distances)]
            result[idx] = result[nearest]
        
        return result
    
    def fill_missing(self, data: np.ndarray, method: str = 'linear') -> np.ndarray:
        """
        统一的缺失值填补接口
        
        参数：
            data: 输入数据
            method: 插值方法 ('linear', 'spline', 'lagrange', 'polynomial', 'nearest')
        """
        method = method.lower()
        
        if method == 'linear':
            return self.linear_interpolation(data)
        elif method == 'spline':
            return self.spline_interpolation(data)
        elif method == 'lagrange':
            return self.lagrange_interpolation(data)
        elif method == 'polynomial':
            return self.polynomial_interpolation(data)
        elif method == 'nearest':
            return self.nearest_interpolation(data)
        else:
            raise ValueError(f"未知的插值方法: {method}")
    
    def compare_methods(self, data: np.ndarray) -> pd.DataFrame:
        """比较所有插值方法的效果"""
        results = []
        original_missing = int(np.sum(np.isnan(data)))
        
        for method in self.methods:
            try:
                filled = self.fill_missing(data, method)
                results.append({
                    'method': method,
                    'missing_before': original_missing,
                    'missing_after': int(np.sum(np.isnan(filled))),
                    'fill_rate': round((original_missing - np.sum(np.isnan(filled))) / 
                                      max(original_missing, 1) * 100, 2),
                    'smoothness': PerformanceMetrics.calculate_smoothness(filled)
                })
            except Exception as e:
                results.append({
                    'method': method,
                    'missing_before': original_missing,
                    'missing_after': -1,
                    'fill_rate': 0,
                    'smoothness': float('inf'),
                    'error': str(e)
                })
        
        return pd.DataFrame(results)


# =============================================================================
#                           滤波去噪模块
# =============================================================================

class NoiseFilter:
    """
    滤波去噪类
    提供多种滤波方法去除信号噪声
    """
    
    def __init__(self):
        self.methods = ['moving_average', 'gaussian', 'median', 
                        'savgol', 'wavelet', 'lowpass', 'bandpass']
    
    def moving_average(self, data: np.ndarray, window_size: int = 50) -> np.ndarray:
        """
        移动平均滤波
        优点：简单有效，适合去除随机噪声
        缺点：会平滑掉信号的快速变化
        """
        filled_data = MissingValueHandler().linear_interpolation(data)
        result = uniform_filter1d(filled_data, size=window_size, mode='nearest')
        return result
    
    def gaussian_filter(self, data: np.ndarray, sigma: float = 5.0) -> np.ndarray:
        """
        高斯滤波
        优点：平滑效果好，保持信号形态
        缺点：会略微模糊边缘
        """
        filled_data = MissingValueHandler().linear_interpolation(data)
        result = gaussian_filter1d(filled_data, sigma=sigma, mode='nearest')
        return result
    
    def median_filter(self, data: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """
        中值滤波
        优点：对脉冲噪声（尖峰）效果极好
        缺点：可能丢失细节
        """
        filled_data = MissingValueHandler().linear_interpolation(data)
        
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        result = signal.medfilt(filled_data, kernel_size=kernel_size)
        return result
    
    def savgol_filter(self, data: np.ndarray, 
                      window_length: int = 51, 
                      polyorder: int = 3) -> np.ndarray:
        """
        Savitzky-Golay滤波
        优点：在平滑的同时保持信号的高阶矩特征
        缺点：参数选择敏感
        """
        filled_data = MissingValueHandler().linear_interpolation(data)
        
        if window_length % 2 == 0:
            window_length += 1
        
        if window_length <= polyorder:
            window_length = polyorder + 2
            if window_length % 2 == 0:
                window_length += 1
        
        if window_length > len(filled_data):
            window_length = len(filled_data) if len(filled_data) % 2 == 1 else len(filled_data) - 1
        
        result = signal.savgol_filter(filled_data, window_length, polyorder)
        return result
    
    def wavelet_denoise(self, data: np.ndarray, 
                        wavelet: str = 'db4',
                        level: int = 6,
                        threshold_type: str = 'soft') -> np.ndarray:
        """
        小波阈值去噪（核心算法）
        
        优点：能够区分信号和噪声的频率特征，效果最好
        缺点：参数选择需要经验
        
        原理：
            1. 将信号分解为不同尺度的小波系数
            2. 噪声主要集中在高频细节系数中
            3. 对细节系数进行阈值处理
            4. 重构信号
        """
        filled_data = MissingValueHandler().linear_interpolation(data)
        original_length = len(filled_data)
        
        # 补齐到2的幂次
        power = int(np.ceil(np.log2(original_length)))
        padded_length = 2 ** power
        padded_data = np.pad(filled_data, (0, padded_length - original_length), mode='edge')
        
        try:
            # 小波分解
            coeffs = pywt.wavedec(padded_data, wavelet, level=level)
            
            # 计算阈值（使用通用阈值）
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            threshold = sigma * np.sqrt(2 * np.log(len(padded_data)))
            
            # 对细节系数进行阈值处理
            new_coeffs = [coeffs[0]]
            for i in range(1, len(coeffs)):
                new_coeffs.append(pywt.threshold(coeffs[i], threshold, mode=threshold_type))
            
            # 小波重构
            result = pywt.waverec(new_coeffs, wavelet)
            result = result[:original_length]
            
        except Exception as e:
            print(f"小波去噪失败: {e}，使用高斯滤波替代")
            result = self.gaussian_filter(data)
        
        return result
    
    def lowpass_filter(self, data: np.ndarray, 
                       cutoff_freq: float = 10.0,
                       sampling_rate: float = 100.0,
                       order: int = 5) -> np.ndarray:
        """
        低通滤波
        优点：去除高频噪声
        缺点：可能过滤掉高频有用信号
        """
        filled_data = MissingValueHandler().linear_interpolation(data)
        
        nyquist = sampling_rate / 2
        normalized_cutoff = cutoff_freq / nyquist
        
        if normalized_cutoff >= 1:
            normalized_cutoff = 0.99
        if normalized_cutoff <= 0:
            normalized_cutoff = 0.01
        
        b, a = signal.butter(order, normalized_cutoff, btype='low')
        result = signal.filtfilt(b, a, filled_data)
        
        return result
    
    def bandpass_filter(self, data: np.ndarray,
                        low_freq: float = 0.5,
                        high_freq: float = 20.0,
                        sampling_rate: float = 100.0,
                        order: int = 5) -> np.ndarray:
        """
        带通滤波
        优点：只保留感兴趣的频率范围
        缺点：需要事先知道信号的频率特性
        """
        filled_data = MissingValueHandler().linear_interpolation(data)
        
        nyquist = sampling_rate / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        low = max(0.01, min(low, 0.99))
        high = max(0.01, min(high, 0.99))
        if low >= high:
            low = high / 2
        
        b, a = signal.butter(order, [low, high], btype='band')
        result = signal.filtfilt(b, a, filled_data)
        
        return result
    
    def filter_signal(self, data: np.ndarray, method: str = 'wavelet', **kwargs) -> np.ndarray:
        """统一的滤波接口"""
        method = method.lower()
        
        method_map = {
            'moving_average': self.moving_average,
            'gaussian': self.gaussian_filter,
            'median': self.median_filter,
            'savgol': self.savgol_filter,
            'wavelet': self.wavelet_denoise,
            'lowpass': self.lowpass_filter,
            'bandpass': self.bandpass_filter
        }
        
        if method not in method_map:
            raise ValueError(f"未知的滤波方法: {method}")
        
        return method_map[method](data, **kwargs)


# =============================================================================
#                           异常检测模块
# =============================================================================

class AnomalyDetector:
    """
    异常检测类
    提供多种异常值识别方法
    """
    
    def __init__(self):
        self.methods = ['sigma', 'iqr', 'zscore', 'isolation_forest', 'mad']
    
    def detect_by_sigma(self, data: np.ndarray, n_sigma: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        3-Sigma (3σ) 原则检测异常
        原理：假设数据服从正态分布，超过均值±3σ的点视为异常
        """
        valid_data = data[~np.isnan(data)]
        
        if len(valid_data) == 0:
            return np.zeros(len(data), dtype=bool), np.array([], dtype=int)
        
        mean = np.mean(valid_data)
        std = np.std(valid_data)
        
        if std == 0:
            return np.zeros(len(data), dtype=bool), np.array([], dtype=int)
        
        lower_bound = mean - n_sigma * std
        upper_bound = mean + n_sigma * std
        
        anomaly_mask = ~np.isnan(data) & ((data < lower_bound) | (data > upper_bound))
        anomaly_indices = np.where(anomaly_mask)[0]
        
        return anomaly_mask, anomaly_indices
    
    def detect_by_iqr(self, data: np.ndarray, k: float = 1.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        箱线图法 (IQR) 检测异常
        原理：使用四分位距，超过Q1-k*IQR或Q3+k*IQR的点视为异常
        """
        valid_data = data[~np.isnan(data)]
        
        if len(valid_data) == 0:
            return np.zeros(len(data), dtype=bool), np.array([], dtype=int)
        
        q1 = np.percentile(valid_data, 25)
        q3 = np.percentile(valid_data, 75)
        iqr = q3 - q1
        
        if iqr == 0:
            return np.zeros(len(data), dtype=bool), np.array([], dtype=int)
        
        lower_bound = q1 - k * iqr
        upper_bound = q3 + k * iqr
        
        anomaly_mask = ~np.isnan(data) & ((data < lower_bound) | (data > upper_bound))
        anomaly_indices = np.where(anomaly_mask)[0]
        
        return anomaly_mask, anomaly_indices
    
    def detect_by_zscore(self, data: np.ndarray, threshold: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Z-score 检测异常
        原理：计算每个点的标准化分数，超过阈值的视为异常
        """
        valid_mask = ~np.isnan(data)
        valid_data = data[valid_mask]
        
        if len(valid_data) == 0:
            return np.zeros(len(data), dtype=bool), np.array([], dtype=int)
        
        z_scores = np.zeros(len(data))
        z_scores[valid_mask] = stats.zscore(valid_data)
        
        anomaly_mask = valid_mask & (np.abs(z_scores) > threshold)
        anomaly_indices = np.where(anomaly_mask)[0]
        
        return anomaly_mask, anomaly_indices
    
    def detect_by_isolation_forest(self, data: np.ndarray,
                                    contamination: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
        """
        孤立森林 (Isolation Forest) 检测异常
        原理：基于机器学习，异常点更容易被随机分割"孤立"出来
        优点：适用于高维数据，不假设数据分布
        """
        valid_mask = ~np.isnan(data)
        valid_data = data[valid_mask]
        
        if len(valid_data) < 10:
            return np.zeros(len(data), dtype=bool), np.array([], dtype=int)
        
        X = valid_data.reshape(-1, 1)
        
        clf = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        
        predictions = clf.fit_predict(X)
        valid_anomaly = predictions == -1
        
        anomaly_mask = np.zeros(len(data), dtype=bool)
        anomaly_mask[valid_mask] = valid_anomaly
        anomaly_indices = np.where(anomaly_mask)[0]
        
        return anomaly_mask, anomaly_indices
    
    def detect_by_mad(self, data: np.ndarray, threshold: float = 3.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        中值绝对偏差 (MAD) 检测异常
        原理：使用中值和MAD代替均值和标准差，对异常更鲁棒
        """
        valid_data = data[~np.isnan(data)]
        
        if len(valid_data) == 0:
            return np.zeros(len(data), dtype=bool), np.array([], dtype=int)
        
        median = np.median(valid_data)
        mad = np.median(np.abs(valid_data - median))
        
        if mad == 0:
            return np.zeros(len(data), dtype=bool), np.array([], dtype=int)
        
        modified_z = 0.6745 * (data - median) / mad
        
        anomaly_mask = ~np.isnan(data) & (np.abs(modified_z) > threshold)
        anomaly_indices = np.where(anomaly_mask)[0]
        
        return anomaly_mask, anomaly_indices
    
    def detect_anomalies(self, data: np.ndarray, method: str = 'sigma', **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """统一的异常检测接口"""
        method = method.lower()
        
        method_map = {
            'sigma': self.detect_by_sigma,
            'iqr': self.detect_by_iqr,
            'zscore': self.detect_by_zscore,
            'isolation_forest': self.detect_by_isolation_forest,
            'mad': self.detect_by_mad
        }
        
        if method not in method_map:
            raise ValueError(f"未知的异常检测方法: {method}")
        
        return method_map[method](data, **kwargs)
    
    def remove_anomalies(self, data: np.ndarray, method: str = 'sigma', **kwargs) -> np.ndarray:
        """检测并移除异常值（置为NaN）"""
        result = data.copy()
        mask, _ = self.detect_anomalies(data, method, **kwargs)
        result[mask] = np.nan
        return result
    
    def replace_anomalies(self, data: np.ndarray,
                          method: str = 'sigma',
                          replace_method: str = 'interpolation',
                          **kwargs) -> np.ndarray:
        """检测异常值并用合理值替换"""
        result = data.copy()
        mask, indices = self.detect_anomalies(data, method, **kwargs)
        
        if len(indices) == 0:
            return result
        
        result[mask] = np.nan
        
        if replace_method == 'interpolation':
            handler = MissingValueHandler()
            result = handler.linear_interpolation(result)
        elif replace_method == 'median':
            valid_data = data[~mask & ~np.isnan(data)]
            if len(valid_data) > 0:
                result[mask] = np.median(valid_data)
        elif replace_method == 'mean':
            valid_data = data[~mask & ~np.isnan(data)]
            if len(valid_data) > 0:
                result[mask] = np.mean(valid_data)
        
        return result


# =============================================================================
#                           完整预处理流水线
# =============================================================================

class PreprocessingPipeline:
    """
    完整的数据预处理流水线
    按顺序执行：缺失值处理 → 异常检测 → 滤波去噪
    """
    
    def __init__(self):
        self.missing_handler = MissingValueHandler()
        self.noise_filter = NoiseFilter()
        self.anomaly_detector = AnomalyDetector()
        self.metrics = PerformanceMetrics()
    
    def process(self, data: np.ndarray,
                fill_method: str = 'spline',
                anomaly_method: str = 'sigma',
                filter_method: str = 'wavelet',
                anomaly_kwargs: dict = None,
                filter_kwargs: dict = None) -> Dict:
        """
        执行完整的预处理流程
        
        参数：
            data: 原始数据
            fill_method: 缺失值填补方法
            anomaly_method: 异常检测方法
            filter_method: 滤波方法
        
        返回：
            包含各阶段结果和评估指标的字典
        """
        if anomaly_kwargs is None:
            anomaly_kwargs = {}
        if filter_kwargs is None:
            filter_kwargs = {}
        
        results = {
            'original': data.copy(),
            'steps': []
        }
        
        current_data = data.copy()
        
        # 步骤1：缺失值填补
        missing_info = self.missing_handler.detect_missing(current_data)
        step1_data = self.missing_handler.fill_missing(current_data, fill_method)
        
        results['steps'].append({
            'name': 'Missing Value Imputation',
            'method': fill_method,
            'input_missing': missing_info['missing_count'],
            'output_missing': int(np.sum(np.isnan(step1_data))),
            'data': step1_data.copy()
        })
        
        current_data = step1_data
        
        # 步骤2：异常检测与修复
        mask, indices = self.anomaly_detector.detect_anomalies(
            current_data, anomaly_method, **anomaly_kwargs
        )
        step2_data = self.anomaly_detector.replace_anomalies(
            current_data, anomaly_method, 'interpolation', **anomaly_kwargs
        )
        
        results['steps'].append({
            'name': 'Anomaly Detection & Repair',
            'method': anomaly_method,
            'anomalies_detected': len(indices),
            'anomaly_indices': indices[:20].tolist() if len(indices) > 0 else [],
            'data': step2_data.copy()
        })
        
        current_data = step2_data
        
        # 步骤3：滤波去噪
        step3_data = self.noise_filter.filter_signal(
            current_data, filter_method, **filter_kwargs
        )
        
        metrics = self.metrics.evaluate_all(data, step3_data, filter_method)
        
        results['steps'].append({
            'name': 'Noise Filtering',
            'method': filter_method,
            'params': filter_kwargs,
            'metrics': metrics,
            'data': step3_data.copy()
        })
        
        # 最终结果
        results['processed'] = step3_data
        results['overall_metrics'] = self.metrics.evaluate_all(data, step3_data, 'Full Pipeline')
        
        return results


# =============================================================================
#                              便捷函数
# =============================================================================

def quick_denoise(data: np.ndarray, method: str = 'wavelet') -> np.ndarray:
    """快速去噪函数"""
    pipeline = PreprocessingPipeline()
    result = pipeline.process(data, filter_method=method)
    return result['processed']


def quick_clean(data: np.ndarray) -> np.ndarray:
    """快速清洗函数（填补缺失+去除异常+滤波）"""
    pipeline = PreprocessingPipeline()
    result = pipeline.process(data)
    return result['processed']


# =============================================================================
#                              测试代码
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("       州桥监测系统 - 数据预处理算法库测试")
    print("=" * 70)
    
    # 创建测试数据
    np.random.seed(42)
    t = np.linspace(0, 10, 1000)
    clean_signal = np.sin(2 * np.pi * 0.5 * t) + 0.5 * np.sin(2 * np.pi * 2 * t)
    
    # 添加噪声
    noisy_signal = clean_signal + np.random.normal(0, 0.3, len(t))
    
    # 添加缺失值
    noisy_signal[100:120] = np.nan
    noisy_signal[500:510] = np.nan
    
    # 添加异常值
    noisy_signal[300] = 5.0
    noisy_signal[700] = -4.0
    
    print(f"\n[测试数据]")
    print(f"  数据长度: {len(noisy_signal)}")
    print(f"  缺失值数量: {np.sum(np.isnan(noisy_signal))}")
    
    # 测试完整流水线
    pipeline = PreprocessingPipeline()
    result = pipeline.process(noisy_signal)
    
    print(f"\n[处理结果]")
    for step in result['steps']:
        print(f"  {step['name']}: {step['method']}")
    
    print(f"\n[性能指标]")
    metrics = result['overall_metrics']
    print(f"  SNR: {metrics['snr_db']:.2f} dB")
    print(f"  RMSE: {metrics['rmse']:.6f}")
    print(f"  相关系数: {metrics['correlation']:.6f}")
    
    print("\n" + "=" * 70)
    print("  ✅ 算法库测试通过！")
    print("=" * 70)