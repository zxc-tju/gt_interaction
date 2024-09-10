import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sys
from tqdm import tqdm

# 添加父目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import Agent, cal_individual_cost, cal_group_cost
from simulator import Simulator
from tools.utility import smooth_ployline, get_intersection_point
from scipy.spatial.distance import cdist
import seaborn as sns
from shapely.geometry import GeometryCollection, Point, LineString, Polygon, MultiPoint

csv_files = [f for f in os.listdir('tj_campus/ros_data/20240829_jql_xjc/chenxuejian/gps_in_meter') if f.endswith('.csv')]

# read map data for both left turn and go straight
left_turn_ref = pd.read_csv('tj_campus/map/gps_data/left_turn_ref.csv')
go_straight_ref = pd.read_csv('tj_campus/map/gps_data/go_straight_ref.csv')

def plot_gps_data_with_reference(df, left_turn_ref, go_straight_ref):
    """
    绘制GPS数据和参考线。

    参数:
    df (pd.DataFrame): 包含GPS数据的数据框
    left_turn_ref (pd.DataFrame): 包含左转参考线数据的数据框
    go_straight_ref (pd.DataFrame): 包含直行参考线数据的数据框
    """
    # 创建图形和坐标轴对象
    fig, ax = plt.subplots(figsize=(12, 8))

    # 确保参考线数据是 numpy 数组
    left_turn_lon = np.array(left_turn_ref['longitude'])
    left_turn_lat = np.array(left_turn_ref['latitude'])
    go_straight_lon = np.array(go_straight_ref['longitude'])
    go_straight_lat = np.array(go_straight_ref['latitude'])

    # 绘制参考线
    ax.plot(left_turn_lon, left_turn_lat, label='Left Turn Reference', color='red')
    ax.plot(go_straight_lon, go_straight_lat, label='Go Straight Reference', color='blue')

    # 绘制 GPS 数据
    ax.scatter(df['longitude'], df['latitude'], label='GPS Data', s=1, color='green', alpha=0.5)
    ax.scatter(df['longitude_o'], df['latitude_o'], label='GPS Data (O)', s=1, color='orange', alpha=0.5)

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Reference Lines and GPS Data')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)

    # 确保坐标轴比例相等
    ax.set_aspect('equal', 'box')

    # 调整布局并显示图形
    plt.tight_layout()
    plt.show()

def process_file(csv_file, i, data_path, save_path, left_turn_ref, go_straight_ref):
    df = pd.read_csv(f'{data_path}/{csv_file}')
    file_name = f'{save_path}/' + str(i) + '_ipv_results.xlsx'

    if 'longitude' not in df.columns or 'latitude' not in df.columns:
        print(f"Warning: 'longitude' or 'latitude' columns not found in file {csv_file}")
        print("Available columns:", df.columns)
        return None

    # 数据预处理
    for col in ['longitude', 'latitude', 'longitude_o', 'latitude_o']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['longitude', 'latitude', 'longitude_o', 'latitude_o'])

    # 设置位置信息
    if i < 15:
        gs_position = df[['longitude', 'latitude']].values
        lt_position = df[['longitude_o', 'latitude_o']].values
    else:
        gs_position = df[['longitude_o', 'latitude_o']].values
        lt_position = df[['longitude', 'latitude']].values
    
    # 计算速度和航向
    delta_gs = np.diff(gs_position, axis=0)
    delta_lt = np.diff(lt_position, axis=0)
    
    velocity_x_gs, velocity_y_gs = delta_gs[:, 0] / 0.1, delta_gs[:, 1] / 0.1
    velocity_x_lt, velocity_y_lt = delta_lt[:, 0] / 0.1, delta_lt[:, 1] / 0.1
    
    heading_gs = np.arctan2(velocity_y_gs, velocity_x_gs)
    heading_lt = np.arctan2(velocity_y_lt, velocity_x_lt)

    gs_info = np.column_stack((gs_position[:-1], velocity_x_gs, velocity_y_gs, heading_gs))
    lt_info = np.column_stack((lt_position[:-1], velocity_x_lt, velocity_y_lt, heading_lt))

    lt_ref = left_turn_ref[['longitude', 'latitude']].values
    gs_ref = go_straight_ref[['longitude', 'latitude']].values

    # 计算 IPV
    ipv_collection = np.zeros_like(lt_info[:, 0:2])
    ipv_error_collection = np.ones_like(lt_info[:, 0:2])
    
    t_max = min(np.size(lt_info, 0), np.size(gs_info, 0))
    t_max = min(t_max, 140)  # 限制最大时间步数为140
    
    for t in tqdm(range(4, t_max), desc=f"Estimating IPV for file {i+1}/{len(csv_files)}", leave=False):
        start_time = max(0, t - 10)

        # 左转车辆
        init_position_lt = lt_info[start_time, 0:2]
        init_velocity_lt = lt_info[start_time, 2:4]
        init_heading_lt = lt_info[start_time, 4]
        agent_lt = Agent(init_position_lt, init_velocity_lt, init_heading_lt, 'tj_campus_lt')
        agent_lt.reference = lt_ref
        lt_track = lt_info[start_time:t + 1, 0:2]

        # 直行车辆
        init_position_gs = gs_info[start_time, 0:2]
        init_velocity_gs = gs_info[start_time, 2:4]
        init_heading_gs = gs_info[start_time, 4]
        agent_gs = Agent(init_position_gs, init_velocity_gs, init_heading_gs, 'tj_campus_gs')
        agent_gs.reference = gs_ref
        gs_track = gs_info[start_time:t + 1, 0:2]

        # 估计 IPV
        agent_lt.estimate_self_ipv(lt_track, gs_track)
        ipv_collection[t, 0] = agent_lt.ipv
        ipv_error_collection[t, 0] = agent_lt.ipv_error

        agent_gs.estimate_self_ipv(gs_track, lt_track)
        ipv_collection[t, 1] = agent_gs.ipv
        ipv_error_collection[t, 1] = agent_gs.ipv_error

    # 保存数据
    df_ipv_lt = pd.DataFrame(ipv_collection[:, 0], columns=["ipv_lt"])
    df_ipv_lt_error = pd.DataFrame(ipv_error_collection[:, 0], columns=["ipv_lt_error"])
    df_motion_lt = pd.DataFrame(lt_info[:, 0:5], columns=["lt_px", "lt_py", "lt_vx", "lt_vy", "lt_heading"])

    df_ipv_gs = pd.DataFrame(ipv_collection[:, 1], columns=["ipv_gs"])
    df_ipv_gs_error = pd.DataFrame(ipv_error_collection[:, 1], columns=["ipv_gs_error"])
    df_motion_gs = pd.DataFrame(gs_info[:, 0:5], columns=["gs_px", "gs_py", "gs_vx", "gs_vy", "gs_heading"])

    with pd.ExcelWriter(file_name) as writer:
        df_ipv_lt.to_excel(writer, startcol=0, index=False)
        df_ipv_lt_error.to_excel(writer, startcol=1, index=False)
        df_motion_lt.to_excel(writer, startcol=2, index=False)

        df_ipv_gs.to_excel(writer, startcol=7, index=False)
        df_ipv_gs_error.to_excel(writer, startcol=8, index=False)
        df_motion_gs.to_excel(writer, startcol=9, index=False)

    # 返回可视化所需的数据，而不是直接绘图
    return lt_info, gs_info, lt_ref, gs_ref, ipv_collection, ipv_error_collection

def visualize_results(i, lt_info, gs_info, lt_ref, gs_ref, ipv_collection, ipv_error_collection, save_path):
    plt.close('all')  # 关闭所有现有的图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[16, 8])
    ax1.set(ylim=[-2, 2])

    x_range = range(np.size(lt_info, 0))

    smoothed_ipv_lt, _ = smooth_ployline(np.array([x_range, ipv_collection[x_range, 0]]).T)
    smoothed_ipv_error_lt, _ = smooth_ployline(np.array([x_range, ipv_error_collection[x_range, 0]]).T)
    smoothed_x = smoothed_ipv_lt[:, 0]
    # plot ipv
    ax1.plot(smoothed_x, smoothed_ipv_lt[:, 1], 'blue')
    # plot error bar
    ax1.fill_between(smoothed_x, smoothed_ipv_lt[:, 1] - smoothed_ipv_error_lt[:, 1],
                        smoothed_ipv_lt[:, 1] + smoothed_ipv_error_lt[:, 1],
                        alpha=0.4,
                        color='blue',
                        label='estimated lt IPV')

    smoothed_ipv_gs, _ = smooth_ployline(np.array([x_range, ipv_collection[x_range, 1]]).T)
    smoothed_ipv_error_gs, _ = smooth_ployline(np.array([x_range, ipv_error_collection[x_range, 1]]).T)
    # plot ipv
    ax1.plot(smoothed_x, smoothed_ipv_gs[:, 1], 'red')
    # plot error bar
    ax1.fill_between(smoothed_x, smoothed_ipv_gs[:, 1] - smoothed_ipv_error_gs[:, 1],
                        smoothed_ipv_gs[:, 1] + smoothed_ipv_error_gs[:, 1],
                        alpha=0.4,
                        color='red',
                        label='estimated gs IPV')
    ax1.legend()

    ax2.plot(gs_info[:,0], gs_info[:,1], color='red',label='GS Ground Truth')
    ax2.scatter(gs_info[0,0], gs_info[0,1], color='red',label='GS Start Point')
    ax2.plot(lt_info[:,0], lt_info[:,1], color='blue',label='LT Ground Truth')
    ax2.scatter(lt_info[0,0], lt_info[0,1], color='blue',label='LT Start Point')
    ax2.plot(gs_ref[:,0], gs_ref[:,1], color='red',linestyle='--',label='GS Reference')
    ax2.plot(lt_ref[:,0], lt_ref[:,1], color='blue',linestyle='--',label='LT Reference')
    ax2.legend()

    plt.savefig(f'{save_path}/' + str(i) + '_ipv_results.png')
    plt.close(fig)

def main():
    data_path = 'tj_campus/ros_data/20240829_jql_xjc/chenxuejian/gps_in_meter'
    save_path = 'tj_campus/ros_data/20240829_jql_xjc/chenxuejian/ipv_results'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
    left_turn_ref = pd.read_csv('tj_campus/map/gps_data/left_turn_ref.csv')
    go_straight_ref = pd.read_csv('tj_campus/map/gps_data/go_straight_ref.csv')

    for i, csv_file in enumerate(tqdm(csv_files, desc="Processing files")):
        result = process_file(csv_file, i, data_path, save_path, left_turn_ref, go_straight_ref)
        if result is not None:
            lt_info, gs_info, lt_ref, gs_ref, ipv_collection, ipv_error_collection = result
            visualize_results(i, lt_info, gs_info, lt_ref, gs_ref, ipv_collection, ipv_error_collection, save_path)

    print("All files processed successfully.")

if __name__ == "__main__":
    main()


