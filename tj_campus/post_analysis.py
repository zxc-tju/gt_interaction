import os
import pandas as pd
import numpy as np
import sys
# 添加父目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import Agent, cal_individual_cost, cal_group_cost
from simulator import Simulator
from tools.utility import smooth_ployline, get_intersection_point
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from tqdm import tqdm
import seaborn as sns
from shapely.geometry import GeometryCollection, Point, LineString, Polygon, MultiPoint


def find_cp(trajectory1, trajectory2):
    # Compute all pairwise distances between points in the two trajectories
    distances = cdist(trajectory1, trajectory2)

    # Find the minimum distance and corresponding indices
    min_dist_idx = np.unravel_index(distances.argmin(), distances.shape)
    return min_dist_idx, distances[min_dist_idx]
    
def get_first_point(geometry):
    if isinstance(geometry, Point):
        return np.array([geometry.x, geometry.y])
    elif isinstance(geometry, MultiPoint):
        if len(geometry.geoms) > 0:
            return np.array([geometry.geoms[0].x, geometry.geoms[0].y])
    elif isinstance(geometry, (LineString, Polygon)):
        return np.array([Point(geometry.coords[0]).x, Point(geometry.coords[0]).y])
    elif isinstance(geometry, GeometryCollection):
        for geom in geometry.geoms:
            first_point = get_first_point(geom)
            if first_point[0]:
                return first_point
    return None

def cal_pet_with_ref_path(trj_a, trj_b, ref_a, ref_b, type_cal='pet'):
    """
    Calculate the PET of two given trajectory and according reference path
    计算两条给定轨迹的PET或APET
    Parameters
    ----------
    trj_a
    trj_b
    type_cal: 'pet'或'apet'


    """

    smoothed_ref_b, _ = smooth_ployline(ref_b, point_num=100)
    smoothed_ref_a, _ = smooth_ployline(ref_a, point_num=100)
    
    "find the conflict point"
    "寻找两条轨迹的冲突点"
    # find the most closed point
    (index_a,index_b), min_dist = find_cp(trj_a, trj_b)
    conflict_point = (trj_a[index_a, :] + trj_b[index_b, :])/2
    # print(conflict_point)
    # print('min_dist', dist)
    min_dist_nominal = min_dist
    if min_dist > 1:  # if the min distance is too large, refer to the coross point of the ref path
        (index_a,index_b_ref), dist_a_to_b_ref = find_cp(trj_a, smoothed_ref_b)
        if dist_a_to_b_ref < min_dist:
            min_dist_nominal = dist_a_to_b_ref
            conflict_point = (trj_a[index_a, :] + smoothed_ref_b[index_b_ref, :])/2
            
        (index_a_ref,index_b), dist_b_to_a_ref = find_cp(smoothed_ref_a, trj_b)
        if dist_b_to_a_ref < dist_a_to_b_ref:
            min_dist_nominal = dist_b_to_a_ref
            conflict_point = (smoothed_ref_a[index_a_ref, :] + trj_b[index_b, :])/2
   

    if min_dist_nominal > 1:
        conflict_point_of_path_str = get_intersection_point(ref_a, ref_b)  # by ref path
            
        if conflict_point_of_path_str.is_empty:  # there is no intersection between given polylines
            print('No Interaction')
            if type_cal == 'pet':
                return 10, None
            elif type_cal == 'apet':
                return None, None, None
        else:
            conflict_point = get_first_point(conflict_point_of_path_str)

    # "find the point that most closed to cp in each trajectory"
    # "寻找两条轨迹各自距离冲突点最近的轨迹点"
    # smoothed_trj_a, smoothed_progress_a = smooth_ployline(trj_a, point_num=100)
    # cp2trj_a = np.linalg.norm(smoothed_trj_a - conflict_point, axis=1)
    # min_dcp2trj_a = np.amin(cp2trj_a)
    # cp_index_a = np.where(min_dcp2trj_a == cp2trj_a)

    # smoothed_trj_b, smoothed_progress_b = smooth_ployline(trj_b, point_num=100)
    # cp2trj_b = np.linalg.norm(smoothed_trj_b - conflict_point, axis=1)
    # min_dcp2trj_b = np.amin(cp2trj_b)
    # cp_index_b = np.where(min_dcp2trj_b == cp2trj_b)

    "从初始时刻到冲突点所需的纵向行程"
    smoothed_ref_a, smoothed_progress_a = smooth_ployline(ref_a, point_num=100)
    smoothed_ref_b, smoothed_progress_b = smooth_ployline(ref_b, point_num=100)
    # print(ref_b)
    # print(smoothed_progress_b)
    # print(smoothed_ref_b)
    
    cp2ref_a = np.linalg.norm(smoothed_ref_a - conflict_point, axis=1)
    min_dcp2ref_a = np.amin(cp2ref_a)
    cp_index_a = np.where(min_dcp2ref_a == cp2ref_a)

    cp2ref_b = np.linalg.norm(smoothed_ref_b - conflict_point, axis=1)
    min_dcp2ref_b = np.amin(cp2ref_b)
    cp_index_b = np.where(min_dcp2ref_b == cp2ref_b)

    init2ref_a = np.linalg.norm(smoothed_ref_a - trj_a[0,:], axis=1)
    min_dinit2ref_a = np.amin(init2ref_a)
    init_index_a = np.where(min_dinit2ref_a == init2ref_a)

    init2ref_b = np.linalg.norm(smoothed_ref_b - trj_b[0,:], axis=1)
    min_dinit2ref_b = np.amin(init2ref_b)
    init_index_b = np.where(min_dinit2ref_b == init2ref_b)

    progress_2_cp_a = smoothed_progress_a[cp_index_a] - smoothed_progress_a[init_index_a]
    progress_2_cp_b = smoothed_progress_b[cp_index_b] - smoothed_progress_b[init_index_b]
    
    
    "calculate time to cp"
    "计算达到冲突点的时间差"
    seg_len_a = np.linalg.norm(trj_a[1:, :] - trj_a[:-1, :], axis=1)
    seg_len_b = np.linalg.norm(trj_b[1:, :] - trj_b[:-1, :], axis=1)
    vel_a = seg_len_a / 0.12
    vel_b = seg_len_b / 0.12
    seg_len_a = np.concatenate([np.array([0]), seg_len_a])
    seg_len_b = np.concatenate([np.array([0]), seg_len_b])
    longi_progress_a = np.cumsum(seg_len_a)
    longi_progress_b = np.cumsum(seg_len_b)
    

    # dis2conf_a = -(longi_progress_a - smoothed_progress_a[cp_index_a])
    # dis2conf_b = -(longi_progress_b - smoothed_progress_b[cp_index_b])
    dis2conf_a = -(longi_progress_a - progress_2_cp_a)
    # print(dis2conf_a)
    dis2conf_b = -(longi_progress_b - progress_2_cp_b)
    # print(dis2conf_b)

    ttcp_a = dis2conf_a[:-1] / vel_a  # ttcp:time to conflict point
    ttcp_b = dis2conf_b[:-1] / vel_b

    solid_len = min(np.size(ttcp_a[ttcp_a > 0], 0), np.size(ttcp_b[ttcp_b > 0], 0))
    if solid_len == 0:
        solid_len = 1
    "PET and APET"
    apet = np.abs(ttcp_a[:solid_len] - ttcp_b[:solid_len])

    pet = max(ttcp_a[solid_len - 1], ttcp_b[solid_len - 1]) - min(ttcp_a[solid_len - 1], ttcp_b[solid_len - 1])

    if type_cal == 'pet':
        return pet, conflict_point
    elif type_cal == 'apet':
        return apet, ttcp_a, ttcp_b
    
result_file_path = 'tj_campus/ros_data/20240829_jql_xjc/chenxuejian/ipv_results'

# read map data for both left turn and go straight
left_turn_ref = pd.read_csv('tj_campus/map/gps_data/left_turn_ref.csv')
go_straight_ref = pd.read_csv('tj_campus/map/gps_data/go_straight_ref.csv')
lt_ref = left_turn_ref[['longitude', 'latitude']].values
gs_ref = go_straight_ref[['longitude', 'latitude']].values

solid_interaction_number = 0
ipv_collection = []
lt_ipv_with_error = []
gs_ipv_with_error = []
PET_collection = []
APET_collection = []

# 尝试不同的编码
encodings = ['utf-8', 'iso-8859-1', 'cp1252', 'gb18030']

# iterate through all csv files in the result_file_path
for file in tqdm(os.listdir(result_file_path), desc="Processing files"):
    if file.endswith('.xlsx'):
        flag_gs = 1
        flag_lt = 1 
        # 尝试不同的编码读取文件
        for encoding in encodings:
            try:
                results = pd.read_excel(os.path.join(result_file_path, file), engine='openpyxl')
                break  # 如果成功读取，跳出循环
            except Exception as e:
                print(f"Failed to read {file} with {encoding} encoding: {str(e)}")
                if encoding == encodings[-1]:
                    print(f"Unable to read {file} with any encoding. Skipping this file.")
                    continue
        
        ipv_result = results[['ipv_lt', 'ipv_lt_error','ipv_gs', 'ipv_gs_error']].values
        ipv_result = ipv_result[6:,]  # ipv is estimated with observation of at least 6 steps
        # filter ipv
        # find the index where the left-turn vehicles' ipv is not zero
        solid_lt_ipv_index = np.where(ipv_result[:, 1] < 6e-1)[0]
        if solid_lt_ipv_index.size > 0:
            solid_lt_ipv = ipv_result[solid_lt_ipv_index, 0:2]
        else:
            flag_lt = 0  # left-turn vehicle is not influenced in the event
        # find the index where the go-straight vehicles' ipv is not zero
        solid_gs_ipv_index = np.where(ipv_result[:, 3] < 6e-1)[0]
        if solid_gs_ipv_index.size > 0:
            solid_gs_ipv = ipv_result[solid_gs_ipv_index, 2:]
        else:
            flag_gs = 0  # go-straight vehicle is not influenced in the event

        # get trajectory data for PET calculation
        trj_result = results[['lt_px', 'lt_py','gs_px', 'gs_py']].values
        # create a mask to identify rows with NaN
        mask = ~np.isnan(trj_result).any(axis=1)
        # apply the mask to filter out the rows with NaN
        trj_result = trj_result[mask]
        solid_range = min(np.size(trj_result[:,0:2], 0), np.size(trj_result[:,2:], 0))

        # calculate PET
        PET, cf_point = cal_pet_with_ref_path(trj_result[:solid_range,0:2], trj_result[:solid_range,2:], lt_ref, gs_ref, 'pet')
        APET, ttcp_a, ttcp_b = cal_pet_with_ref_path(trj_result[:solid_range,0:2], trj_result[:solid_range,2:], lt_ref, gs_ref, 'apet')
        if cf_point is None:
            print(file)
            continue
        
        if flag_lt and flag_gs:
            solid_interaction_number += 1
            ipv_collection.append([np.mean(solid_lt_ipv[:,0]), np.mean(solid_gs_ipv[:,0])])
            lt_ipv_with_error.append([np.mean(solid_lt_ipv[:,0]),np.mean(solid_lt_ipv[:,1])])
            gs_ipv_with_error.append([np.mean(solid_gs_ipv[:,0]),np.mean(solid_gs_ipv[:,1])])
            PET_collection.append(min(PET, 10))
            # init_APET_collection[gs_type][lt_type][result_type].append(min(APET[0], 10))
            APET_collection.append(APET)

# print(solid_interaction_number)
# print(ipv_collection)
# print(PET_collection)
# print(APET_collection)
#plot the ipv_with_error
plt.figure(figsize=(10, 6))

# 确保 lt_ipv_with_error 和 gs_ipv_with_error 是 numpy 数组
lt_ipv_with_error = np.array(lt_ipv_with_error)
gs_ipv_with_error = np.array(gs_ipv_with_error)

# 检查数组是否为空
if len(lt_ipv_with_error) == 0 or len(gs_ipv_with_error) == 0:
    print("警告：一个或两个 IPV 误差数组为空。无法绘图。")
else:
    # 绘制左转 IPV 误差
    plt.scatter(range(1,16), lt_ipv_with_error[:15, 0], label='左转', alpha=0.7)
    plt.plot(range(1,16), lt_ipv_with_error[:15, 0], label='左转', alpha=0.7)

    
    # 绘制直行 IPV 误差
    plt.scatter(range(16,31), gs_ipv_with_error[15:, 0], label='直行', alpha=0.7)
    plt.plot(range(16,31), gs_ipv_with_error[15:, 0], label='直行', alpha=0.7)

    
    plt.xlabel('索引')
    plt.ylabel('IPV 误差')
    plt.title('左转和直行车辆的 IPV 误差')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 保存图片
    plt.savefig(result_file_path+'/ipv_errors.png')
    print("图片已保存为 'ipv_errors.png'")
    
    # 显示图片（如果在交互式环境中）
    plt.show()

# 关闭图形以释放内存
plt.close()


