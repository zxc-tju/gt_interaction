# read the map from text file [index, longitude, latitude].
# the index is the index of the point in the reference line.

import numpy as np
import matplotlib.pyplot as plt

def read_map(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    map_data = []
    for line in lines:
        index, longitude, latitude, vx, vy, yaw = map(float, line.split())

        map_data.append((index, longitude, latitude, vx, vy, yaw))
    
    return map_data

def convert_map(map_data, file_name):
    # 转换地图数据为参考线格式
    reference_line = []
    if file_name == "left_turn_ref":
        end_index = round(0.35*len(map_data))
    else:
        end_index = len(map_data)
    for _, longitude, latitude, _, _, _ in map_data[:end_index:10]:
        reference_line.append({'longitude': longitude, 'latitude': latitude})
    
    # 设置锚点
    anchor_point = {
        'latitude': 3483200,
        'longitude': 13492500
    }
    # 将经纬度四舍五入到小数点后3位
    for data in reference_line:
        data['latitude'] = round(float(data['latitude']) * 111320 - anchor_point['latitude'], 3)
        data['longitude'] = round(float(data['longitude']) * 111320 - anchor_point['longitude'], 3)
    
    # save the reference line to dataframe
    import pandas as pd
    df = pd.DataFrame(reference_line)
    df.to_csv(f'tj_campus/map/gps_data/{file_name}.csv', index=False)

    return reference_line

if __name__ == "__main__":
    file_name = "go_straight_ref"
    file_path = f"tj_campus\map\gps_data\{file_name}.txt"
    map_data = read_map(file_path)
    reference_line = convert_map(map_data, file_name)
    
    # 修改绘图代码
    plt.figure(figsize=(10, 6))
    longitudes = [point['longitude'] for point in reference_line]
    latitudes = [point['latitude'] for point in reference_line]
    plt.plot(longitudes, latitudes, label='Reference Line')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(f'Reference Line of {file_name}')
    plt.legend()
    plt.show()

