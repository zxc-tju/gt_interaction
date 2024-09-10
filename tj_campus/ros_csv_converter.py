# read all the csv files in a given directory and create a list of filenames
import os
import csv
import matplotlib.pyplot as plt

def read_csv_files(directory):
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    return csv_files

#read the csv file and convert it to a list of dictionaries
def read_csv_file(filename):
    with open(filename, 'r') as file:
        # get the gps position data in 'latitude', 'longitude', 'latitude_o', 'longitude_o' columns 
        gps_data = []
        reader = csv.DictReader(file)
        for row in reader:
            gps_data.append({
                'latitude': row['latitude'],
                'longitude': row['longitude'],
                'latitude_o': row['latitude_o'],
                'longitude_o': row['longitude_o']
            })
        # convert the gps data with the unit of degree to meter
        for data in gps_data:
            # set anchor point
            anchor_point = {
                'latitude': 3483200,
                'longitude': 13492500
            }
            # round the latitude and longitude to 3 decimal places
            data['latitude'] = round(float(data['latitude']) * 111320 - anchor_point['latitude'], 3)
            data['longitude'] = round(float(data['longitude']) * 111320 - anchor_point['longitude'], 3)  
            data['latitude_o'] = round(float(data['latitude_o']) * 111320 - anchor_point['latitude'], 3)
            data['longitude_o'] = round(float(data['longitude_o']) * 111320 - anchor_point['longitude'], 3)

    return gps_data

if __name__ == "__main__":
    directory = "tj_campus/ros_data/20240829_jql_xjc/lijianqiang"
    csv_files = read_csv_files(directory)
    # print(csv_files)    

    # iterate the csv_files and read the csv file   
    for csv_file in csv_files:
        gps_data = read_csv_file(f'{directory}/{csv_file}')
        # write the gps data to a csv file
        # create a new folder named 'gps_in_meter'
        os.makedirs(f'{directory}/gps_in_meter', exist_ok=True)
        with open(f'{directory}/gps_in_meter/gps_in_meter_{csv_file}', 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['latitude', 'longitude', 'latitude_o', 'longitude_o'])
            writer.writeheader()
            for data in gps_data:
                writer.writerow(data)

        plt.figure(figsize=(10, 6))
        
        latitudes = [data['latitude'] for data in gps_data]
        longitudes = [data['longitude'] for data in gps_data]
        latitudes_o = [data['latitude_o'] for data in gps_data]
        longitudes_o = [data['longitude_o'] for data in gps_data]
        
        plt.plot(latitudes, longitudes, label='GPS Data')
        plt.plot(latitudes_o, longitudes_o, label='GPS Data (O)')
        plt.xlabel('Latitude')
        plt.ylabel('Longitude')
        plt.title('GPS Data')
        # keep aspect ratio
        plt.gca().set_aspect('equal', adjustable='box')
        plt.legend()
        # plt.show()
        # save the plot as a png file
        plt.savefig(f'{directory}/gps_in_meter/gps_data_{csv_file[:-4]}.png')
        plt.close()  # 关闭图形，释放内存

