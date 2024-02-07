import os
import csv

def generate_csv(main_folder_path, csv_file_path):
    # List to hold all the file paths
    file_paths = []

    # Walk through the main folder to get all mp4 file paths
    for root, dirs, files in os.walk(main_folder_path):
        for file in files:
            if file.endswith('.mp4'):
                # Construct the absolute path of the mp4 file
                abs_path = os.path.join(root, file)
                file_paths.append([abs_path])

    # Write the file paths to a csv file
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(file_paths)

# Specify the path to the main folder and the desired csv file path
main_folder_path = '/grand/EVITA/ben/AudioSet/eval/videos' # /grand/EVITA/ben/Audioset/videos
csv_file_path = '/home/ben2002chou/code/cav-mae/data/video_path_eval.csv'

# Call the function
generate_csv(main_folder_path, csv_file_path)
