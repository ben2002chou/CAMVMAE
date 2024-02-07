# -*- coding: utf-8 -*-
# @Author  : Benjamin Chou
# @Affiliation  : Purdue University, West Lafayette, IN, USA
# @Email   : chou150@purdue.edu
# @File    : create_json_as.py
import os
import json
import argparse

def clean_json(dataset_json_file, output_file, custom_audio_base_path, custom_video_base_path):
    new_data = []
    with open(dataset_json_file, "r") as fp:
        data_json = json.load(fp)
        data = data_json["data"]
        print("before clean {:d} files".format(len(data)))
        for entry in data:
            video_id = entry["video_id"]
            wav1 = os.path.join(custom_audio_base_path, video_id + ".wav")
            wav2 = entry["wav2"]
            video_path = custom_video_base_path
            labels = entry["labels"]
            new_entry = {
                "video_id": video_id,
                "wav1": wav1,
                "wav2": wav2,
                "video_path": video_path,
                "labels": labels
            }
            new_data.append(new_entry)

    output = {"data": new_data}
    print("after clean {:d} files".format(len(new_data)))

    output_file_name = output_file
    with open(output_file_name, "w") as f:
        json.dump(output, f, indent=1)

def main():
    parser = argparse.ArgumentParser(description='Clean JSON file paths.')
    parser.add_argument('json_file', help='Path to the JSON file to be processed')
    parser.add_argument('output_file', help='Path to the output JSON file')
    parser.add_argument('audio_path', help='New base path for audio files')
    parser.add_argument('video_path', help='New base path for video files')

    args = parser.parse_args()

    clean_json(args.json_file, args.output_file, args.audio_path, args.video_path)

if __name__ == "__main__":
    main()
