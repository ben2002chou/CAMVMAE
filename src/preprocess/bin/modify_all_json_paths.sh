#!/bin/bash

# Define the new base paths for audio and video files
NEW_DATA_PATH="/depot/yunglu/data/AudioSet"


# Define the directory containing the JSON files
JSON_DIR="/home/chou150/code/cav-mae-pl/cav-mae/data"

cd ..
# Process each JSON file
python create_json_as.py "$JSON_DIR/audioset_2m_filtered_piano_roll.json" "$JSON_DIR/audioset_2m_gilbreth.json" $NEW_DATA_PATH/unbalanced/audio_samples $NEW_DATA_PATH/unbalanced/video_frames
python create_json_as.py "$JSON_DIR/audioset_20k_filtered_piano_roll.json" "$JSON_DIR/audioset_20k_gilbreth.json" $NEW_DATA_PATH/balanced/audio_samples $NEW_DATA_PATH/balanced/video_frames
python create_json_as.py "$JSON_DIR/audioset_eval_filtered_piano_roll.json" "$JSON_DIR/audioset_eval_gilbreth.json" $NEW_DATA_PATH/eval/audio_samples $NEW_DATA_PATH/eval/video_frames

echo "JSON files processing complete."
