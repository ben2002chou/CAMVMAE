import os
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as T
from torchvision.utils import save_image
from multiprocessing import Pool
import pandas as pd

preprocess = T.Compose([T.Resize(224), T.CenterCrop(224), T.ToTensor()])


def extract_frame(input_video_path, target_fold, extract_frame_num=10):
    ext_len = len(input_video_path.split("/")[-1].split(".")[-1])
    video_id_parts = input_video_path.split("/")[-1][: -ext_len - 1].split("_")[1:]
    # Join the parts into one string, separated by "_"
    video_id = "_".join(video_id_parts)

    video_output_folder = os.path.join(target_fold, video_id, "video_frames")

    # Check if the video has already been processed
    if os.path.exists(video_output_folder):
        print(f"Skipping {video_id}, already processed.")
        return

    # Continue with frame extraction if not processed
    vidcap = cv2.VideoCapture(input_video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    total_frame_num = min(int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)), int(fps * 10))

    for i in range(extract_frame_num):
        frame_idx = int(i * (total_frame_num / extract_frame_num))
        print(
            f"Extract frame {i} from original frame {frame_idx}, total video frame {total_frame_num} at frame rate {int(fps)}."
        )
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 1)
        _, frame = vidcap.read()
        cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im)
        image_tensor = preprocess(pil_im)

        frame_folder = os.path.join(target_fold, video_id, f"frame_{i}")
        if not os.path.exists(frame_folder):
            os.makedirs(frame_folder)

        save_image(image_tensor, os.path.join(frame_folder, f"{video_id}.jpg"))


def process_videos(file_id):
    try:
        print(f"Processing video {file_id}: {input_filelist[file_id]}")
        extract_frame(input_filelist[file_id], args.target_fold)
    except Exception as e:
        print(f"Error with {input_filelist[file_id]}: {e}")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description="Python script to extract frames from a video, save as jpgs."
    )
    parser.add_argument(
        "-input_file_list",
        type=str,
        default="sample_video_extract_list.csv",
        help="Should be a csv file of a single column, each row is the input video path.",
    )
    parser.add_argument(
        "-target_fold",
        type=str,
        default="./sample_frames/",
        help="The place to store the video frames.",
    )
    args = parser.parse_args()

    input_filelist = pd.read_csv(args.input_file_list, header=None).squeeze().tolist()
    print(input_filelist)
    num_file = len(input_filelist)
    print(f"Total {num_file} videos are input")

    with Pool(10) as p:
        p.map(process_videos, range(num_file))

# (
#     extract_video_frame_urmp.py
#     - input_file_list / grand / EVITA / ben / URMP / URMP_file_paths.csv
#     - target_fold / grand / EVITA / ben / URMP / Dataset
# )
