
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import cv2
import torch
import argparse
import numpy as np

from PIL import Image
from cotracker.utils.visualizer import Visualizer, read_video_from_path
from cotracker.predictor import CoTrackerPredictor

DEFAULT_DEVICE = ('cuda' if torch.cuda.is_available() else
                  'mps' if torch.backends.mps.is_available() else
                  'cpu')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_path",
        default="./assets/apple.mp4",
        help="path to a video",
    )
    parser.add_argument(
        "--mask_path",
        default="./assets/apple_mask.png",
        help="path to a segmentation mask",
    )
    parser.add_argument(
        "--checkpoint",
        default="./checkpoints/cotracker_stride_4_wind_8.pth",
        help="cotracker model",
    )
    parser.add_argument("--grid_size", type=int, default=0, help="Regular grid size")
    parser.add_argument(
        "--grid_query_frame",
        type=int,
        default=15,
        help="Compute dense and grid tracks starting from this frame ",
    )

    parser.add_argument(
        "--backward_tracking",
        action="store_true",
        help="Compute tracks in both directions, not only forward",
    )

    args = parser.parse_args()

    # load the input video frame by frame
    video = read_video_from_path(args.video_path)
    video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()
    segm_mask = np.array(Image.open(os.path.join(args.mask_path)))
    segm_mask = torch.from_numpy(segm_mask)[None, None]

    model = CoTrackerPredictor(checkpoint=args.checkpoint)
    model = model.to(DEFAULT_DEVICE)
    video = video.to(DEFAULT_DEVICE)

    #对视频分成多个小段，每个小段都进行预测
    # pred_tracks = []
    # pred_visibility = []
    # for i in range(0,video.shape[2],10):
    #     video_seg = video[:,:,i:i+10,:,:]
    #     pred_tracks_seg, pred_visibility_seg = model(
    #         video_seg,
    #         grid_size=args.grid_size,
    #         grid_query_frame=args.grid_query_frame,
    #         backward_tracking=args.backward_tracking,
    #         # segm_mask=segm_mask
    #     )
    #     print(f"computed{i}")
    #     #分段可视化并保存
    #     seq_name = args.video_path.split("/")[-1]
    #     vis = Visualizer(save_dir="./saved_videos", pad_value=120, linewidth=3,mode='optical_flow')
    #     vis.visualize(video_seg, pred_tracks_seg, pred_visibility_seg, query_frame=args.grid_query_frame,filename=f'water_flow_{i}.mp4')

    pred_tracks, pred_visibility = model(
        video,
        grid_size=args.grid_size,
        grid_query_frame=args.grid_query_frame,
        backward_tracking=args.backward_tracking,
        segm_mask=segm_mask
    )
    print("computed")

    # save a video with predicted tracks
    seq_name = args.video_path.split("/")[-1]
    vis = Visualizer(save_dir="./saved_videos", pad_value=120, linewidth=3)
    filename = f"water_{seq_name}"
    vis.visualize(video, pred_tracks, pred_visibility, query_frame=args.grid_query_frame,filename=filename)
