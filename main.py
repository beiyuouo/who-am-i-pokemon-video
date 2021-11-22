#!/usr/bin/env python
# -*- encoding: utf-8 -*-
""" 
@File    :   main.py 
@Time    :   2021-11-21 22:39:20 
@Author  :   Bingjie Yan 
@Email   :   bj.yan.pa@qq.com 
@License :   Apache License 2.0 
"""

import os
import argparse
import cv2
import numpy as np

from ffmpy3 import FFmpeg

parser = argparse.ArgumentParser()
parser.add_argument('--template',
                    type=str,
                    default=os.path.join('template', 'who-am-i.mp4'),
                    help='Path to the video')
parser.add_argument('--output_path',
                    type=str,
                    default=os.path.join('output'),
                    help='Path to the output video')
parser.add_argument('--output',
                    type=str,
                    default=os.path.join('output', 'output.mp4'),
                    help='Path to the output video')
parser.add_argument('--images',
                    type=str,
                    default=os.path.join('pokemon'),
                    help='Path to the images')
parser.add_argument('--fps', type=int, default=24, help='FPS of output video')
parser.add_argument('--width', type=int, default=1920, help='Width of output video')
parser.add_argument('--height', type=int, default=1080, help='Height of output video')

args = parser.parse_args()

black_image_in = 2.0
black_image_out = 5.8
raw_image_in = 7.25
raw_image_out = 11.3

bg_color = (109, 209, 138)
vertex_point = (210, 390)


def deal(raw_image, mask_image, image_name):
    video = cv2.VideoCapture(args.template)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_path = os.path.join(args.output_path, f"{image_name.split('.')[0]}.mp4")
    temp_video_path = os.path.join(args.output_path, f"{image_name.split('.')[0]}_temp.mp4")
    video_writer = cv2.VideoWriter(video_path, fourcc, args.fps,
                                   (int(args.width), int(args.height)))

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        frame = cv2.resize(frame, (int(args.width), int(args.height)))

        cur_time = video.get(cv2.CAP_PROP_POS_MSEC) / 1000
        if cur_time < black_image_in:
            video_writer.write(frame)
            continue
        if cur_time < black_image_out:
            frame[vertex_point[0]:vertex_point[0] + raw_image.shape[0],
                  vertex_point[1]:vertex_point[1] + raw_image.shape[1]][mask_image == 0] = (0,
                                                                                            0,
                                                                                            0)

            video_writer.write(frame)
            continue
        if cur_time < raw_image_in:
            video_writer.write(frame)
            continue
        if cur_time < raw_image_out:
            frame[vertex_point[0]:vertex_point[0] + raw_image.shape[0],
                  vertex_point[1]:vertex_point[1] +
                  raw_image.shape[1]][mask_image == 0] = raw_image[mask_image == 0]
            video_writer.write(frame)
            continue

    video.release()
    video_writer.release()
    ff = FFmpeg(inputs={
        video_path: None,
        args.template: '-vn'
    },
                outputs={temp_video_path: '-y'})
    print(ff.cmd)
    ff.run()
    return temp_video_path


def main():
    images = os.listdir(args.images)
    os.makedirs(args.output_path, exist_ok=True)
    paths = []
    for image in images:
        image_path = os.path.join(args.images, image)
        # image must be a png and a square
        if not image_path.endswith('.png'):
            continue
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        # resize to 340x340
        image = cv2.resize(image, (340, 340))
        # cv2.imshow('image', image)
        cv2.waitKey(0)

        image_alpha = image[:, :, 3]

        mask_image = np.zeros((340, 340), np.uint8)
        mask_image[np.where(image_alpha == 0)] = 255

        raw_image = np.zeros((340, 340, 3), np.uint8)
        raw_image[np.where(image_alpha != 0)] = image[np.where(image_alpha != 0)][:, :3]
        # cv2.imshow('raw_image', raw_image)

        path = deal(raw_image, mask_image, os.path.basename(image_path))
        paths.append(path)

    with open(os.path.join('paths.txt'), 'w') as f:
        for path in paths:
            f.write(f"file '{path}'\n")

    inputs = {'paths.txt': '-f concat -safe 0'}
    outputs = {args.output: '-y -c copy'}
    ff = FFmpeg(inputs=inputs, outputs=outputs)
    print(ff.cmd)
    ff.run()


if __name__ == '__main__':
    main()