"""
this file is used to convert video to image sequence,
image name is defined as: frame_id.jpg
"""
import os
import cv2
import json
import time

base_dir = "/data/ipad_3d/cjh/byteTrack/datasets/UOT100/"
# there are many seb folders in the base_dir, each sub folder contains a video
# we need to convert each video to images
sub_dirs = os.listdir(base_dir)
for sub_dir in sub_dirs:
    sub_dir_path = os.path.join(base_dir, sub_dir)
    # there are only a video in each sub folder,format is mp4,but has other files,so we need to filter them
    video_name = [name for name in os.listdir(sub_dir_path) if name.endswith(".mp4")][0]
    video_path = os.path.join(sub_dir_path, video_name)
    # in same cases, there are already have a folder named "images" in the sub folder,so we just skip it
    if os.path.exists(os.path.join(sub_dir_path, "images")) or os.path.exists(os.path.join(sub_dir_path, "img")):
        continue
    # create a folder named "images" in the sub folder
    os.mkdir(os.path.join(sub_dir_path, "images"))
    # read the video
    cap = cv2.VideoCapture(video_path)
    # get the total frame number of the video
    total_frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # get the fps of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    # get the width and height of the video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # get the total time of the video
    total_time = total_frame_num / fps
    # get the frame interval
    frame_interval = 1 / fps
    # get the frame id
    frame_id = 0
    # read the video frame by frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # save the frame as a image
        cv2.imwrite(os.path.join(sub_dir_path, "images", str(frame_id) + ".jpg"), frame)
        frame_id += 1
    # release the video
    cap.release()
    # create a json file to store the video information
    json_file = open(os.path.join(sub_dir_path, "images", "video_info.json"), "w")
    # write the video information to the json file
    json_file.write(json.dumps({"total_frame_num": total_frame_num, "fps": fps, "width": width, "height": height,
                                "total_time": total_time, "frame_interval": frame_interval}))
    # close the json file
    json_file.close()
    # print time to finish convert a video and its name, time format is day/month/year hour:minute:second
    print(time.strftime("%d/%m/%Y %H:%M:%S", time.localtime()) + " " + video_name + " finished")