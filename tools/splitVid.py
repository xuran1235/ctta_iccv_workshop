import os
from PIL import Image
from moviepy.editor import VideoFileClip

def split_videos_to_frames(folder_path):
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        outpath = "/data/SHIFT/continuous/videos/1x/val/front/vid/"
        # 检查文件是否为MP4视频文件
        if filename.endswith(".mp4"):
            # 创建用于保存帧的文件夹
            frame_folder = outpath + os.path.splitext(filename)[0]
            os.makedirs(frame_folder, exist_ok=True)
            
            # 使用moviepy库加载视频
            video = VideoFileClip(filepath)            
            # 按帧遍历视频并保存为图片文件
            for i, frame in enumerate(video.iter_frames()):
                frame_filename = os.path.join(frame_folder, f"{i:08}_img_front.jpg")
                frame_image = Image.fromarray(frame)
                frame_image.save(frame_filename)
                
            # 关闭视频
            video.close()

# 指定包含MP4视频文件的文件夹路径
folder_path = "/data/SHIFT/continuous/videos/1x/val/front/img/"
split_videos_to_frames(folder_path)

