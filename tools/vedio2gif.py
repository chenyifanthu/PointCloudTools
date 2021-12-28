import cv2
import numpy as np
from moviepy.editor import VideoFileClip

speed = 8
clip = VideoFileClip("/Users/chenyifan/Desktop/20211208_154121.mp4")
clip = clip.resize(0.4)
clip = clip.speedx(12)

clip.write_gif("doc/temporal_synchronization_demo.gif",fps=12) 