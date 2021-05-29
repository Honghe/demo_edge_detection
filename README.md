# Demo Edge Detection of Images
Libraries used:
- Kornia
- OpenCV

## For video
Let convert the video to images via ffmpeg. 
```
rm -rf ./input/*.*
ffmpeg -i ./data/cyberpunk2077.flv -vf fps=10 ./input/%04d.png
rm -rf ./output/*.*
ffmpeg -r 10 -i ./output/%04d.png -c:v h264_nvenc -vf fps=10 ./out.mp4 -y
ffmpeg -i data/cyberpunk2077.flv -vn -c:a copy  -async 1 -y ./data/cyberpunk2077.m4a
ffmpeg -i out.mp4 -i data/cyberpunk2077.m4a -c:v copy -c:a copy outall.mp4
```
