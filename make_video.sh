#!/bin/bash

name=resources/videos/out.mp4
frames=resources/out%04d.pam

#ffmpeg -loglevel 24 -framerate 30 -i $frames $name
ffmpeg -loglevel 24 -framerate 30 -i $frames -c:v libx264 -crf 18 -movflags +faststart -vf 'drawtext=fontfile=/Windows/Fonts/8514oem.fon: text=%{n}: x=0: y=0: fontcolor=white: box=1: boxcolor=0x000000FF' -pix_fmt yuv420p $name