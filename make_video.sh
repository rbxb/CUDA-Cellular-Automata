#!/bin/bash

name=./resources/videos/out.mp4
frames=./resources/out%04d.pam

#ffmpeg -loglevel 24 -framerate 30 -i $frames $name
ffmpeg -loglevel 24 -framerate 60 -i resources/out%04d.PAM -c:v libx264 -crf 18 -movflags +faststart -vf 'colorchannelmixer=rr=0:rb=1:br=1:bb=0' -pix_fmt yuv420p $name

$SHELL