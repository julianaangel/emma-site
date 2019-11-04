# !/bin/bash
# installation helper for running SRModule.exe by taking snapshots of framebuffer

Xvfb :1 &
export DISPLAY=:1 & winetricks vb5run
import -display :1 -window root screenshot.png

