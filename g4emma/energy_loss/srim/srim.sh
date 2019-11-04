#!/bin/bash
#
#  Small bash script to run wine SRModule.exe, as python trips over stderr for some reason

SRIM_OUT="SRIM_OUT"
xvfb-run wine SRmodule.exe 2> $SRIM_OUT
cat $SRIM_OUT
