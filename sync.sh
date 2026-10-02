#!/bin/sh
# Push the working tree to the GPU box over the shared SSH control socket.
rsync -az --delete -e "ssh -S $HOME/.ssh/cm-lisplab2" \
  --exclude .git --exclude runs --exclude '__pycache__' --exclude '*.wav' --exclude legacy --exclude docs/audio \
  ./ f004h1v@lisplab-2.thayer.dartmouth.edu:/scratch2/f004h1v/sliders/repo/
