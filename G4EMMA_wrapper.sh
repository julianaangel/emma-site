#!/bin/bash

#server vm (set the geant4 and root env vars)
source $THISROOT_PATH
source $THISGEANT4_PATH

# $1 is the MainDir
# $2 is the UserDir

# with autorun enabled
"$G4EMMA_SIM_PATH"/EMMAapp visOff $1 $2 <<< exit
