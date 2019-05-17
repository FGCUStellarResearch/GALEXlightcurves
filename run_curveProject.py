"""
Helper script for running Dory's `curveProject.py` code, which creates light curves
from GALEX data given an RA, DEC, and targetID

This script is about a half-step up from being hard-coded;
`.tsv` files are outputs from a ViZier search, with a substantial header
`.csv` files are just basic ascii lists with a single line header

If you're updating this to run with different target lists, or using it as a template
for your own helper `run` script, keep this in mind!

"""

import os

inlist_file = input("Enter target list file: ")

if '.tsv' in inlist_file:
    with open(inlist_file, 'r') as infile:
        # let's ignore the header data
        asu_data = infile.readlines()[51:]

    for target in asu_data:
        columns = target.split(" ")
        if not len(columns) < 2:
            ra = columns[0]
            dec = columns[1]
            targetID = str(columns[-1].split('\t')[-1].split('\n')[0])
            print(targetID)
            command = "python3 curveProject4L.py -ra {} -dec {} -target {} ".format(ra, dec, targetID)
            print(command)
            #break

            os.system(command)

elif '.csv' in inlist_file:
    with open(inlist_file, 'r') as infile:
        # let's ignore the header data
        asu_data = infile.readlines()[1:]

    for target in asu_data:
        columns = target.split(",")
        if len(columns) > 2:
            ra = columns[1]
            dec = columns[2]
            targetID = str(columns[0])
            print(targetID)
            command = "python3 curveProject4L.py -ra {} -dec {} -target {} ".format(ra, dec, targetID)
            print(command)
            #break

            os.system(command)
