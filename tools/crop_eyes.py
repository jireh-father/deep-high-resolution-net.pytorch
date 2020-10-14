import json
import os
import glob
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--result_file', type=str, default=None)
parser.add_argument('--output_dir', type=str, default=None)
parser.add_argument('--image_dir', type=str, default=None)

args = parser.parse_args()

results = json.load(open(args.result_file))

for item in results:
    image_id = item['image_id']
    #"_face.jpg"

