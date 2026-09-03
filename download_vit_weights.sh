#!/usr/bin/env bash
set -euo pipefail
mkdir -p weights
url="https://storage.googleapis.com/vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz"
output="weights/vit_small_patch16_224.npz"
curl -L --fail --retry 3 "$url" -o "$output"
echo "545815b4e770d2fa6ca4b3ccba7c16b035e474354e52d17ca197ea4efecbf4d3  $output" | sha256sum -c -

