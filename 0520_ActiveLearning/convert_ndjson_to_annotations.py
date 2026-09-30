#!/usr/bin/env python3
"""
Convert Labelbox ndjson export (one object per line) to per‑image annotation JSON files.
Expected format per line: {"id_name": "apracom_bc_84.png", "height": 1080, "width": 900,
                           "boxes": [[cx,cy,w,h], ...], "labels": ["l_num", "bacillus_bact", ...]}
Filters only "bacillus_bact" boxes, assigns class from filename prefix.
Output: for each image, a JSON file with list of {"bbox": [cx,cy,w,h], "class": "BC"} etc.
"""

import os
import json
import re
from pathlib import Path

def extract_plate_class(filename: str) -> str:
    """Map filename prefix to class name (BC, ETB, YM, XSA)."""
    name = filename.lower()
    if 'bc' in name:
        return 'BC'
    elif 'etb' in name:
        return 'ETB'
    elif 'ym' in name:
        return 'YM'
    elif 'xsa' in name:
        return 'XSA'
    else:
        raise ValueError(f"Cannot determine class from filename: {filename}")

def convert_ndjson(ndjson_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
    with open(ndjson_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            
            # Extract image metadata
            img_filename = data['id_name']
            img_height = data['height']
            img_width = data['width']
            boxes = data['boxes']
            labels = data['labels']
            
            # Filter only colony boxes (label == "bacillus_bact")
            colony_boxes = []
            for bbox, lbl in zip(boxes, labels):
                if (lbl == 'mold_bact') | (lbl == 'yeast_bact'):
                    # bbox is already normalized [cx, cy, w, h] in [0,1]
                    colony_boxes.append({
                        'bbox': bbox,  # keep as list of 4 floats
                        'class': extract_plate_class(img_filename)
                    })
            
            if not colony_boxes:
                # Still create a JSON file with an empty list
                colony_boxes = []   # explicit empty list
            
            # Create output JSON file
            base_name = os.path.splitext(img_filename)[0]
            output_file = os.path.join(output_dir, f"{base_name}.json")
            with open(output_file, 'w') as out_f:
                json.dump(colony_boxes, out_f, indent=2)
            
            print(f"Converted {img_filename} -> {output_file} ({len(colony_boxes)} colonies)")

if __name__ == "__main__":
    # Change these paths
    input_ndjson = "targets_apracom_ym_v1.1.ndjson"   # or the full path
    output_annot_dir = "./dataset/annotations"
    
    convert_ndjson(input_ndjson, output_annot_dir)
    print("\nConversion complete. Place your images in ./dataset/images/ and run the active learning script.")