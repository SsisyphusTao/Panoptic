#!/usr/bin/env python
'''
This script converts data in panoptic COCO format to semantic segmentation. All
segments with the same semantic class in one image are combined together.

Additional option:
- using option '--things_others' the script combine all segments of thing
classes into one segment with semantic class 'other'.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os, sys
import argparse
import numpy as np
import math
import json
import time
import multiprocessing
from collections import defaultdict

import PIL.Image as Image

from panopticapi.utils import get_traceback, rgb2id, save_json

try:
    # set up path for pycocotools
    # sys.path.append('./cocoapi-master/PythonAPI/')
    from pycocotools import mask as COCOmask
except Exception:
    raise Exception("Please install pycocotools module from https://github.com/cocodataset/cocoapi")

OTHER_CLASS_ID = 183

_valid_ids = [
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 
      14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 
      24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 
      37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 
      48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 
      58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 
      72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 
      82, 84, 85, 86, 87, 88, 89, 90]
cat_ids = {v: i+1 for i, v in enumerate(_valid_ids)}
cat_ids.update({0: 0})

@get_traceback
def extract_semantic_single_core(proc_id,
                                 annotations_set,
                                 segmentations_folder,
                                 output_json_file,
                                 semantic_seg_folder,
                                 categories,
                                 save_as_png,
                                 things_other):
    annotation_semantic_seg = []
    for working_idx, annotation in enumerate(annotations_set):
        if working_idx % 100 == 0:
            print('Core: {}, {} from {} images processed'.format(proc_id,
                                                                 working_idx,
                                                                 len(annotations_set)))
        try:
            pan_format = np.array(
                Image.open(os.path.join(segmentations_folder, annotation['file_name'])),
                dtype=np.uint32
            )
        except IOError:
            raise KeyError('no prediction png file for id: {}'.format(annotation['image_id']))

        pan = rgb2id(pan_format)
        semantic = np.zeros(pan.shape, dtype=np.uint8)
        semantic_x = np.zeros((pan.shape[0],pan.shape[1],3), dtype=np.float)
        semantic_y = np.zeros((pan.shape[0],pan.shape[1],3), dtype=np.float)
        gradient_x = np.expand_dims(np.array([x for x in range(pan.shape[1])], dtype=np.float), 0).repeat(pan.shape[0], 0)
        gradient_y = np.expand_dims(np.array([x for x in range(pan.shape[0])], dtype=np.float), 1).repeat(pan.shape[1], 1)
        get_gradient = np.vectorize(lambda x, y: int(('%03d'%x)[y]))

        RLE_per_category = defaultdict(list)
        for segm_info in annotation['segments_info']:
            cat_id = segm_info['category_id']
            if cat_id < 91:
                if things_other and categories[cat_id]['isthing'] == 1:
                    cat_id = OTHER_CLASS_ID
                mask = pan == segm_info['id']
                if save_as_png:
                    semantic[mask] = cat_ids[cat_id]
                    x,y,w,h = segm_info['bbox']
                    mid_x = int(x+w/2)
                    mid_y = int(y+h/2)
                    assert mid_x < 1000 and mid_y < 1000
                    semantic_x[...,0][mask] = get_gradient(mid_x, 0)
                    semantic_y[...,0][mask] = get_gradient(mid_y, 0)
                    semantic_x[...,1][mask] = get_gradient(mid_x, 1)
                    semantic_y[...,1][mask] = get_gradient(mid_y, 1)
                    semantic_x[...,2][mask] = get_gradient(mid_x, 2)
                    semantic_y[...,2][mask] = get_gradient(mid_y, 2)
                else:
                    RLE = COCOmask.encode(np.asfortranarray(mask.astype('uint8')))
                    RLE['counts'] = RLE['counts'].decode('utf8')
                    RLE_per_category[cat_id].append(RLE)

        if save_as_png:
            # Image.fromarray(semantic).save(os.path.join(semantic_seg_folder, 'seg', annotation['file_name']))
            Image.fromarray(semantic_x.astype(np.uint8)).save(os.path.join(semantic_seg_folder, 'gx', annotation['file_name']))
            Image.fromarray(semantic_y.astype(np.uint8)).save(os.path.join(semantic_seg_folder, 'gy', annotation['file_name']))
        else:
            for cat_id, RLE_list in RLE_per_category.items():
                if len(RLE_list) == 1:
                    RLE = RLE_list[0]
                else:
                    RLE = COCOmask.merge(RLE_list)
                semantic_seg_record = {}
                semantic_seg_record["image_id"] = annotation['image_id']
                semantic_seg_record["category_id"] = cat_id
                semantic_seg_record["segmentation"] = RLE
                semantic_seg_record["area"] = int(COCOmask.area(RLE))
                semantic_seg_record["bbox"] = list(COCOmask.toBbox(RLE))
                semantic_seg_record["iscrowd"] = 0
                annotation_semantic_seg.append(semantic_seg_record)
    print('Core: {}, all {} images processed'.format(proc_id, len(annotations_set)))

    return annotation_semantic_seg


def extract_semantic(input_json_file,
                     segmentations_folder,
                     output_json_file,
                     semantic_seg_folder,
                     categories_json_file,
                     things_other):
    start_time = time.time()
    with open(input_json_file, 'r') as f:
        d_coco = json.load(f)
    annotations = d_coco['annotations']

    if segmentations_folder is None:
        segmentations_folder = input_json_file.rsplit('.', 1)[0]

    print("EXTRACTING FROM...")
    print("COCO panoptic format:")
    print("\tSegmentation folder: {}".format(segmentations_folder))
    print("\tJSON file: {}".format(input_json_file))
    print("SEMANTIC SEGMENTATION")

    if output_json_file is not None and semantic_seg_folder is not None:
        raise Exception("'--output_json_file' and '--semantic_seg_folder' \
                        options cannot be used together")

    save_as_png = False
    if output_json_file is None:
        if semantic_seg_folder is None:
            raise Exception("One of '--output_json_file' and '--semantic_seg_folder' \
                            options must be used specified")
        else:
            save_as_png = True
            print("in PNG format:")
            print("\tFolder with semnatic segmentations: {}".format(semantic_seg_folder))
            if not os.path.isdir(semantic_seg_folder):
                print("Creating folder {} for semantic segmentation PNGs".format(semantic_seg_folder))
                os.mkdir(semantic_seg_folder)
    else:
        print("in COCO detection format:")
        print("\tJSON file: {}".format(output_json_file))
    if things_other:
        print("Merging all things categories into 'other' category")
    print('\n')

    with open(categories_json_file, 'r') as f:
        categories_list = json.load(f)
    categories = {category['id']: category for category in categories_list}

    cpu_num = multiprocessing.cpu_count()
    annotations_split = np.array_split(annotations, cpu_num)
    print("Number of cores: {}, images per core: {}".format(cpu_num, len(annotations_split[0])))
    workers = multiprocessing.Pool(processes=cpu_num)
    processes = []
    for proc_id, annotations_set in enumerate(annotations_split):
        p = workers.apply_async(extract_semantic_single_core,
                                (proc_id, annotations_set, segmentations_folder,
                                 output_json_file, semantic_seg_folder,
                                 categories, save_as_png, things_other))
        processes.append(p)
    annotations_coco_semantic_seg = []
    for p in processes:
        annotations_coco_semantic_seg.extend(p.get())

    if not save_as_png:
        for idx, ann in enumerate(annotations_coco_semantic_seg):
            ann['id'] = idx
        d_coco['annotations'] = annotations_coco_semantic_seg
        categories_coco_semantic_seg = []
        for category in categories_list:
            if things_other and category['isthing'] == 1:
                continue
            category.pop('isthing')
            category.pop('color')
            categories_coco_semantic_seg.append(category)
        if things_other:
            categories_coco_semantic_seg.append({'id': OTHER_CLASS_ID,
                                                 'name': 'other',
                                                 'supercategory': 'other'})
        d_coco['categories'] = categories_coco_semantic_seg
        save_json(d_coco, output_json_file)

    t_delta = time.time() - start_time
    print("Time elapsed: {:0.2f} seconds".format(t_delta))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script converts data in panoptic COCO format to \
        semantic segmentation. All segments with the same semantic class in one \
        image are combined together. See this file's head for more information."
    )
    parser.add_argument('--input_json_file', type=str,
                        help="JSON file with panoptic data")
    parser.add_argument(
        '--segmentations_folder', type=str, default=None, help="Folder with \
         panoptic COCO format segmentations. Default: X if input_json_file is \
         X.json"
    )
    parser.add_argument('--output_json_file', type=str, default=None,
                        help="JSON file with semantic data. If '--output_json_file' \
                        is specified, resulting semantic segmentation will be \
                        stored as a JSON file in COCO stuff format (see \
                        http://cocodataset.org/#format-data for details).")
    parser.add_argument('--semantic_seg_folder', type=str, default=None,
                        help="Folder for semantic segmentation. If '--semantic_seg_folder' \
                        is specified, resulting semantic segmentation will be \
                        stored in the specified folder in PNG format.")
    parser.add_argument('--categories_json_file', type=str,
                        help="JSON file with Panoptic COCO categories information",
                        default='./panoptic_coco_categories.json')
    parser.add_argument('--things_other', action='store_true',
                        help="Is set, all things classes are merged into one \
                        'other' class")
    args = parser.parse_args()
    extract_semantic(args.input_json_file,
                     args.segmentations_folder,
                     args.output_json_file,
                     args.semantic_seg_folder,
                     args.categories_json_file,
                     args.things_other)
