# Ultralytics YOLO 🚀, AGPL-3.0 license

import os
import json
import random
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ultralytics.utils import DATASETS_DIR, LOGGER, NUM_THREADS, TQDM
from ultralytics.utils.downloads import download
from ultralytics.utils.files import increment_path
from ultralytics.data.converter import coco91_to_coco80_class

from pycocotools.coco import COCO


def convert_coco(
    imgs_dir="../coco/images/",
    labels_dir="../coco/annotations/",
    save_dir="coco_converted/",
    use_segments=False,
    use_keypoints=False,
    cls91to80=True,
    lvis=False,
):

    # Create dataset directory
    save_dir = increment_path(save_dir)  # increment if save directory already exists
    for p in save_dir / "labels", save_dir / "images":
        p.mkdir(parents=True, exist_ok=True)  # make dir

    # Convert classes
    coco80 = coco91_to_coco80_class()

    # Import json
    for json_file in sorted(Path(labels_dir).resolve().glob("*.json")):
        lname = "" if lvis else json_file.stem.replace("instances_", "")
        fn = Path(save_dir) / "labels" / lname  # folder name
        # print(fn)
        fn.mkdir(parents=True, exist_ok=True)
        origin_img_dir = Path(imgs_dir) / lname
        # print("origin_img_dir: ",origin_img_dir)
        # 图像保存路径
        save_dir_images = Path(save_dir) / "images" / lname
        save_dir_images.mkdir(parents=True, exist_ok=True)

        if lvis:
            # NOTE: create folders for both train and val in advance,
            # since LVIS val set contains images from COCO 2017 train in addition to the COCO 2017 val split.
            (fn / "train2017").mkdir(parents=True, exist_ok=True)
            (fn / "val2017").mkdir(parents=True, exist_ok=True)
        with open(json_file) as f:
            data = json.load(f)

        # Create image dict
        images = {f'{x["id"]:d}': x for x in data["images"]}
        # Create image-annotations dict
        imgToAnns = defaultdict(list)
        for ann in data["annotations"]:
            imgToAnns[ann["image_id"]].append(ann)

        image_txt = []
        # Write labels file
        for img_id, anns in TQDM(imgToAnns.items(), desc=f"Annotations {json_file}"):
            img = images[f"{img_id:d}"]
            h, w = img["height"], img["width"]
            f = (
                str(Path(img["coco_url"]).relative_to("http://images.cocodataset.org"))
                if lvis
                else img["file_name"]
            )
            if lvis:
                image_txt.append(str(Path("./images") / f))

            bboxes = []
            segments = []
            keypoints = []
            for ann in anns:
                if ann.get("iscrowd", False):
                    continue
                # The COCO box format is [top left x, top left y, width, height]
                box = np.array(ann["bbox"], dtype=np.float64)
                box[:2] += box[2:] / 2  # xy top-left corner to center
                box[[0, 2]] /= w  # normalize x
                box[[1, 3]] /= h  # normalize y
                if box[2] <= 0 or box[3] <= 0:  # if w <= 0 and h <= 0
                    continue

                cls = (
                    coco80[ann["category_id"] - 1]
                    if cls91to80
                    else ann["category_id"] - 1
                )  # class
                box = [cls] + box.tolist()
                if box not in bboxes:
                    bboxes.append(box)
                    if use_segments and ann.get("segmentation") is not None:
                        if len(ann["segmentation"]) == 0:
                            segments.append([])
                            continue
                        elif len(ann["segmentation"]) > 1:
                            s = merge_multi_segment(ann["segmentation"])
                            s = (
                                (np.concatenate(s, axis=0) / np.array([w, h]))
                                .reshape(-1)
                                .tolist()
                            )
                        else:
                            s = [
                                j for i in ann["segmentation"] for j in i
                            ]  # all segments concatenated
                            s = (
                                (np.array(s).reshape(-1, 2) / np.array([w, h]))
                                .reshape(-1)
                                .tolist()
                            )
                        s = [cls] + s
                        segments.append(s)
                    if use_keypoints and ann.get("keypoints") is not None:
                        keypoints.append(
                            box
                            + (
                                np.array(ann["keypoints"]).reshape(-1, 3)
                                / np.array([w, h, 1])
                            )
                            .reshape(-1)
                            .tolist()
                        )

            # Write
            with open((fn / f).with_suffix(".txt"), "a") as file:
                for i in range(len(bboxes)):
                    if use_keypoints:
                        line = (*(keypoints[i]),)  # cls, box, keypoints
                    else:
                        line = (
                            *(
                                segments[i]
                                if use_segments and len(segments[i]) > 0
                                else bboxes[i]
                            ),
                        )  # cls, box or segments
                    file.write(("%g " * len(line)).rstrip() % line + "\n")

            # 如果bboxes不为空，则保存图片
            # print(f"{len(bboxes)} bboxes found in {f}")
            # print(Path(origin_img_dir) / f)
            # print(save_dir_images / f)
            if len(bboxes) > 0:
                shutil.copyfile(
                    Path(origin_img_dir) / f,
                    save_dir_images / f,
                )

        if lvis:
            with open(
                (
                    Path(save_dir)
                    / json_file.name.replace("lvis_v1_", "").replace(".json", ".txt")
                ),
                "a",
            ) as f:
                f.writelines(f"{line}\n" for line in image_txt)

    LOGGER.info(
        f"{'LVIS' if lvis else 'COCO'} data converted successfully.\nResults saved to {save_dir.resolve()}"
    )


def merge_multi_segment(segments):
    """
    Merge multiple segments into one list by connecting the coordinates with the minimum distance between each segment.
    This function connects these coordinates with a thin line to merge all segments into one.

    Args:
        segments (List[List]): Original segmentations in COCO's JSON file.
                               Each element is a list of coordinates, like [segmentation1, segmentation2,...].

    Returns:
        s (List[np.ndarray]): A list of connected segments represented as NumPy arrays.
    """
    s = []
    segments = [np.array(i).reshape(-1, 2) for i in segments]
    idx_list = [[] for _ in range(len(segments))]

    # Record the indexes with min distance between each segment
    for i in range(1, len(segments)):
        idx1, idx2 = min_index(segments[i - 1], segments[i])
        idx_list[i - 1].append(idx1)
        idx_list[i].append(idx2)

    # Use two round to connect all the segments
    for k in range(2):
        # Forward connection
        if k == 0:
            for i, idx in enumerate(idx_list):
                # Middle segments have two indexes, reverse the index of middle segments
                if len(idx) == 2 and idx[0] > idx[1]:
                    idx = idx[::-1]
                    segments[i] = segments[i][::-1, :]

                segments[i] = np.roll(segments[i], -idx[0], axis=0)
                segments[i] = np.concatenate([segments[i], segments[i][:1]])
                # Deal with the first segment and the last one
                if i in {0, len(idx_list) - 1}:
                    s.append(segments[i])
                else:
                    idx = [0, idx[1] - idx[0]]
                    s.append(segments[i][idx[0] : idx[1] + 1])

        else:
            for i in range(len(idx_list) - 1, -1, -1):
                if i not in {0, len(idx_list) - 1}:
                    idx = idx_list[i]
                    nidx = abs(idx[1] - idx[0])
                    s.append(segments[i][nidx:])
    return s


def min_index(arr1, arr2):
    """
    Find a pair of indexes with the shortest distance between two arrays of 2D points.

    Args:
        arr1 (np.ndarray): A NumPy array of shape (N, 2) representing N 2D points.
        arr2 (np.ndarray): A NumPy array of shape (M, 2) representing M 2D points.

    Returns:
        (tuple): A tuple containing the indexes of the points with the shortest distance in arr1 and arr2 respectively.
    """
    dis = ((arr1[:, None, :] - arr2[None, :, :]) ** 2).sum(-1)
    return np.unravel_index(np.argmin(dis, axis=None), dis.shape)


if __name__ == "__main__":
    # DUO 数据集转换
    # convert_coco(
    #     imgs_dir="../datasets/DUO/images/",
    #     labels_dir="../datasets/DUO/annotations/",
    #     save_dir="../datasets/DUO/",
    #     use_segments=False,
    #     use_keypoints=False,
    #     cls91to80=False,
    # )
    # TrashCAN material version 数据集转换
    convert_coco(
        imgs_dir="../datasets/TrashCAN/material_version/images/",
        labels_dir="../datasets/TrashCAN/material_version/annotations/",
        save_dir="../datasets/TrashCAN_material/",
        use_segments=False,
        use_keypoints=False,
        cls91to80=False,
    )

