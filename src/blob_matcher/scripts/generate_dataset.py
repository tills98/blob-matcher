# Copyright 2019 Hendrik Sauer
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from functools import reduce
import logging
import math
import os
import random
import sys
import typing


import kornia
import numpy as np
import pdf2image
import torch
import torchvision
from torchvision.transforms import v2


from blob_matcher.homography import sample_homography
from blob_matcher.keypoints import (
    augment_ellipse,
    conic_to_ellipse,
    ellipse_to_affine,
    get_patch,
    keypoint_to_mapped_conic,
    keypoints_to_torch,
    map_keypoint,
    physical_to_logical_distance,
)
from blob_matcher.utils import read_json
from blob_matcher.utils.functional import curry, fchain, flip


try:
    path_to_blobboards = os.path.join(os.getcwd(), "../BlobBoards.jl/python")
    sys.path.insert(0, path_to_blobboards)
    from BlobBoards.core import ureg, BlobGenerator
    dpi = 1 / ureg.inch
except ModuleNotFoundError:
    import pint
    ureg = pint.UnitRegistry()
    dpi = 1 / ureg.inch


# The datasets to be created. Containing a name, a transform for the images, parameters for the homography sampling, and keypoint augmentation parameters
DATASETS: list[tuple[str, v2.Transform, dict[str, typing.Any], dict[str, typing.Any]]] = [
    (
        "generated_single_2",
        v2.Compose(
            [v2.GaussianNoise(), v2.ColorJitter([0.1, 0.8]), v2.GaussianBlur(kernel_size=(5, 5))]
        ),
        {
            "base_scale": 0.1,
            "allow_artifacts": True,
            "scaling_amplitude": 0.4,
            "perspective": True,
        },
        {},
    ),
]

IMAGE_RESOLUTION = (1280, 1920)
BLOBBOARD_RESOLUTION = (6143, 6143)


def map_blobs(
    background: torch.Tensor,
    homography: torch.Tensor,
    blobs: torch.Tensor,
    blob_alpha=1.0,
) -> torch.Tensor:
    """
    Warps the blobsheet into the background according to a homography.

    Arguments:
        background: A 2-D array containing the image data for the background.
        homography: A valid homography with shape (3, 3) that describes the transformation from the blobboard into
        the image.
        blobs: The blobboard to be mapped.

    Returns:
        An image with the blobsheet mapped in front of a background.
    """
    warped_blobs = kornia.geometry.transform.warp_perspective(
        blobs,
        homography,
        (background.shape[2], background.shape[3]),
    )
    mask = kornia.geometry.transform.warp_perspective(
        torch.ones_like(blobs) * blob_alpha,
        homography,
        (background.shape[2], background.shape[3]),
    )
    # mask = (mask > 0.999).float()
    warped_blobs = warped_blobs * mask + background * (1 - mask)
    return warped_blobs


def generate_dataset(
    cfg,
    path,
    backgrounds,
    boards,
    image_transform,
    homography_kwargs,
    augmentation_args,
    is_validation=False,
    max_boards_per_image=1
):
    os.makedirs(os.path.join(path, "warped_images"), exist_ok=True)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.mps.is_available() else "cpu")
    )
    print(device)

    resize = torchvision.transforms.Resize(IMAGE_RESOLUTION).to(device)

    blobboard_shape = BLOBBOARD_RESOLUTION
    homographies = {
        background: {
            board: sample_homography(blobboard_shape, IMAGE_RESOLUTION, **homography_kwargs).to(device)
            for board in random.choices(boards, k=random.randint(1, max_boards_per_image))
        } for background in backgrounds
    }
    torch.save(homographies, os.path.join(path, "homographies.pt"))

    # create YOLO labels dir
    os.makedirs(os.path.join(path, "labels"), exist_ok=True)
    
    blobboard_json = read_json(boards[0][0])
    blobboard_shape = blobboard_json["preamble"]["board_config"]["canvas_size"]
    blobboard_shape = (
        physical_to_logical_distance(
            blobboard_shape["height"]["value"],
            blobboard_json["preamble"]["board_config"]["print_density"]["value"],
        ),
        physical_to_logical_distance(
            blobboard_shape["height"]["value"],
            blobboard_json["preamble"]["board_config"]["print_density"]["value"],
        ),
    )

    keypoints = []
    for i in range(len(backgrounds)):
        if os.path.exists(os.path.join(path, "warped_images", f"{i:04}.png")):
            continue

        img = (
            torchvision.io.decode_image(
                os.path.join(os.getcwd(), backgrounds[i]),
                torchvision.io.ImageReadMode.GRAY,
            ).to(torch.float32)
            / 255
        )
        assert img.size(0) == 1
        # if img.shape[1] > img.shape[2]:
        #     crop1 = (img.shape[1] - img.shape[2]) // 2
        #     crop2 = crop1 if 2 * crop1 + img.shape[2] == img.shape[1] else crop1 + 1
        #     img = resize(img[:, crop1:-crop2, :])
        # elif img.shape[2] > img.shape[1]:
        #     crop1 = (img.shape[2] - img.shape[1]) // 2
        #     crop2 = crop1 if 2 * crop1 + img.shape[1] == img.shape[2] else crop1 + 1
        #     img = resize(img[:, :, crop1:-crop2])
        # else:
        img = resize(img).to(device).unsqueeze(0)
        for board in homographies[backgrounds[i]]:
            blobboard_img = pdf2image.convert_from_path(board[1], dpi=1200, grayscale=True)[0]
            blobboard_t = torchvision.transforms.functional.pil_to_tensor(blobboard_img)

            blobboard = (
                blobboard_t.to(torch.float32)
                .to(device)
                / 255
            )
            img = map_blobs(
                img, homographies[backgrounds[i]][board].unsqueeze(0), blobboard.unsqueeze(0)
            )
        img = image_transform(img)
        torchvision.utils.save_image(
            img, os.path.join(path, "warped_images", f"{i:04}.png")
        )

        # write label file for image
        with open(os.path.join(path, "labels", f"{i:04}.txt"), "w", encoding="utf-8") as f:
            for board in homographies[backgrounds[i]]:
                blobboard_info = read_json(board[0])

                res_dpi = int(blobboard_info["preamble"]["board_config"]["print_density"]["value"])
                bb_x_center_mm = float(blobboard_info["preamble"]["board_config"]["board_size"]["width"]["value"]) / 2
                bb_y_center_mm = float(blobboard_info["preamble"]["board_config"]["board_size"]["height"]["value"]) / 2

                bb_width_mm = float(blobboard_info["preamble"]["board_config"]["board_size"]["width"]["value"])
                bb_height_mm = float(blobboard_info["preamble"]["board_config"]["board_size"]["height"]["value"])

                bb_tl_mm = (0, 0)
                bb_tr_mm = (bb_width_mm, 0)
                bb_bl_mm = (0, bb_height_mm)
                bb_br_mm = (bb_width_mm, bb_height_mm)

                bb_tl_px = physical_to_logical_distance(bb_tl_mm[0], res_dpi), physical_to_logical_distance(bb_tl_mm[1], res_dpi)
                bb_tr_px = physical_to_logical_distance(bb_tr_mm[0], res_dpi), physical_to_logical_distance(bb_tr_mm[1], res_dpi)
                bb_bl_px = physical_to_logical_distance(bb_bl_mm[0], res_dpi), physical_to_logical_distance(bb_bl_mm[1], res_dpi)
                bb_br_px = physical_to_logical_distance(bb_br_mm[0], res_dpi), physical_to_logical_distance(bb_br_mm[1], res_dpi)

                bb_c_px = physical_to_logical_distance(bb_x_center_mm, res_dpi), physical_to_logical_distance(bb_y_center_mm, res_dpi)

                homography = homographies[backgrounds[i]][board]

                img_bb_tl_px = map_keypoint(homography, torch.tensor(bb_tl_px, dtype=torch.float32).to(device))
                img_bb_tr_px = map_keypoint(homography, torch.tensor(bb_tr_px, dtype=torch.float32).to(device))
                img_bb_bl_px = map_keypoint(homography, torch.tensor(bb_bl_px, dtype=torch.float32).to(device))
                img_bb_br_px = map_keypoint(homography, torch.tensor(bb_br_px, dtype=torch.float32).to(device))
                img_bb_c_px = map_keypoint(homography, torch.tensor(bb_c_px, dtype=torch.float32).to(device))

                # Calculate axis-aligned bounding box from the four transformed corners
                all_x = torch.stack([img_bb_tl_px[0], img_bb_tr_px[0], img_bb_bl_px[0], img_bb_br_px[0]])
                all_y = torch.stack([img_bb_tl_px[1], img_bb_tr_px[1], img_bb_bl_px[1], img_bb_br_px[1]])

                min_x = torch.min(all_x)
                max_x = torch.max(all_x)
                min_y = torch.min(all_y)
                max_y = torch.max(all_y)

                # Calculate center and dimensions of the covering rectangle
                x_center = (min_x + max_x) / 2 / IMAGE_RESOLUTION[1]
                y_center = (min_y + max_y) / 2 / IMAGE_RESOLUTION[0]
                width = (max_x - min_x) / IMAGE_RESOLUTION[1]
                height = (max_y - min_y) / IMAGE_RESOLUTION[0]
                
                line = f"0 {x_center.item()} {y_center.item()} {abs(width.item())} {abs(height.item())}\n"

                f.write(line)

        continue

        blobboard_info = read_json(boards[i][0])
        keypoints = keypoints_to_torch(blobboard_info)
        sigma_cutoff = blobboard_info["preamble"]["pattern_config"]["sigma_cutoff"]

        keypoint_pairs = zip(
            map(lambda k: (k[:2], (k[2], k[2]), 0), keypoints),
            map(
                augment_ellipse(**augmentation_args),
                map(
                    conic_to_ellipse,
                    map(curry(keypoint_to_mapped_conic)(homographies[i]), keypoints),
                ),
            ),
        )
        min_keypoint_size = 4
        try:
            anchor_keypoints, positive_keypoints = list(
                zip(
                    *filter(
                        lambda x: x[1][0][0] >= 0
                        and x[1][0][0] < warped_image.shape[3]
                        and x[1][0][1] >= 0
                        and x[1][0][1] < warped_image.shape[2]
                        and (x[1][1][0] + x[1][1][1]) >= min_keypoint_size,
                        filter(
                            lambda x: x[0] is not None and x[1] is not None,
                            keypoint_pairs,
                        ),
                    )
                )
            )
        except ValueError:
            logging.warning(f"Skipping image {i:04}")
            continue
        anchor_keypoints = list(
            map(
                lambda k: (
                    k[0],
                    k[1],
                    (k[2] + torch.rand((1,)) * 2 * math.pi % math.pi),
                ),
                anchor_keypoints,
            )
        )
        garbage_locations = torch.rand((len(positive_keypoints), 2)) * torch.tensor(
            [[4000, 6000]]
        ).expand(len(positive_keypoints), -1)
        positive_keypoint_scales = torch.tensor(
            list(map(lambda k: k[1][0], positive_keypoints))
        )
        garbage_scales = torch.normal(
            torch.mean(positive_keypoint_scales)
            .unsqueeze(0)
            .expand(garbage_locations.size(0)),
            torch.where(
                torch.std(positive_keypoint_scales)
                .unsqueeze(0)
                .expand(garbage_locations.size(0))
                >= 0,
                torch.std(positive_keypoint_scales)
                .unsqueeze(0)
                .expand(garbage_locations.size(0)),
                torch.zeros((garbage_locations.size(0),)),
            ),
        )
        garbage_orientations = torch.rand((len(positive_keypoints),)) * 2 * np.pi
        garbage_keypoints = list(
            zip(
                garbage_locations,
                zip(garbage_scales, garbage_scales),
                garbage_orientations,
            )
        )

        anchor_transforms = torch.stack(
            list(map(ellipse_to_affine, anchor_keypoints))
        ).to(device)
        positive_transforms = torch.stack(
            list(map(ellipse_to_affine, positive_keypoints))
        ).to(device)
        garbage_transforms = torch.stack(
            list(map(ellipse_to_affine, garbage_keypoints))
        ).to(device)

        for psf in [64, 96, 128]:
            os.makedirs(
                os.path.join(path, "patches", f"{int(psf)}", "anchors"), exist_ok=True
            )
            os.makedirs(
                os.path.join(path, "patches", f"{int(psf)}", "positives"), exist_ok=True
            )
            os.makedirs(
                os.path.join(path, "patches", f"{int(psf)}", "garbage"), exist_ok=True
            )
            if is_validation:
                os.makedirs(
                    os.path.join(path, "patches", f"{int(psf)}", "false_anchors"),
                    exist_ok=True,
                )
                false_anchor_map = torch.empty((2, anchor_transforms.size(0)))
                false_anchor_map[0] = torch.randperm(anchor_transforms.size(0))
                false_anchor_map[1, 1:] = false_anchor_map[0, :-1]
                false_anchor_map[1, 0] = false_anchor_map[0, -1]
                false_anchor_indices = false_anchor_map[
                    :, false_anchor_map[0].argsort()
                ]
                false_anchor_tranforms = anchor_transforms[
                    false_anchor_indices[1, :].to(int)
                ]
                false_anchor_patches = get_patch(
                    blobboard,
                    false_anchor_tranforms,
                    cfg,
                    sigma_cutoff=sigma_cutoff,
                    psf=psf,
                )
                for j in range(false_anchor_patches.size(0)):
                    torchvision.utils.save_image(
                        false_anchor_patches[j],
                        os.path.join(
                            path,
                            "patches",
                            f"{int(psf)}",
                            "false_anchors",
                            f"{i:04}_{j:04}.png",
                        ),
                    )

            anchor_patches = get_patch(
                blobboard, anchor_transforms, cfg, sigma_cutoff=sigma_cutoff, psf=psf
            )
            positive_patches = get_patch(
                warped_image,
                positive_transforms,
                cfg,
                sigma_cutoff=sigma_cutoff,
                psf=psf,
            )
            garbage_patches = get_patch(img, garbage_transforms, cfg, psf=psf)
            for j in range(anchor_patches.size(0)):
                torchvision.utils.save_image(
                    anchor_patches[j],
                    os.path.join(
                        path, "patches", f"{int(psf)}", "anchors", f"{i:04}_{j:04}.png"
                    ),
                )
                torchvision.utils.save_image(
                    positive_patches[j],
                    os.path.join(
                        path,
                        "patches",
                        f"{int(psf)}",
                        "positives",
                        f"{i:04}_{j:04}.png",
                    ),
                )
                if j < garbage_patches.size(0):
                    torchvision.utils.save_image(
                        garbage_patches[j],
                        os.path.join(
                            path,
                            "patches",
                            f"{int(psf)}",
                            "garbage",
                            f"{i:04}_{j:04}.png",
                        ),
                    )


def generate_real_dataset(homographies, cfg, path, validation_boards):
    augmentation = v2.ColorJitter(brightness=(0.3, 1), contrast=0.8, saturation=0.5)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.mps.is_available() else "cpu")
    )
    print(device)

    for i, (image, board) in enumerate(homographies.keys()):
        subdir = "validation" if board in validation_boards else "training"
        img = (
            torchvision.io.decode_image(image, torchvision.io.ImageReadMode.GRAY)
            .to(device)
            .to(torch.float32)
            / 255
        )
        img = augmentation(img)
        blobboard_info = read_json(
            f"./data/real_image_data/2025_11_05/boards/blob_board_{board}.json"
        )
        keypoints = keypoints_to_torch(blobboard_info)
        sigma_cutoff = blobboard_info["preamble"]["pattern_config"]["sigma_cutoff"]

        blobboard = pdf2image.convert_from_path(
            f"./data/real_image_data/2025_11_05/boards/blob_board_{board}.pdf",
            dpi=blobboard_info["preamble"]["board_config"]["print_density"]["value"],
            grayscale=True,
        )[0]
        blobboard = (
            torchvision.transforms.functional.pil_to_tensor(blobboard)
            .to(device)
            .to(torch.float32)
            / 255
        )
        # Map the canvas into normalized coordinates
        translation = torch.tensor(
            [
                physical_to_logical_distance(
                    -(
                        blobboard_info["preamble"]["board_config"]["canvas_size"][
                            "width"
                        ]["value"]
                        - blobboard_info["preamble"]["board_config"]["board_size"][
                            "width"
                        ]["value"]
                    )
                    / 2
                    - blobboard_info["preamble"]["board_config"]["border_width"][
                        "value"
                    ],
                    blobboard_info["preamble"]["board_config"]["print_density"][
                        "value"
                    ],
                ),
                physical_to_logical_distance(
                    -(
                        blobboard_info["preamble"]["board_config"]["canvas_size"][
                            "height"
                        ]["value"]
                        - blobboard_info["preamble"]["board_config"]["board_size"][
                            "height"
                        ]["value"]
                    )
                    / 2
                    - blobboard_info["preamble"]["board_config"]["border_width"][
                        "value"
                    ],
                    blobboard_info["preamble"]["board_config"]["print_density"][
                        "value"
                    ],
                ),
            ],
            dtype=torch.float32,
        )
        translation_mat = torch.eye(3, dtype=torch.float32)
        translation_mat[:2, 2] = translation
        scale = torch.tensor(
            [
                1
                / physical_to_logical_distance(
                    blobboard_info["preamble"]["board_config"]["board_size"]["width"][
                        "value"
                    ]
                    - 2
                    * blobboard_info["preamble"]["board_config"]["border_width"][
                        "value"
                    ],
                    blobboard_info["preamble"]["board_config"]["print_density"][
                        "value"
                    ],
                ),
                1
                / physical_to_logical_distance(
                    blobboard_info["preamble"]["board_config"]["board_size"]["height"][
                        "value"
                    ]
                    - 2
                    * blobboard_info["preamble"]["board_config"]["border_width"][
                        "value"
                    ],
                    blobboard_info["preamble"]["board_config"]["print_density"][
                        "value"
                    ],
                ),
                1,
            ],
            dtype=torch.float32,
        )
        normalization = torch.diag(scale) @ translation_mat
        os.makedirs(os.path.join(path, subdir, "warped_images"), exist_ok=True)
        torchvision.utils.save_image(
            map_blobs(
                img.unsqueeze(0),
                (homographies[(image, board)].to(torch.float32) @ normalization)
                .to(device)
                .unsqueeze(0),
                blobboard.unsqueeze(0),
                blob_alpha=0.5,
            ),
            os.path.join(path, subdir, "warped_images", f"{i:04}.png"),
        )

        keypoint_pairs = zip(
            map(lambda k: (k[:2], (k[2], k[2]), 0), keypoints),
            map(
                conic_to_ellipse,
                map(
                    curry(keypoint_to_mapped_conic)(
                        homographies[(image, board)].to(torch.float32) @ normalization
                    ),
                    keypoints,
                ),
            ),
        )
        min_keypoint_size = 4
        anchor_keypoints, positive_keypoints = list(
            zip(
                *filter(
                    lambda x: x[1][0][0] >= 0
                    and x[1][0][0] < img.shape[2]
                    and x[1][0][1] >= 0
                    and x[1][0][1] < img.shape[1]
                    and (x[1][1][0] + x[1][1][1]) >= min_keypoint_size,
                    filter(
                        lambda x: x[0] is not None and x[1] is not None, keypoint_pairs
                    ),
                )
            )
        )
        anchor_keypoints = list(
            map(
                lambda k: (
                    k[0],
                    k[1],
                    (k[2] + torch.rand((1,)) * 2 * math.pi % math.pi),
                ),
                anchor_keypoints,
            )
        )
        garbage_locations = torch.rand((len(positive_keypoints), 2)) * torch.tensor(
            [[4000, 6000]]
        ).expand(len(positive_keypoints), -1)
        positive_keypoint_scales = torch.tensor(
            list(map(lambda k: k[1][0], positive_keypoints))
        )
        garbage_scales = torch.normal(
            torch.mean(positive_keypoint_scales)
            .unsqueeze(0)
            .expand(garbage_locations.size(0)),
            torch.std(positive_keypoint_scales)
            .unsqueeze(0)
            .expand(garbage_locations.size(0)),
        )
        garbage_orientations = torch.rand((len(positive_keypoints),)) * 2 * np.pi
        garbage_keypoints = list(
            zip(
                garbage_locations,
                zip(garbage_scales, garbage_scales),
                garbage_orientations,
            )
        )

        anchor_transforms = torch.stack(
            list(map(ellipse_to_affine, anchor_keypoints))
        ).to(device)
        positive_transforms = torch.stack(
            list(map(ellipse_to_affine, positive_keypoints))
        ).to(device)
        garbage_transforms = torch.stack(
            list(map(ellipse_to_affine, garbage_keypoints))
        ).to(device)

        for psf in [64, 96, 128]:
            os.makedirs(
                os.path.join(path, subdir, "patches", f"{int(psf)}", "anchors"),
                exist_ok=True,
            )
            os.makedirs(
                os.path.join(path, subdir, "patches", f"{int(psf)}", "positives"),
                exist_ok=True,
            )
            os.makedirs(
                os.path.join(path, subdir, "patches", f"{int(psf)}", "garbage"),
                exist_ok=True,
            )
            is_validation = True
            if is_validation:
                os.makedirs(
                    os.path.join(
                        path, subdir, "patches", f"{int(psf)}", "false_anchors"
                    ),
                    exist_ok=True,
                )
                false_anchor_map = torch.empty((2, anchor_transforms.size(0)))
                false_anchor_map[0] = torch.randperm(anchor_transforms.size(0))
                false_anchor_map[1, 1:] = false_anchor_map[0, :-1]
                false_anchor_map[1, 0] = false_anchor_map[0, -1]
                false_anchor_indices = false_anchor_map[
                    :, false_anchor_map[0].argsort()
                ]
                false_anchor_tranforms = anchor_transforms[
                    false_anchor_indices[1, :].to(int)
                ]
                false_anchor_patches = get_patch(
                    blobboard,
                    false_anchor_tranforms,
                    cfg,
                    sigma_cutoff=sigma_cutoff,
                    psf=psf,
                )
                for j in range(false_anchor_patches.size(0)):
                    torchvision.utils.save_image(
                        false_anchor_patches[j],
                        os.path.join(
                            path,
                            subdir,
                            "patches",
                            f"{int(psf)}",
                            "false_anchors",
                            f"{i:04}_{j:04}.png",
                        ),
                    )

            anchor_patches = get_patch(
                blobboard, anchor_transforms, cfg, sigma_cutoff=sigma_cutoff, psf=psf
            )
            positive_patches = get_patch(
                img, positive_transforms, cfg, sigma_cutoff=sigma_cutoff, psf=psf
            )
            garbage_patches = get_patch(img, garbage_transforms, cfg, psf=psf)
            for j in range(anchor_patches.size(0)):
                torchvision.utils.save_image(
                    anchor_patches[j],
                    os.path.join(
                        path,
                        subdir,
                        "patches",
                        f"{int(psf)}",
                        "anchors",
                        f"{i:04}_{j:04}.png",
                    ),
                )
                torchvision.utils.save_image(
                    positive_patches[j],
                    os.path.join(
                        path,
                        subdir,
                        "patches",
                        f"{int(psf)}",
                        "positives",
                        f"{i:04}_{j:04}.png",
                    ),
                )
                if j < garbage_patches.size(0):
                    torchvision.utils.save_image(
                        garbage_patches[j],
                        os.path.join(
                            path,
                            subdir,
                            "patches",
                            f"{int(psf)}",
                            "garbage",
                            f"{i:04}_{j:04}.png",
                        ),
                    )


def main():
    sys.path.insert(0, os.getcwd())
    from blob_matcher.configs.defaults import _C as cfg

    config_path = os.path.join(os.getcwd(), "configs", "init.yml")
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "--path", default=os.path.join(os.getcwd(), "data/datasets/new")
    )
    argument_parser.add_argument(
        "--backgrounds", default=os.path.join(os.getcwd(), "data/backgrounds")
    )
    argument_parser.add_argument(
        "--boards", default=os.path.join(os.getcwd(), "data/patterns")
    )
    argument_parser.add_argument(
        "--config_file", default=config_path, help="path to config file", type=str
    )
    argument_parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = argument_parser.parse_args()

    # if args.config_file != "":
    #     cfg.merge_from_file(args.config_file)

    cfg.merge_from_list(args.opts)

    if not cfg.TRAINING.NO_CUDA:
        torch.cuda.manual_seed_all(cfg.TRAINING.SEED)
        torch.backends.cudnn.deterministic = True

    # set random seeds
    # random.seed(cfg.TRAINING.SEED)
    # torch.manual_seed(cfg.TRAINING.SEED)
    # np.random.seed(cfg.TRAINING.SEED)


    validation_split = 0.1

    boards = []
    board_files = os.listdir(args.boards)
    board_files.sort()
    for i in range(0, len(board_files) - 1, 2):
        assert board_files[i][:-4] == board_files[i + 1][:-3]
        boards.append(
            (
                os.path.join(args.boards, board_files[i]),
                os.path.join(args.boards, board_files[i + 1]),
            )
        )
    random.shuffle(boards)

    background_filenames: list = fchain(
        list,
        curry(filter)(curry(flip(str.endswith))(".png")),
        curry(reduce)(list.__add__),
        curry(map)(lambda x: list(map(lambda f: os.path.join(x[0], f), x[2]))),
    )(os.walk(os.path.join(args.backgrounds)))
    random.shuffle(background_filenames)

    num_images_total = min(len(boards), len(background_filenames))
    boards = boards[:num_images_total]
    background_filenames = background_filenames[:num_images_total]

    num_validation_images = int(validation_split * num_images_total)
    num_training_images = num_images_total - num_validation_images
    validation_boards = boards[:num_validation_images]
    validation_backgrounds = background_filenames[:num_validation_images]
    training_boards = boards[num_validation_images:]
    training_backgrounds = background_filenames[num_validation_images:]

    for i, (
        dataset_name,
        image_transform,
        homography_kwargs,
        augmentation_args,
    ) in enumerate(DATASETS):
        _training_backgrounds = training_backgrounds[
            i * (num_training_images // len(DATASETS)):
            (i + 1) * (num_training_images // len(DATASETS))
        ]
        _training_boards = training_boards[
            i * (num_training_images // len(DATASETS)):
            (i + 1) * (num_training_images // len(DATASETS))
        ]
        _validation_backgrounds = validation_backgrounds[
            i * (num_validation_images // len(DATASETS)):
            (i + 1) * (num_validation_images // len(DATASETS))
        ]
        _validation_boards = validation_boards[
            i * (num_validation_images // len(DATASETS)):
            (i + 1) * (num_validation_images // len(DATASETS))
        ]
        os.makedirs(os.path.join(args.path, "./data/datasets/new", dataset_name, "training"), exist_ok=True)
        with open(os.path.join(args.path, "./data/datasets/new", dataset_name, "training/backgrounds.txt"), "x", encoding="utf-8") as f:
            f.writelines(map(lambda b: b + "\n", _training_backgrounds))
        with open(os.path.join(args.path, "./data/datasets/new", dataset_name, "training/boards.txt"), "x", encoding="utf-8") as f:
            f.writelines(map(lambda b: b[0] + "\n", _training_boards))
        os.makedirs(os.path.join(args.path, "./data/datasets/new", dataset_name, "validation"), exist_ok=True)
        with open(os.path.join(args.path, "./data/datasets/new", dataset_name, "validation/backgrounds.txt"), "x", encoding="utf-8") as f:
            f.writelines(map(lambda b: b + "\n", _validation_backgrounds))
        with open(os.path.join(args.path, "./data/datasets/new", dataset_name, "validation/boards.txt"), "x", encoding="utf-8") as f:
            f.writelines(map(lambda b: b[0] + "\n", _validation_boards))
        generate_dataset(
            cfg,
            os.path.join(args.path, dataset_name, "training"),
            _training_backgrounds,
            _training_boards,
            image_transform,
            homography_kwargs,
            augmentation_args,
            is_validation=False,
        )
        generate_dataset(
            cfg,
            os.path.join(args.path, dataset_name, "validation"),
            _validation_backgrounds,
            _validation_boards,
            image_transform,
            homography_kwargs,
            augmentation_args,
            is_validation=True,
        )
    return
    generate_real_dataset(torch.load("real_homographies.pt"), cfg, "./data/datasets/new/real", ["3f9"])


if __name__ == "__main__":
    main()
