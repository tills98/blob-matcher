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


import math
import typing


import numpy as np
import pint
import torch


ureg = pint.UnitRegistry()
dpi = 1 / ureg.inch


def physical_to_logical_distance(x, resolution):
    return ((x * ureg.millimeter).to(ureg.inch) * (resolution * dpi)).magnitude


def physical_to_logical_coordinates(x, resolution, border_width, canvas_offset):
    return (
        (physical_to_logical_distance(border_width + canvas_offset[0], resolution)
            + (x[0] * ureg.millimeter).to(ureg.inch) * (resolution * dpi)).magnitude,
        (physical_to_logical_distance(border_width + canvas_offset[1], resolution)
            + (x[1] * ureg.millimeter).to(ureg.inch) * (resolution * dpi)).magnitude
    )


def map_keypoint(homography: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """
    Transforms a keypoint into a conic section and warps it via a homography.

    When ignoring orientation a keypoint can be repersented by a circle. If the circle is represented as a conic
    section, it can be transformed with a homography. We expect to obtain an elliptical conic section as a result,
    since the perspective change of the used homographies are not that large.

    Arguments:
        homography: The homography with which the keypoint will be mapped.
        k: The keypoint.

    Returns:
        An tensor of shape (3,3) representing the conic section of the mapped keypoint.
    """
    assert k.shape == (2,)

    x = k[0]
    y = k[1]
    
    # x' = Hx
    mapped_keypoint = homography @ torch.tensor([x, y, 1.0], dtype=torch.float32).to(homography.device)

    # normalize keypoint to get homogeneous coordinates
    mapped_keypoint = mapped_keypoint / mapped_keypoint[2]
    
    return mapped_keypoint[:2]


def keypoint_to_mapped_conic(homography: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """
    Transforms a keypoint into a conic section and warps it via a homography.

    When ignoring orientation a keypoint can be repersented by a circle. If the circle is represented as a conic
    section, it can be transformed with a homography. We expect to obtain an elliptical conic section as a result,
    since the perspective change of the used homographies are not that large.

    Arguments:
        homography: The homography with which the keypoint will be mapped.
        k: The keypoint.

    Returns:
        An tensor of shape (3,3) representing the conic section of the mapped keypoint.
    """
    assert k.shape == (4,)
    scale = k[2]

    x = k[0]
    y = k[1]
    conic = torch.tensor([[1, 0, -x], [0, 1, -y], [-x, -y, x**2 + y**2 - scale**2]], dtype=torch.float32).to(homography.device)
    inverse_homography = torch.linalg.inv(homography)
    mapped_conic = torch.transpose(inverse_homography, 0, 1) @ conic @ inverse_homography
    mapped_conic = (mapped_conic + torch.transpose(mapped_conic, 0, 1)) / 2
    return mapped_conic


def conic_to_ellipse(conic) -> typing.Optional[tuple[torch.Tensor, tuple[float, float], float]]:
    """
    Extracts ellipse parameters from a conic section.

    Arguments:
        conic: A (3, 3) array with the conic section.

    Returns:
        A tuple consisting of
            - a 2 entry tensor with the location,
            - a tuple containing the semi-major axis and the semi-minor axis, and
            - the angle of the semi-major axis in radians.
        Or `None` if the conic does not represent a valid homography.
    """
    location = -torch.linalg.inv(conic[:2, :2] * 2) @ (conic[:2, 2] * 2)
    A = conic[0, 0].item()
    B = conic[0, 1].item() * 2
    C = conic[1, 1].item()
    D = conic[0, 2].item() * 2
    E = conic[1, 2].item() * 2
    F = conic[2, 2].item()
    x_0 = location[0].item()
    y_0 = location[1].item()

    F_c = A * x_0 * x_0 + B * x_0 * y_0 + C * y_0 * y_0 + D * x_0 + E * y_0 + F
    if F_c >= 0:
        return None

    semi_axis_factor1 = 2 * (A * E ** 2 + C * D ** 2 - B * D * E + (B ** 2 - 4 * A * C) * F)
    semi_minor_axis_factor2 = (A + C) - np.sqrt((A - C) ** 2 + B ** 2)
    semi_major_axis_factor2 = (A + C) + np.sqrt((A - C) ** 2 + B ** 2)
    semi_axis_quotient = B ** 2 - 4 * A * C
    semi_minor_axis = -np.sqrt(semi_axis_factor1 * semi_minor_axis_factor2) / semi_axis_quotient
    semi_major_axis = -np.sqrt(semi_axis_factor1 * semi_major_axis_factor2) / semi_axis_quotient
    angle = np.atan2(-B, C - A) / 2

    return location, (semi_major_axis, semi_minor_axis), angle


def augment_ellipse(location_aug=0.1, scale_aug=0.1):
    def _augment_ellipse(ellipse):
        if ellipse is None:
            return None
        location, (semi_major, semi_minor), orientation = ellipse
        return (
            location + torch.distributions.normal.Normal(loc=0, scale=location_aug*semi_minor).sample((2,)).to(location.device),
            (
                semi_major + torch.distributions.normal.Normal(loc=0, scale=scale_aug*semi_major).sample((1,)),
                semi_minor + torch.distributions.normal.Normal(loc=0, scale=scale_aug*semi_minor).sample((1,))
            ),
            orientation + torch.distributions.uniform.Uniform(0, 2 * math.pi).sample((1,))
        )
    return _augment_ellipse


def ellipse_to_affine(ellipse) -> torch.Tensor:
    """
    Finds an affine transformation that transforms the unit circle into the ellipse.

    Arguments:
        ellipse: The ellipse in which the unit circle should be transformed.

    Returns:
        An (3, 3) tensor representing the affine transformation.
    """
    location, (semi_major_axis, semi_minor_axis), angle = ellipse
    scale = torch.diag(torch.tensor([semi_major_axis, semi_minor_axis, 1], dtype=torch.float32))
    rotation = torch.eye(3)
    angle = torch.tensor(angle)
    rotation[:2, :2] = torch.tensor(
        [[torch.cos(angle), torch.sin(angle)], [-torch.sin(angle), torch.cos(angle)]],
        dtype=torch.float32
    )
    translation = torch.eye(3)
    translation[:2, 2] = location
    return translation @ rotation @ scale


def get_patch(img: torch.Tensor, A: torch.Tensor, cfg, sigma_cutoff=1.0, psf=None, pad_with=1.0, resolution=128, batch_size=400):
    """
    Extract log-polar interpolated patches from an image given an affine transform of the base patch.

    The method used to extract the patch is now described in more detail. Given patch coordinates :math:`(x,y)` with
    :math:`0 \\leq x, y < P` for a given patch size :math:`P` we first normalize them into the half-open interval
    :math:`[0,1)`, which leaves us with :math:`\\overline{x} = \\frac{x}{P}, \\overline{x} = \\frac{x}{P}`. We set
    :math:`x` to be the radial dimension and :math:`y` to be the angular dimension of the patch. We can obtain the
    log-polar coordinates by:

    .. math::
        r = e^{\\ln(\\lambda/\\sigma_{cutoff}) \\cdot \\overline{x}} \\sigma_{cutoff}

    and

    .. math::
        \\theta = 2 \\cdot \\pi \\cdot \\overline{x}

    where :math:`\\lambda` is the size of the support-region, a configuration value that controls the size of the patch
    in relation to the radius :math:`\\sigma` of the blob. Note that :math:`r \\in [\\sigma_{cutoff}, \\lambda
    \\sigma_{cutoff})` and :math:`\\theta \\in [0, 2\\pi)`.

    Arguments:
        img: The source image. This can be a single (C, H, W) image or a batch (B, C, H, W) of images.
        A: A tensor of affine transforms (B, 3, 3) that maps the unit circle into the keypoints.
        pad_with: A color value to use for padding if part of the patch lie outside the source image.

    Returns:
        A (B, P, P) tensor containing the image data of the patches.
    """
    if batch_size < 1:
        raise ValueError(f"get_patch cannot work with {batch_size=}")
    try:
        device = img.device
        patch_size = resolution
        PSF = psf if psf is not None else cfg.BLOBINATOR.PATCH_SCALE_FACTOR
        oversampling_factor = 4
        P = oversampling_factor * patch_size
        *_, Hs, Ws = img.shape
        if img.ndim == 3:
            img = img.unsqueeze(0)
        A_chunks = A.split(batch_size)
        outputs = []
        for A_chunk in A_chunks:
            # Create normalized grid for output patch
            y, x = torch.meshgrid(
                torch.linspace(0, 1, P, device=device, dtype=torch.float32),
                torch.linspace(0, 1, P, device=device, dtype=torch.float32),
                indexing='ij'
            )
            # Each (x,y) is a location in patch coords, shape (P, P)
            # We treat x as radial and y as angular dimension

            # Compute log-polar coordinates
            r = torch.exp(torch.log(torch.tensor(PSF, device=device) / sigma_cutoff) * x) * sigma_cutoff    # [sigma_cutoff, PSF)
            theta = 2 * math.pi * y                                                                         # [0, 2π)

            # Convert to Cartesian coordinates in source patch
            xs = r * torch.cos(theta)
            ys = r * torch.sin(theta)

            # Stack homogeneous coords
            ones = torch.ones((1,)).expand(xs.size()).to(xs.device)
            coords = torch.stack([xs, ys, ones], dim=-1).to(A_chunk.device)
            coords = coords.view(1, P, P, 3)
            coords = coords.expand(A_chunk.size(0), -1, -1, -1)             # (B, P, P, 3)

            # Flatten spatial grid for batch multiplication
            coords_flat = coords.view(A_chunk.size(0), -1, 3)               # (B, P*P, 3)
            # Apply affine transform A to each batch item
            coords_flat = torch.bmm(coords_flat, A_chunk.transpose(1, 2))   # (B, P*P, 3)
            # Reshape back
            coords = coords_flat.view(A_chunk.size(0), P, P, 3)

            x_src, y_src = coords[..., 0], coords[..., 1]

            # Normalize coordinates to [-1, 1] for grid_sample
            x_norm = 2 * (x_src / (Ws - 1)) - 1
            y_norm = 2 * (y_src / (Hs - 1)) - 1
            grid = torch.stack([x_norm, y_norm], dim=-1)  # (B, P, P, 2)

            # Sample the image
            warped = torch.nn.functional.grid_sample(
                img.expand(grid.size(0), -1, -1, -1),
                grid,
                mode='bilinear',
                padding_mode='zeros',  # outside = 0
                align_corners=True
            )

            # Blend with constant background
            mask = torch.nn.functional.grid_sample(
                torch.ones((1, 1, 1, 1), device=device).expand(grid.size(0), 1, Hs, Ws),
                grid,
                mode='nearest',
                padding_mode='zeros',
                align_corners=True
            )

            background = torch.full_like(warped, pad_with)
            output = warped * mask + background * (1 - mask)
            output = torch.nn.functional.avg_pool2d(output, (oversampling_factor, oversampling_factor))
            outputs.append(output)
        output = torch.cat(outputs)
        assert output.shape[2] == patch_size and output.shape[3] == patch_size
        return output
    except torch.OutOfMemoryError:
        return get_patch(img, A, cfg, sigma_cutoff, psf, pad_with, resolution, batch_size // 2)


def keypoint_to_torch(resolution, border_width, canvas_offset):
    def __keypoint_to_torch(keypoint):
        return torch.tensor(
            [
                *physical_to_logical_coordinates(
                    (keypoint["center"][0]["value"], keypoint["center"][1]["value"]),
                    resolution,
                    border_width,
                    canvas_offset
                ),
                physical_to_logical_distance(keypoint["σ"]["value"], resolution),
                0
            ],
            dtype=torch.float32
        )
    return __keypoint_to_torch


def keypoints_to_torch(blob_info):
    resolution = blob_info["preamble"]["board_config"]["print_density"]["value"]
    border_width = blob_info["preamble"]["board_config"]["border_width"]["value"]
    canvas_offset = (
        (blob_info["preamble"]["board_config"]["canvas_size"]["width"]["value"]
            - blob_info["preamble"]["board_config"]["board_size"]["width"]["value"]) / 2,
        (blob_info["preamble"]["board_config"]["canvas_size"]["height"]["value"]
            - blob_info["preamble"]["board_config"]["board_size"]["height"]["value"]) / 2,
    )
    return torch.stack(list(map(
        keypoint_to_torch(resolution, border_width, canvas_offset),
        blob_info["blobs"])
    ))
