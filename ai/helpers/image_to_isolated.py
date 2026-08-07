"""Isolate the damaged building from its background.

Everything downstream in this project works better on a building that has been
cut out of its surroundings: facade parsing must not be confused by scenery,
depth estimation must not waste its range on the sky, and the point cloud
builder in ``images_to_3d.py`` explicitly discards the background before
meshing.

The core decision is which prior to trust. A salient object model (u2net,
ISNet, the usual "remove background" tools) is the obvious choice and the wrong
one: it looks for a single dominant subject, so on a wide aerial scene it locks
onto one block and ignores the rest, and on a facade that fills the frame it
returns almost nothing at all. What this project needs is a model that knows
what a building *is*, per pixel.

So the prior here is semantic segmentation: SegFormer trained on ADE20K, whose
150 classes include building, house and skyscraper alongside sky, earth, road
and tree. Rather than listing the classes that count as a building, the mask is
built by subtraction, keeping every pixel that is *not* recognisable scenery.
That way structure the model mislabels as some indoor class, which happens
often on a collapsed building where interiors are exposed, is still kept.

Two classical cues back the prior up and survive without it:

* Sky removal by flood fill from the top border. Deterministic, and because it
  grows from many seeds it follows a sunset gradient instead of stopping at a
  single colour tolerance.
* Structural energy, measured as local edge density. Facades are dense with
  window, floor and rubble edges, whereas sky, sand and asphalt are flat.

The fused evidence becomes a GrabCut trimap rather than a final answer, which
keeps a mistake in any single cue recoverable and snaps the boundary onto real
colour edges. The mask is computed on a downscaled copy for speed and
stability, then upsampled and feathered against the full resolution image.

If the segmentation model cannot be loaded, for instance with no network on the
first run, the classical cues alone still produce a usable mask.
"""

import cv2
import numpy as np

# Longest side used while analysing the image. The mask is computed at this
# scale and refined back at full resolution.
ANALYSIS_MAX_SIDE = 1024

# Semantic segmentation model providing the foreground prior.
# The b4 variant is a good accuracy/speed balance; b5 is sharper but heavier.
SEGMENTATION_MODEL_NAME = "nvidia/segformer-b4-finetuned-ade-512-512"

# Short side the image is resized to before segmentation. The checkpoint was
# trained at 512, but running a little higher recovers thin structure such as
# balconies and floor slabs, at a modest cost.
SEGMENTATION_SHORT_SIDE = 768

# ADE20K classes that are scenery rather than built structure. Everything else
# is treated as part of the building, so structure that the model mislabels as
# an indoor class still survives.
BACKGROUND_CLASS_IDS = frozenset(
    {
        2,  # sky
        4,  # tree
        6,  # road
        9,  # grass
        11,  # sidewalk
        12,  # person
        13,  # earth
        16,  # mountain
        17,  # plant
        20,  # car
        21,  # water
        26,  # sea
        29,  # field
        32,  # fence
        34,  # rock
        43,  # signboard
        46,  # sand
        52,  # path
        60,  # river
        66,  # flower
        68,  # hill
        72,  # palm
        76,  # boat
        80,  # bus
        83,  # truck
        87,  # streetlight
        91,  # dirt track
        93,  # pole
        94,  # land
        102,  # van
        103,  # ship
        113,  # waterfall
        114,  # tent
        116,  # minibike
        123,  # trade name
        126,  # animal
        127,  # bicycle
        128,  # lake
        136,  # traffic light
        140,  # pier
        149,  # flag
    }
)

# Weights used to fuse the semantic prior with the structural energy map.
# The prior carries the decision; structure only rescues borderline pixels.
SEMANTIC_PRIOR_WEIGHT = 0.85
STRUCTURE_WEIGHT = 0.15

# Trimap decision thresholds applied to the fused evidence score
DEFINITE_FOREGROUND_SCORE = 0.75
PROBABLE_FOREGROUND_SCORE = 0.45
DEFINITE_BACKGROUND_SCORE = 0.15

# Number of GrabCut refinement iterations
GRABCUT_ITERATIONS = 5

# A sky region larger than this fraction of the frame means the flood fill
# leaked into the building, so the sky cue is dropped for that image
MAX_SKY_AREA_RATIO = 0.75

# Connected components smaller than this fraction of the largest component are
# discarded as noise. A scene can legitimately hold several separate blocks, so
# raise this towards 1.0 to keep only the main structure.
MIN_COMPONENT_RATIO = 0.04

# Holes fully enclosed by the building are filled only when small. Larger gaps
# are genuine see-through damage, such as a collapsed room open to the sky, and
# must stay transparent.
MAX_FILLED_HOLE_RATIO = 0.01

# Segmentation model and processor are loaded once and reused
_segmentation_model = None
_segmentation_processor = None
_segmentation_device = None
_segmentation_failed = False


def _load_segmentation_model():
    """
    Load the SegFormer model once and reuse it for every image.

    Returns:
        Tuple of (model, processor, device), or (None, None, None) if the
        model is unavailable
    """
    global _segmentation_model, _segmentation_processor, _segmentation_device
    global _segmentation_failed

    if _segmentation_model is not None or _segmentation_failed:
        return _segmentation_model, _segmentation_processor, _segmentation_device

    try:
        import torch
        from transformers import (
            SegformerForSemanticSegmentation,
            SegformerImageProcessor,
        )

        _segmentation_device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading segmentation model on {_segmentation_device}...")

        # Resizing is handled here instead of by the processor, so the aspect
        # ratio is preserved rather than squashed into a square
        _segmentation_processor = SegformerImageProcessor.from_pretrained(
            SEGMENTATION_MODEL_NAME
        )
        _segmentation_model = SegformerForSemanticSegmentation.from_pretrained(
            SEGMENTATION_MODEL_NAME
        ).to(_segmentation_device)
        _segmentation_model.eval()
    except Exception as error:
        # Missing package, no network for the first download, or an
        # unsupported runtime. The classical cues cover this case.
        print(f"Warning: semantic prior unavailable ({error}). Using classical cues only.")
        _segmentation_failed = True
        _segmentation_model = None
        _segmentation_processor = None
        _segmentation_device = None

    return _segmentation_model, _segmentation_processor, _segmentation_device


def _semantic_foreground_prior(image):
    """
    Estimate per-pixel probability that a pixel belongs to built structure.

    The probability is taken as one minus the total probability mass assigned
    to scenery classes, which gives a soft score rather than a hard label and
    lets borderline pixels be resolved later by GrabCut.

    Args:
        image: Input image array in BGR format

    Returns:
        Float32 probability map in the range [0, 1], or None if unavailable
    """
    model, processor, device = _load_segmentation_model()
    if model is None:
        return None

    try:
        import torch

        height, width = image.shape[:2]
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize by the short side, preserving the aspect ratio
        scale = SEGMENTATION_SHORT_SIDE / min(height, width)
        resized = cv2.resize(
            rgb_image,
            (int(round(width * scale)), int(round(height * scale))),
            interpolation=cv2.INTER_LINEAR,
        )

        inputs = processor(images=resized, return_tensors="pt", do_resize=False)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits

        # SegFormer predicts at a quarter resolution, so upsample to the
        # analysis size before converting the logits to probabilities
        logits = torch.nn.functional.interpolate(
            logits, size=(height, width), mode="bilinear", align_corners=False
        )
        probabilities = torch.softmax(logits, dim=1)[0]

        background_ids = sorted(BACKGROUND_CLASS_IDS)
        background_probability = probabilities[background_ids].sum(dim=0)
        foreground_probability = (1.0 - background_probability).cpu().numpy()
    except Exception as error:
        print(f"Warning: semantic prior failed ({error}). Using classical cues only.")
        return None

    return np.clip(foreground_probability, 0.0, 1.0).astype(np.float32)


def _flatness_mask(image):
    """
    Mark image regions with little local gradient.

    Sky, sand and asphalt are flat, whereas facades and rubble are not. This is
    used both to seed and to constrain the sky flood fill.

    Args:
        image: Input image array in BGR format

    Returns:
        Boolean array that is True on flat pixels
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (0, 0), 2.0)

    # Scharr is more rotation-accurate than Sobel at this kernel size
    gradient_x = cv2.Scharr(blurred, cv2.CV_32F, 1, 0)
    gradient_y = cv2.Scharr(blurred, cv2.CV_32F, 0, 1)
    gradient_magnitude = cv2.magnitude(gradient_x, gradient_y)

    # Relative rather than absolute threshold, so exposure does not matter
    flat_cutoff = np.percentile(gradient_magnitude, 55)

    return gradient_magnitude < flat_cutoff


def _sky_mask(image):
    """
    Detect the sky by flood filling inward from the top border.

    Seeds are spread along the top edge rather than using a single point, so a
    sunset gradient is followed correctly instead of being cut off at one
    colour tolerance.

    Args:
        image: Input image array in BGR format

    Returns:
        Boolean array that is True on sky pixels
    """
    height, width = image.shape[:2]

    # Flood fill on a smoothed copy so texture noise does not stop the fill
    smoothed = cv2.GaussianBlur(image, (0, 0), 3.0)
    flat = _flatness_mask(image)

    # Shared mask, so regions grown from different seeds accumulate.
    # floodFill requires a mask two pixels larger than the image.
    flood_mask = np.zeros((height + 2, width + 2), dtype=np.uint8)

    seed_step = max(1, width // 40)
    flood_flags = cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE | (255 << 8) | 4

    for x in range(0, width, seed_step):
        # Skip seeds that are already filled or sit on textured pixels
        if flood_mask[1, x + 1] != 0 or not flat[0, x]:
            continue

        cv2.floodFill(
            smoothed,
            flood_mask,
            (x, 0),
            0,  # ignored, the source image is not modified in MASK_ONLY mode
            (14, 14, 14),  # loDiff
            (14, 14, 14),  # upDiff
            flood_flags,
        )

    sky = flood_mask[1:-1, 1:-1].astype(bool)

    # Constrain the fill to flat regions so it cannot leak down a uniformly lit
    # facade. The dilation keeps the soft sky/roof boundary intact.
    flat_dilated = cv2.dilate(
        flat.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    ).astype(bool)
    sky = sky & flat_dilated

    # A fill that swallowed most of the frame is not sky, so discard the cue
    if sky.mean() > MAX_SKY_AREA_RATIO:
        return np.zeros((height, width), dtype=bool)

    return sky


def _structure_energy(image):
    """
    Measure local edge density as evidence of built structure.

    Windows, floor slabs, balconies and rubble produce dense edges. Sky, sand
    and road surfaces do not, which separates the building from both the top
    and the bottom of a typical frame.

    Args:
        image: Input image array in BGR format

    Returns:
        Float32 energy map normalised to the range [0, 1]
    """
    height, width = image.shape[:2]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Bilateral filtering removes sensor noise while keeping structural edges
    denoised = cv2.bilateralFilter(gray, 7, 50, 50)

    # Hysteresis thresholds derived from image statistics, so the detector
    # behaves consistently across bright aerial and shaded street-level shots
    median_intensity = float(np.median(denoised))
    lower_threshold = int(max(0, 0.66 * median_intensity))
    upper_threshold = int(min(255, 1.33 * median_intensity))
    edges = cv2.Canny(denoised, lower_threshold, upper_threshold)

    # Average the edge map over a neighbourhood scaled to the image size
    window = max(9, int(min(height, width) * 0.035))
    window += 1 - (window % 2)  # box filter expects an odd window
    density = cv2.boxFilter(edges.astype(np.float32) / 255.0, -1, (window, window))

    # Normalise against a high percentile rather than the maximum, so a single
    # very busy patch does not flatten the rest of the map
    density_scale = np.percentile(density, 98)
    if density_scale <= 1e-6:
        return np.zeros((height, width), dtype=np.float32)

    return np.clip(density / density_scale, 0.0, 1.0).astype(np.float32)


def _fuse_evidence(semantic_prior, structure, sky):
    """
    Combine the individual cues into a single foreground score.

    Args:
        semantic_prior: Structure probability map, or None if unavailable
        structure: Structural energy map
        sky: Boolean sky mask

    Returns:
        Float32 score map in the range [0, 1]
    """
    if semantic_prior is None:
        # Without the semantic prior the structural energy carries the
        # decision, smoothed so isolated edges do not survive as speckle
        score = cv2.GaussianBlur(structure, (0, 0), 3.0)
    else:
        score = SEMANTIC_PRIOR_WEIGHT * semantic_prior + STRUCTURE_WEIGHT * structure

    # Sky is never part of the building, whatever the other cues say
    score[sky] = 0.0

    return np.clip(score, 0.0, 1.0).astype(np.float32)


def _build_trimap(score, sky):
    """
    Convert the fused score into GrabCut labels.

    Args:
        score: Fused foreground score map
        sky: Boolean sky mask

    Returns:
        Uint8 trimap using the cv2.GC_* label values
    """
    trimap = np.full(score.shape, cv2.GC_PR_BGD, dtype=np.uint8)
    trimap[score >= PROBABLE_FOREGROUND_SCORE] = cv2.GC_PR_FGD

    # Definite foreground is eroded so only confidently interior pixels are
    # locked, leaving the boundary for GrabCut to decide
    confident_foreground = (score >= DEFINITE_FOREGROUND_SCORE).astype(np.uint8)
    confident_foreground = cv2.erode(
        confident_foreground,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    )
    trimap[confident_foreground.astype(bool)] = cv2.GC_FGD

    trimap[score <= DEFINITE_BACKGROUND_SCORE] = cv2.GC_BGD
    trimap[sky] = cv2.GC_BGD

    return trimap


def _refine_with_grabcut(image, trimap):
    """
    Snap the mask boundary onto real colour edges.

    GrabCut fits Gaussian mixture colour models to the labelled regions and
    then finds a minimum cut, recovering the true silhouette where the fused
    evidence was only approximately right.

    Args:
        image: Input image array in BGR format
        trimap: Uint8 trimap of cv2.GC_* labels

    Returns:
        Uint8 binary mask (0 or 255)
    """
    has_foreground = np.any((trimap == cv2.GC_FGD) | (trimap == cv2.GC_PR_FGD))
    has_background = np.any((trimap == cv2.GC_BGD) | (trimap == cv2.GC_PR_BGD))

    # GrabCut needs examples of both classes to fit its colour models
    if not (has_foreground and has_background):
        foreground = (trimap == cv2.GC_FGD) | (trimap == cv2.GC_PR_FGD)
        return (foreground * 255).astype(np.uint8)

    grabcut_mask = trimap.copy()
    background_model = np.zeros((1, 65), dtype=np.float64)
    foreground_model = np.zeros((1, 65), dtype=np.float64)

    try:
        cv2.grabCut(
            image,
            grabcut_mask,
            None,
            background_model,
            foreground_model,
            GRABCUT_ITERATIONS,
            cv2.GC_INIT_WITH_MASK,
        )
    except cv2.error as error:
        # Degenerate colour models, fall back to the unrefined trimap
        print(f"Warning: GrabCut refinement skipped ({error})")
        grabcut_mask = trimap

    foreground = (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD)
    return (foreground * 255).astype(np.uint8)


def _fill_small_holes(mask):
    """
    Close small interior gaps while preserving see-through damage.

    A collapsed room that shows sky behind it must stay transparent, so only
    holes below a size threshold are filled, and holes touching the image
    border are left alone since they are background, not holes.

    Args:
        mask: Uint8 binary mask

    Returns:
        Uint8 binary mask with small interior holes filled
    """
    height, width = mask.shape
    inverted = cv2.bitwise_not(mask)

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(inverted, 8)
    filled = mask.copy()
    max_hole_area = mask.size * MAX_FILLED_HOLE_RATIO

    # Label 0 is the background of the inverted image, so start at 1
    for label in range(1, component_count):
        x, y, component_width, component_height, area = stats[label]

        touches_border = (
            x == 0
            or y == 0
            or x + component_width >= width
            or y + component_height >= height
        )

        if not touches_border and area <= max_hole_area:
            filled[labels == label] = 255

    return filled


def _clean_mask(mask):
    """
    Remove speckle and keep only the significant building components.

    Args:
        mask: Uint8 binary mask

    Returns:
        Cleaned uint8 binary mask
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # Opening drops isolated specks, closing seals thin cracks in the silhouette
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, 8)
    if component_count <= 1:
        return cleaned

    # Keep every component that is a meaningful fraction of the largest one,
    # since a scene can legitimately contain several separate blocks
    areas = stats[1:, cv2.CC_STAT_AREA]
    minimum_area = areas.max() * MIN_COMPONENT_RATIO

    kept = np.zeros_like(cleaned)
    for label, area in enumerate(areas, start=1):
        if area >= minimum_area:
            kept[labels == label] = 255

    return _fill_small_holes(kept)


def _feather_alpha(image, mask):
    """
    Turn the binary mask into a soft matte aligned with the image edges.

    A guided filter is used when the contrib module is present, because it
    pulls the matte onto the underlying image structure instead of merely
    blurring it. A plain blur is the fallback.

    Args:
        image: Full resolution image array in BGR format
        mask: Uint8 binary mask at full resolution

    Returns:
        Uint8 alpha matte
    """
    radius = max(4, int(min(image.shape[:2]) * 0.01))

    try:
        alpha = cv2.ximgproc.guidedFilter(
            guide=image,
            src=mask,
            radius=radius,
            eps=1e-3 * 255 * 255,
        )
    except AttributeError:
        # opencv-contrib is not installed, soften the edge instead
        alpha = cv2.GaussianBlur(mask, (0, 0), radius / 3.0)

    return np.clip(alpha, 0, 255).astype(np.uint8)


def _compose_outputs(image, alpha):
    """
    Build the deliverables produced for every isolated image.

    Args:
        image: Full resolution image array in BGR format
        alpha: Uint8 alpha matte at full resolution

    Returns:
        List of tuples (image, variant name)
    """
    alpha_3channel = cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0

    # Transparent cut-out, the archival result
    cutout = np.dstack([image, alpha])

    # White backdrop, consumed directly by images_to_3d.py, which discards
    # near-white pixels when building the point cloud
    white_background = (
        image.astype(np.float32) * alpha_3channel + 255.0 * (1.0 - alpha_3channel)
    )

    # Green tint over the kept region, for quick visual verification
    tint = np.zeros_like(image, dtype=np.float32)
    tint[:, :, 1] = 255.0
    overlay = image.astype(np.float32) * (1.0 - 0.45 * alpha_3channel) + tint * (
        0.45 * alpha_3channel
    )

    return [
        (cutout, "cutout"),
        (white_background.astype(np.uint8), "white_bg"),
        (alpha, "mask"),
        (overlay.astype(np.uint8), "overlay"),
    ]


def image_to_isolated(input_image):
    """
    Isolate the building from its background.

    Args:
        input_image: Input image array in BGR format

    Returns:
        List of tuples (processed_image, variant name):
        the transparent cut-out, a white background version, the alpha matte,
        and a verification overlay. Returns an empty list on invalid input.
    """
    if input_image is None:
        print("Error: Failed to load image")
        return []

    full_height, full_width = input_image.shape[:2]

    # Analyse a downscaled copy: GrabCut is expensive, and the cues are more
    # stable once fine texture has been averaged away
    scale = min(1.0, ANALYSIS_MAX_SIDE / max(full_height, full_width))
    if scale < 1.0:
        analysis_image = cv2.resize(
            input_image,
            (int(round(full_width * scale)), int(round(full_height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        analysis_image = input_image

    # Gather the independent cues
    semantic_prior = _semantic_foreground_prior(analysis_image)
    sky = _sky_mask(analysis_image)
    structure = _structure_energy(analysis_image)

    # Fuse, refine, and clean
    score = _fuse_evidence(semantic_prior, structure, sky)
    trimap = _build_trimap(score, sky)
    mask = _refine_with_grabcut(analysis_image, trimap)
    mask = _clean_mask(mask)

    if not np.any(mask):
        print("Warning: isolation produced an empty mask, keeping the original image")
        mask = np.full((analysis_image.shape[0], analysis_image.shape[1]), 255, np.uint8)

    # Back to full resolution, then align the matte to the real image edges
    if scale < 1.0:
        mask = cv2.resize(
            mask, (full_width, full_height), interpolation=cv2.INTER_LINEAR
        )

    alpha = _feather_alpha(input_image, mask)

    return _compose_outputs(input_image, alpha)
