"""
Photo tagging pipeline for the FPG Photo Library Portal (Phase 1 MVP)

For each image in an input folder, this script:
  1. Reads embedded EXIF metadata (capture date, copyright, GPS if present)
  2. Calls Google Cloud Vision for label detection, landmark detection,
     object localization, and web entity detection
  3. Infers season from capture date
  4. Attempts to match the image to a known Boston park/landmark
  5. Writes one JSON record per photo into a "review queue" file

Setup:
    pip install google-cloud-vision exifread
"""

import os
import sys
import json
import glob
from datetime import datetime

from google.cloud import vision
import exifread

# CAN EDIT: for more specific results to help identify parks in Boston
# Extend this as FPG confirms more landmark/statue names and the historic
# Lightroom tag list becomes available
# Matched case-insensitively against Vision's landmark + web-entity results.
KNOWN_PARKS = {
    "Boston Public Garden": [
        "public garden", "swan boat", "george washington statue",
        "ether monument", "make way for ducklings",
    ],
    "Boston Common": [
        "boston common", "frog pond", "soldiers and sailors monument",
        "central burying ground",
    ],
    "Commonwealth Avenue Mall": [
        "commonwealth avenue mall", "comm ave mall",
    ],
}

# Rough GPS bounding boxes (lat_min, lat_max, lon_min, lon_max) — a fallback
# for when a photo has GPS EXIF but Vision doesn't name-match the landmark.
# These are approximate; tighten them once real geotagged photos come in.
PARK_BOUNDING_BOXES = {
    "Boston Public Garden": (42.3524, 42.3555, -71.0724, -71.0688),
    "Boston Common": (42.3529, 42.3570, -71.0672, -71.0620),
    "Commonwealth Avenue Mall": (42.3491, 42.3525, -71.0850, -71.0770),
}


def get_season(date: datetime) -> str:
    """Meteorological season for New England, based on capture month."""
    month = date.month
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    return "Fall"


def dms_to_decimal(dms, ref):
    degrees, minutes, seconds = [float(x.num) / float(x.den) for x in dms.values]
    decimal = degrees + minutes / 60 + seconds / 3600
    if ref in ("S", "W"):
        decimal *= -1
    return decimal


def extract_exif(path: str) -> dict:
    """Pull capture date, copyright, and GPS coords (if present) from EXIF."""
    with open(path, "rb") as f:
        tags = exifread.process_file(f, details=False)

    result = {"capture_date": None, "copyright": None, "gps": None}

    date_tag = tags.get("EXIF DateTimeOriginal") or tags.get("Image DateTime")
    if date_tag:
        try:
            result["capture_date"] = datetime.strptime(str(date_tag), "%Y:%m:%d %H:%M:%S")
        except ValueError:
            pass

    copyright_tag = tags.get("Image Copyright")
    if copyright_tag:
        result["copyright"] = str(copyright_tag)

    lat = tags.get("GPS GPSLatitude")
    lat_ref = tags.get("GPS GPSLatitudeRef")
    lon = tags.get("GPS GPSLongitude")
    lon_ref = tags.get("GPS GPSLongitudeRef")
    if lat and lat_ref and lon and lon_ref:
        result["gps"] = (
            dms_to_decimal(lat, str(lat_ref)),
            dms_to_decimal(lon, str(lon_ref)),
        )

    # Fall back to file modified time if no EXIF capture date at all —
    # better than nothing for old scanned photos, but flag it as low
    # confidence so staff know to double check during review.
    if result["capture_date"] is None:
        mtime = os.path.getmtime(path)
        result["capture_date"] = datetime.fromtimestamp(mtime)
        result["capture_date_is_fallback"] = True
    else:
        result["capture_date_is_fallback"] = False

    return result


def match_park_by_gps(gps):
    if not gps:
        return None
    lat, lon = gps
    for park, (lat_min, lat_max, lon_min, lon_max) in PARK_BOUNDING_BOXES.items():
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            return park
    return None


def match_park_by_text(candidate_strings):
    lowered = " ".join(candidate_strings).lower()
    for park, keywords in KNOWN_PARKS.items():
        if any(keyword in lowered for keyword in keywords):
            return park
    return None


def analyze_image(client: vision.ImageAnnotatorClient, path: str) -> dict:
    with open(path, "rb") as f:
        content = f.read()
    image = vision.Image(content=content)

    # Multiple features in one request. Vision bills per feature type used,
    # so this costs roughly 4x a single label_detection call — worth
    # accounting for when tagging the full 30k-100k image backlog
    # (spec Section 8 estimate assumed single-feature label detection).
    response = client.annotate_image({
        "image": image,
        "features": [
            {"type_": vision.Feature.Type.LABEL_DETECTION, "max_results": 15},
            {"type_": vision.Feature.Type.LANDMARK_DETECTION, "max_results": 5},
            {"type_": vision.Feature.Type.OBJECT_LOCALIZATION, "max_results": 15},
            {"type_": vision.Feature.Type.WEB_DETECTION, "max_results": 10},
        ],
    })

    if response.error.message:
        raise RuntimeError(response.error.message)

    labels = [{"description": l.description, "score": round(l.score, 3)}
              for l in response.label_annotations]
    landmarks = [{"description": lm.description, "score": round(lm.score, 3)}
                 for lm in response.landmark_annotations]
    objects = [{"name": o.name, "score": round(o.score, 3)}
               for o in response.localized_object_annotations]
    web_entities = [{"description": w.description, "score": round(w.score, 3)}
                    for w in response.web_detection.web_entities if w.description]

    return {
        "labels": labels,
        "landmarks": landmarks,
        "objects": objects,
        "web_entities": web_entities,
    }


def build_record(path: str, client: vision.ImageAnnotatorClient) -> dict:
    exif = extract_exif(path)
    vision_data = analyze_image(client, path)

    # Try to identify a specific park: named landmark first, then web
    # entities (often catches named places Vision doesn't flag as a formal
    # "landmark"), then GPS bounding box as a last resort.
    text_candidates = (
        [lm["description"] for lm in vision_data["landmarks"]]
        + [w["description"] for w in vision_data["web_entities"]]
    )
    park = match_park_by_text(text_candidates) or match_park_by_gps(exif["gps"])

    capture_date = exif["capture_date"]
    season = get_season(capture_date) if capture_date else None

    return {
        "filename": os.path.basename(path),
        "status": "pending_review",  # staff confirm/edit before publish (Sec 6.3)
        "suggested_tags": {
            "general_labels": [l["description"] for l in vision_data["labels"]],
            "objects": [o["name"] for o in vision_data["objects"]],
            "landmarks": [lm["description"] for lm in vision_data["landmarks"]],
            "park": park,
            "season": season,
        },
        "metadata": {
            "capture_date": capture_date.isoformat() if capture_date else None,
            "capture_date_is_estimated": exif["capture_date_is_fallback"],
            "copyright": exif["copyright"],
            "gps": exif["gps"],
        },
        "raw_vision_response": vision_data,  # kept for staff review / debugging
    }


def main(input_dir: str, output_path: str):
    client = vision.ImageAnnotatorClient()
    image_paths = sorted(
        p for ext in ("*.jpg", "*.jpeg", "*.png")
        for p in glob.glob(os.path.join(input_dir, "**", ext), recursive=True)
    )

    print(f"Found {len(image_paths)} images in {input_dir}")
    if not image_paths:
        print("No images found — check INPUT_DIR path.")
        return

    records = []
    for i, path in enumerate(image_paths, 1):
        print(f"[{i}/{len(image_paths)}] {os.path.basename(path)}")
        try:
            records.append(build_record(path, client))
        except Exception as e:
            print(f"  ERROR: {e}")
            records.append({
                "filename": os.path.basename(path),
                "status": "error",
                "error": str(e),
            })

    with open(output_path, "w") as f:
        json.dump(records, f, indent=2)

    print(f"\nWrote {len(records)} records to {output_path}")


if __name__ == "__main__":
    input_dir = sys.argv[1] if len(sys.argv) > 1 else "./photos"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "review_queue.json"
    main(input_dir, output_path)