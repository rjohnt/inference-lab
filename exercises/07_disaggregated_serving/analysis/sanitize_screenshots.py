"""Crop private terminal chrome and redact PCI addresses without resampling.

Usage: python sanitize_screenshots.py PRIVATE_SOURCE_DIR
Original captures remain outside Git. Coordinates refer to original pixels.
"""
import argparse
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('source', type=Path)
p.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'results/screenshots')
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
paired_redactions = [(1650, 0, 3305, 75), (2420, 379, 2728, 418),
                     (2420, 529, 2728, 566), (15, 1515, 210, 1645),
                     (2110, 884, 2235, 976)]
captures = [
    ('nvidia-smi', (1594, 926), (10, 145, 1568, 904),
     [(780, 170, 1065, 212), (780, 318, 1065, 360)]),
    ('nvtop', (3574, 2268), (125, 210, 3430, 2025), []),
    ('replicas-load', (3574, 2268), (125, 210, 3430, 2025),
     paired_redactions),
    ('pd-load', (3574, 2268), (125, 210, 3430, 2025), paired_redactions),
]
for name, size, box, redactions in captures:
    original = Image.open(a.source / f'{name}-original.png').convert('RGB')
    assert original.size == size
    cropped = original.crop(box)
    before = np.asarray(cropped).copy()
    allowed = np.zeros(before.shape[:2], dtype=bool)
    for x0, y0, x1, y1 in redactions:
        ImageDraw.Draw(cropped).rectangle((x0, y0, x1 - 1, y1 - 1), fill=(10, 16, 14))
        allowed[y0:y1, x0:x1] = True
    destination = a.output / (f'{name}.png' if name.endswith('-load') else f'{name}-setup.png')
    # Reconstruct a fresh image to discard original metadata; PNG is lossless.
    Image.fromarray(np.asarray(cropped)).save(destination, format='PNG')
    saved = np.asarray(Image.open(destination))
    assert np.array_equal(before[~allowed], saved[~allowed])
    print(f'{destination.name}: {cropped.size}; retained pixels verified unchanged')
