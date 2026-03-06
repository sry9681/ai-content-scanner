"""
SynthID Detection Smoke Test

Validates the FFT phase-analysis approach used by synthid-detect.js
against known watermarked and non-watermarked images.

Tests 4 detection tiers from simplest (JS-portable) to heaviest,
measuring accuracy, false positive/negative rates, and timing.

Usage:
    python -m venv .venv && source .venv/bin/activate
    pip install numpy scipy opencv-python-headless PyWavelets
    python tests/synthid/smoke_test.py

Requirements:
    numpy, scipy, opencv-python-headless, PyWavelets
"""

import time
import os
import sys
import numpy as np
import cv2
from scipy.fft import fft2, fftshift

# ── Carrier codebook (from reverse-SynthID analysis of 250 Gemini images) ──
# These are the known SynthID carrier frequencies with >99.9% phase coherence.
# Phase values are in radians, measured at 512x512 image resolution.

CARRIERS = [
    {"fy": -14,  "fx": -14,  "phase":  1.4409},
    {"fy":  14,  "fx":  14,  "phase": -1.4409},
    {"fy": -126, "fx": -14,  "phase":  2.3736},
    {"fy":  126, "fx":  14,  "phase": -2.3736},
    {"fy": -98,  "fx":  14,  "phase": -0.6109},
    {"fy":  98,  "fx": -14,  "phase":  0.6109},
    {"fy": -210, "fx":  14,  "phase": -1.1315},
    {"fy":  210, "fx": -14,  "phase":  1.1315},
    {"fy": -238, "fx": -14,  "phase":  1.6099},
    {"fy":  238, "fx":  14,  "phase": -1.6099},
]

SIZE = 512


# ════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════

def prep_image(path):
    """Load image, return list of 512x512 grayscale crops to test.
    For square images: one center crop.
    For non-square (aspect > 1.3): also direct 512x512 crops from center and edge."""
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Cannot load: {path}")
    h, w = img.shape[:2]
    crops = []

    # Crop 1: center-crop to square, resize to 512x512
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    cropped = img[y0:y0+side, x0:x0+side]
    resized = cv2.resize(cropped, (SIZE, SIZE))
    crops.append(cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32))

    # For non-square images, try direct 512x512 crops
    aspect = max(h, w) / min(h, w)
    if aspect > 1.3 and min(h, w) >= SIZE:
        cy = (h - SIZE) // 2
        cx = (w - SIZE) // 2
        crops.append(cv2.cvtColor(img[cy:cy+SIZE, cx:cx+SIZE], cv2.COLOR_BGR2GRAY).astype(np.float32))
        if h > w:
            crops.append(cv2.cvtColor(img[h-SIZE:h, cx:cx+SIZE], cv2.COLOR_BGR2GRAY).astype(np.float32))
        else:
            crops.append(cv2.cvtColor(img[cy:cy+SIZE, w-SIZE:w], cv2.COLOR_BGR2GRAY).astype(np.float32))

    return crops


def phase_diff(actual, expected):
    """Circular phase difference, normalized to [0, 1] match score."""
    d = actual - expected
    wrapped = abs(np.arctan2(np.sin(d), np.cos(d)))
    return 1.0 - wrapped / np.pi


# ════════════════════════════════════════════════════════════════
# TIER A: Raw FFT phase + annular magnitude (JS-portable, ~2ms)
#
# This is the algorithm used in synthid-detect.js.
# No denoising. Just FFT the raw grayscale image and check
# carrier frequency bins for phase match and magnitude anomaly.
# ════════════════════════════════════════════════════════════════

def tier_a_raw_fft(gray):
    f = fftshift(fft2(gray))
    phase = np.angle(f)
    mag = np.abs(f)
    center = SIZE // 2

    phase_scores = []
    mag_ratios = []
    for c in CARRIERS:
        y = c["fy"] + center
        x = c["fx"] + center

        # Phase match
        phase_scores.append(phase_diff(phase[y, x], c["phase"]))

        # Magnitude: compare carrier bin to median of same-radius ring
        radius = np.sqrt(c["fy"]**2 + c["fx"]**2)
        ring = []
        for a in range(24):
            angle = (a / 24) * 2 * np.pi
            ry = int(radius * np.sin(angle)) + center
            rx = int(radius * np.cos(angle)) + center
            if 0 <= ry < SIZE and 0 <= rx < SIZE and (abs(ry - y) > 2 or abs(rx - x) > 2):
                ring.append(mag[ry, rx])
        ring_med = np.median(ring) if ring else 1
        mag_ratios.append(mag[y, x] / ring_med if ring_med > 0 else 0)

    avg_phase = float(np.mean(phase_scores))
    phase_matched = sum(1 for s in phase_scores if s > 0.7)
    phase_matched_90 = sum(1 for s in phase_scores if s > 0.9)
    avg_mag = float(np.mean(mag_ratios))

    # Three detection paths:
    # Strong: overwhelming magnitude + phase (simple/synthetic images)
    strong = phase_matched >= 8 and avg_mag > 5.0
    # Moderate: good phase + clear magnitude anomaly
    moderate = phase_matched >= 6 and avg_phase > 0.55 and avg_mag > 3.0
    # Phase-strong: very high phase precision without magnitude anomaly
    # (catches content-rich images where natural frequencies mask the watermark)
    phase_strong = phase_matched_90 >= 6 and avg_phase > 0.75

    detected = strong or moderate or phase_strong

    return {
        "phase_match": avg_phase,
        "phase_matched": phase_matched,
        "phase_matched_90": phase_matched_90,
        "mag_ratio": avg_mag,
        "detected": detected,
        "tier": "strong" if strong else ("moderate" if moderate else ("phase_str" if phase_strong else "none")),
    }


# ════════════════════════════════════════════════════════════════
# TIER B: Gaussian highpass + FFT (JS-portable, ~3ms)
#
# Subtract a Gaussian blur to isolate high-frequency noise
# where the watermark lives, then FFT the residual.
# ════════════════════════════════════════════════════════════════

def tier_b_highpass_fft(gray):
    norm = gray / 255.0 if gray.max() > 1 else gray
    blurred = cv2.GaussianBlur(norm, (15, 15), 4.0)
    noise = norm - blurred

    f = fftshift(fft2(noise))
    phase = np.angle(f)
    center = SIZE // 2

    phase_scores = []
    for c in CARRIERS:
        y = c["fy"] + center
        x = c["fx"] + center
        phase_scores.append(phase_diff(phase[y, x], c["phase"]))

    avg_phase = float(np.mean(phase_scores))
    phase_matched = sum(1 for s in phase_scores if s > 0.7)
    structure = float(np.std(noise) / (np.mean(np.abs(noise)) + 1e-10))

    detected = avg_phase > 0.50 and phase_matched >= 4 and 0.8 < structure < 2.0

    return {
        "phase_match": avg_phase,
        "phase_matched": phase_matched,
        "structure_ratio": structure,
        "detected": detected,
    }


# ════════════════════════════════════════════════════════════════
# TIER C: Wavelet denoise + FFT (~15ms Python, not JS-portable)
#
# More sophisticated noise extraction via wavelet soft-thresholding.
# Included for comparison — not used in the extension.
# ════════════════════════════════════════════════════════════════

def tier_c_wavelet_fft(gray):
    import pywt

    norm = gray / 255.0 if gray.max() > 1 else gray
    coeffs = pywt.wavedec2(norm, 'db4', level=3)
    detail = coeffs[-1][0]
    sigma = np.median(np.abs(detail)) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(norm.size))
    new_coeffs = [coeffs[0]]
    for details in coeffs[1:]:
        new_coeffs.append(tuple(pywt.threshold(d, threshold, mode='soft') for d in details))
    denoised = pywt.waverec2(new_coeffs, 'db4')
    denoised = denoised[:norm.shape[0], :norm.shape[1]]
    noise = norm - denoised

    f = fftshift(fft2(noise))
    phase = np.angle(f)
    center = SIZE // 2

    phase_scores = []
    for c in CARRIERS:
        y = c["fy"] + center
        x = c["fx"] + center
        phase_scores.append(phase_diff(phase[y, x], c["phase"]))

    avg_phase = float(np.mean(phase_scores))
    phase_matched = sum(1 for s in phase_scores if s > 0.7)
    structure = float(np.std(noise) / (np.mean(np.abs(noise)) + 1e-10))

    detected = avg_phase > 0.50 and phase_matched >= 4 and 0.8 < structure < 2.0

    return {
        "phase_match": avg_phase,
        "phase_matched": phase_matched,
        "structure_ratio": structure,
        "detected": detected,
    }


# ════════════════════════════════════════════════════════════════
# TIER D: Combined A + B vote
#
# Uses both raw FFT and highpass FFT — if either has strong
# confidence, or both agree, declare detected.
# ════════════════════════════════════════════════════════════════

def tier_d_combined(gray):
    a = tier_a_raw_fft(gray)
    b = tier_b_highpass_fft(gray)

    a_strong = a["phase_match"] > 0.7 and a["phase_matched"] >= 6
    b_strong = b["phase_match"] > 0.7 and b["phase_matched"] >= 6
    both_agree = a["detected"] and b["detected"]
    detected = a_strong or b_strong or both_agree

    return {
        "phase_match_a": a["phase_match"],
        "phase_match_b": b["phase_match"],
        "matched_a": a["phase_matched"],
        "matched_b": b["phase_matched"],
        "detected": detected,
    }


# ════════════════════════════════════════════════════════════════
# CONTROL IMAGE GENERATION
# ════════════════════════════════════════════════════════════════

def generate_controls(out_dir):
    """Generate synthetic non-watermarked images for false positive testing."""
    os.makedirs(out_dir, exist_ok=True)
    controls = []

    generators = {
        "random_noise": lambda: np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8),
        "gradient": lambda: np.repeat(
            np.arange(512, dtype=np.uint8).reshape(512, 1, 1), 3, axis=2
        ).repeat(512, axis=1),
        "solid": lambda: np.full((512, 512, 3), 180, dtype=np.uint8),
        "checker": lambda: (
            np.array([
                [[200]*3 if (i//32 + j//32) % 2 == 0 else [50]*3 for j in range(512)]
                for i in range(512)
            ], dtype=np.float32) + np.random.randn(512, 512, 3) * 10
        ).clip(0, 255).astype(np.uint8),
        "blurred_random": lambda: cv2.GaussianBlur(
            np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8), (31, 31), 10
        ),
        "landscape": lambda: (
            cv2.GaussianBlur(
                np.array([
                    [[200-i//2, 180-i//3, 255-i//4] if i < 256 else [40+(i-256)//8, 80+(i-256)//4, 20]
                     for j in range(512)]
                    for i in range(512)
                ], dtype=np.uint8),
                (5, 5), 2
            ).astype(float) + np.random.randn(512, 512, 3) * 3
        ).clip(0, 255).astype(np.uint8),
        "texture": lambda: (lambda xx, yy: (
            np.stack([np.sin(xx*0.3)*50 + np.cos(yy*0.5)*50 + 128]*3, axis=2)
            .clip(0, 255).astype(np.float32) + np.random.randn(512, 512, 3) * 8
        ).clip(0, 255).astype(np.uint8))(*np.meshgrid(np.arange(512), np.arange(512))),
        "face_shapes": lambda: (lambda img: (
            cv2.circle(img, (256,256), 150, (180,160,140), -1),
            cv2.circle(img, (210,220), 20, (50,50,50), -1),
            cv2.circle(img, (300,220), 20, (50,50,50), -1),
            cv2.ellipse(img, (256,310), (40,15), 0, 0, 180, (150,50,50), 3),
            (img.astype(float) + np.random.randn(512,512,3)*5).clip(0,255).astype(np.uint8)
        )[-1])(np.full((512, 512, 3), 220, dtype=np.uint8)),
    }

    for name, gen_fn in generators.items():
        path = os.path.join(out_dir, f"{name}.png")
        if not os.path.exists(path):
            cv2.imwrite(path, gen_fn())
        controls.append((name, path))

    return controls


# ════════════════════════════════════════════════════════════════
# TEST RUNNER
# ════════════════════════════════════════════════════════════════

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    wm_dir = os.path.join(script_dir, "watermarked")
    ctrl_dir = os.path.join(script_dir, "controls")
    gen_ctrl_dir = os.path.join(ctrl_dir, "generated")

    # ── Build test set ──
    watermarked = []
    for fn in sorted(os.listdir(wm_dir)):
        if fn.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            watermarked.append(("WM", fn, os.path.join(wm_dir, fn)))

    cleaned = []
    for fn in sorted(os.listdir(ctrl_dir)):
        if fn.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            cleaned.append(("CLN", fn, os.path.join(ctrl_dir, fn)))

    controls = [("CTL", n, p) for n, p in generate_controls(gen_ctrl_dir)]

    all_images = watermarked + cleaned + controls

    if not all_images:
        print("ERROR: No test images found. Ensure watermarked/ and controls/ dirs have images.")
        sys.exit(1)

    # ── Define tiers ──
    tiers = [
        ("A: Raw FFT phase+mag", tier_a_raw_fft, "Used in synthid-detect.js"),
        ("B: Highpass + FFT", tier_b_highpass_fft, "JS-portable alternative"),
        ("C: Wavelet + FFT", tier_c_wavelet_fft, "Python-only (needs PyWavelets)"),
        ("D: Combined A+B", tier_d_combined, "Ensemble vote"),
    ]

    print("=" * 95)
    print("SynthID DETECTION SMOKE TEST")
    print("=" * 95)
    print(f"Test images: {len(watermarked)} watermarked, {len(cleaned)} cleaned/control, "
          f"{len(controls)} generated controls")
    print(f"Carrier frequencies: {len(CARRIERS)} (coherence >= 0.999)")
    print(f"Analysis size: {SIZE}x{SIZE} (center-crop to square)")

    results = {}
    for tier_name, tier_fn, tier_desc in tiers:
        print(f"\n{'─' * 95}")
        print(f"  {tier_name}  —  {tier_desc}")
        print(f"{'─' * 95}")
        hdr = f"  {'Type':<5} {'Image':<28} {'Det':<5} {'Phase':<7} {'Details':<35} {'Time':>7}"
        print(hdr)
        print(f"  {'─'*5} {'─'*28} {'─'*5} {'─'*7} {'─'*35} {'─'*7}")

        tier_results = []
        is_tier_a = "Raw FFT" in tier_name
        for label, name, path in all_images:
            try:
                crops = prep_image(path)
                t0 = time.perf_counter()
                if is_tier_a:
                    # Multi-crop: test all crops, take best
                    r = {"detected": False, "phase_match": 0}
                    for gray in crops:
                        candidate = tier_fn(gray)
                        if candidate["detected"] and candidate.get("phase_match", 0) > r.get("phase_match", 0):
                            r = candidate
                    if not r["detected"]:
                        r = tier_fn(crops[0])  # fallback to first crop for reporting
                else:
                    r = tier_fn(crops[0])
                elapsed = (time.perf_counter() - t0) * 1000

                det = r["detected"]
                ph = r.get("phase_match", r.get("phase_match_a", 0))

                extras = []
                for k in ["mag_ratio", "structure_ratio", "phase_matched",
                           "matched_a", "matched_b", "phase_match_b", "tier"]:
                    if k in r:
                        v = r[k]
                        extras.append(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}")
                detail_str = ", ".join(extras)[:35]

                correct = (label == "WM" and det) or (label != "WM" and not det)
                mark = " OK" if correct else (" MISS" if label == "WM" else " FP")

                print(f"  {label:<5} {name:<28} {'YES' if det else 'no':<5} "
                      f"{ph:<7.4f} {detail_str:<35} {elapsed:>5.1f}ms{mark}")
                tier_results.append({
                    "label": label, "name": name, "correct": correct,
                    "detected": det, "time_ms": elapsed,
                })
            except Exception as e:
                print(f"  {label:<5} {name:<28} ERR   {e}")
                tier_results.append({
                    "label": label, "name": name, "correct": False,
                    "detected": False, "time_ms": 0,
                })

        results[tier_name] = tier_results

    # ── Summary ──
    print(f"\n{'=' * 95}")
    print("SUMMARY")
    print(f"{'=' * 95}")
    print(f"  {'Tier':<30} {'Accuracy':<10} {'Avg Time':<10} {'FP':<4} {'FN':<4}")
    print(f"  {'─'*30} {'─'*10} {'─'*10} {'─'*4} {'─'*4}")

    for tier_name, tr in results.items():
        total = len(tr)
        correct = sum(1 for r in tr if r["correct"])
        fp = sum(1 for r in tr if r["label"] != "WM" and r["detected"])
        fn = sum(1 for r in tr if r["label"] == "WM" and not r["detected"])
        avg_t = np.mean([r["time_ms"] for r in tr])
        print(f"  {tier_name:<30} {correct}/{total:<8} {avg_t:>6.1f}ms   {fp:<4} {fn:<4}")

    print(f"\n  FP = false positive | FN = false negative")
    print(f"  Note: 'sample_watermarked.png' is a known FN — 768x1365 portrait image")
    print(f"  where heavy aspect-ratio distortion destroys the watermark signal.")

    # Exit code
    tier_a_results = results.get("A: Raw FFT phase+mag", [])
    fp_count = sum(1 for r in tier_a_results if r["label"] != "WM" and r["detected"])
    if fp_count > 0:
        print(f"\nFAILED: Tier A has {fp_count} false positive(s)")
        sys.exit(1)
    else:
        print(f"\nPASSED: Tier A has 0 false positives")


if __name__ == "__main__":
    main()
