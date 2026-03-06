// ============================================================
// SynthID Watermark Detector
// Detects Google SynthID watermarks via FFT phase analysis
// at known carrier frequencies.
//
// Based on reverse-engineering research by aloshdenny/reverse-SynthID.
// Uses spectral codebook approach: SynthID embeds energy at fixed
// carrier frequencies with consistent phase values. We check phase
// match + magnitude anomaly at those bins.
// ============================================================

/* exported detectSynthIDWatermark */

(() => {
  "use strict";

  const SIZE = 512;

  // ── Versioned codebooks ──
  // Each entry represents a known SynthID embedding scheme.
  // When Google changes the algorithm, add a new entry; old ones stay.
  const CODEBOOKS = [
    {
      id: "gemini-v1",
      label: "Gemini / Imagen (2024-2025)",
      carriers: [
        { fy: -14,  fx: -14,  phase:  1.4409 },
        { fy:  14,  fx:  14,  phase: -1.4409 },
        { fy: -126, fx: -14,  phase:  2.3736 },
        { fy:  126, fx:  14,  phase: -2.3736 },
        { fy: -98,  fx:  14,  phase: -0.6109 },
        { fy:  98,  fx: -14,  phase:  0.6109 },
        { fy: -210, fx:  14,  phase: -1.1315 },
        { fy:  210, fx: -14,  phase:  1.1315 },
        { fy: -238, fx: -14,  phase:  1.6099 },
        { fy:  238, fx:  14,  phase: -1.6099 },
      ],
      thresholds: {
        strongPhaseMatched: 8,
        strongMagRatio: 5.0,
        moderatePhaseMatched: 6,
        moderatePhaseAvg: 0.55,
        moderateMagRatio: 2.0,
        phaseMatchCutoff: 0.7,
      },
    },
  ];

  // ── 1D Radix-2 Cooley-Tukey FFT (in-place) ──

  function fft1d(re, im) {
    const n = re.length;
    // Bit-reversal permutation
    for (let i = 1, j = 0; i < n; i++) {
      let bit = n >> 1;
      while (j & bit) {
        j ^= bit;
        bit >>= 1;
      }
      j ^= bit;
      if (i < j) {
        let tmp = re[i]; re[i] = re[j]; re[j] = tmp;
        tmp = im[i]; im[i] = im[j]; im[j] = tmp;
      }
    }
    // Butterfly
    for (let len = 2; len <= n; len <<= 1) {
      const halfLen = len >> 1;
      const angle = -2 * Math.PI / len;
      const wRe = Math.cos(angle);
      const wIm = Math.sin(angle);
      for (let i = 0; i < n; i += len) {
        let curRe = 1, curIm = 0;
        for (let j = 0; j < halfLen; j++) {
          const a = i + j;
          const b = a + halfLen;
          const tRe = curRe * re[b] - curIm * im[b];
          const tIm = curRe * im[b] + curIm * re[b];
          re[b] = re[a] - tRe;
          im[b] = im[a] - tIm;
          re[a] += tRe;
          im[a] += tIm;
          const nextRe = curRe * wRe - curIm * wIm;
          curIm = curRe * wIm + curIm * wRe;
          curRe = nextRe;
        }
      }
    }
  }

  // ── 2D FFT (row-column decomposition) ──

  function fft2d(gray, n) {
    const re = new Float64Array(n * n);
    const im = new Float64Array(n * n);
    for (let i = 0; i < n * n; i++) re[i] = gray[i];

    // FFT each row
    const rowRe = new Float64Array(n);
    const rowIm = new Float64Array(n);
    for (let r = 0; r < n; r++) {
      const off = r * n;
      for (let c = 0; c < n; c++) {
        rowRe[c] = re[off + c];
        rowIm[c] = im[off + c];
      }
      fft1d(rowRe, rowIm);
      for (let c = 0; c < n; c++) {
        re[off + c] = rowRe[c];
        im[off + c] = rowIm[c];
      }
    }

    // FFT each column
    const colRe = new Float64Array(n);
    const colIm = new Float64Array(n);
    for (let c = 0; c < n; c++) {
      for (let r = 0; r < n; r++) {
        colRe[r] = re[r * n + c];
        colIm[r] = im[r * n + c];
      }
      fft1d(colRe, colIm);
      for (let r = 0; r < n; r++) {
        re[r * n + c] = colRe[r];
        im[r * n + c] = colIm[r];
      }
    }

    return { re, im };
  }

  // ── fftshift: swap quadrants so DC is at center ──

  function fftshift(arr, n) {
    const half = n >> 1;
    const out = new Float64Array(n * n);
    for (let r = 0; r < n; r++) {
      for (let c = 0; c < n; c++) {
        const sr = (r + half) % n;
        const sc = (c + half) % n;
        out[sr * n + sc] = arr[r * n + c];
      }
    }
    return out;
  }

  // ── Image → grayscale pixel array via OffscreenCanvas ──

  function imageToGrayscale(imgElement) {
    const canvas = document.createElement("canvas");
    canvas.width = SIZE;
    canvas.height = SIZE;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) return null;

    // Center-crop to square before drawing
    const natW = imgElement.naturalWidth || imgElement.width;
    const natH = imgElement.naturalHeight || imgElement.height;
    const side = Math.min(natW, natH);
    const sx = (natW - side) / 2;
    const sy = (natH - side) / 2;

    ctx.drawImage(imgElement, sx, sy, side, side, 0, 0, SIZE, SIZE);
    const imageData = ctx.getImageData(0, 0, SIZE, SIZE);
    const pixels = imageData.data; // RGBA

    const gray = new Float64Array(SIZE * SIZE);
    for (let i = 0; i < SIZE * SIZE; i++) {
      const off = i * 4;
      // ITU-R BT.601 luma
      gray[i] = 0.299 * pixels[off] + 0.587 * pixels[off + 1] + 0.114 * pixels[off + 2];
    }
    return gray;
  }

  // Also support ArrayBuffer input (for images fetched as bytes)
  function bufferToGrayscale(buffer) {
    return new Promise((resolve) => {
      const blob = new Blob([buffer]);
      const url = URL.createObjectURL(blob);
      const img = new Image();
      img.onload = () => {
        const gray = imageToGrayscale(img);
        URL.revokeObjectURL(url);
        resolve(gray);
      };
      img.onerror = () => {
        URL.revokeObjectURL(url);
        resolve(null);
      };
      img.src = url;
    });
  }

  // ── Phase difference (circular, normalized to 0-1 match score) ──

  function phaseDiff(actual, expected) {
    const d = actual - expected;
    // atan2(sin(d), cos(d)) handles wrap-around
    const wrapped = Math.abs(Math.atan2(Math.sin(d), Math.cos(d)));
    return 1 - wrapped / Math.PI;
  }

  // ── Core detection ──

  function analyzeSpectrum(gray, codebook) {
    const { re, im } = fft2d(gray, SIZE);

    // Compute magnitude and phase, then fftshift
    const magRaw = new Float64Array(SIZE * SIZE);
    const phaseRaw = new Float64Array(SIZE * SIZE);
    for (let i = 0; i < SIZE * SIZE; i++) {
      magRaw[i] = Math.sqrt(re[i] * re[i] + im[i] * im[i]);
      phaseRaw[i] = Math.atan2(im[i], re[i]);
    }

    const mag = fftshift(magRaw, SIZE);
    const phase = fftshift(phaseRaw, SIZE);

    const center = SIZE >> 1;
    const carriers = codebook.carriers;
    const t = codebook.thresholds;
    const phaseScores = [];
    const magRatios = [];

    for (const c of carriers) {
      const y = c.fy + center;
      const x = c.fx + center;
      if (y < 0 || y >= SIZE || x < 0 || x >= SIZE) continue;

      const idx = y * SIZE + x;
      phaseScores.push(phaseDiff(phase[idx], c.phase));

      // Annular ring comparison: magnitude at carrier vs median of
      // same-radius points at different angles
      const radius = Math.sqrt(c.fy * c.fy + c.fx * c.fx);
      const ringMags = [];
      for (let a = 0; a < 24; a++) {
        const angle = (a / 24) * 2 * Math.PI;
        const ry = Math.round(radius * Math.sin(angle)) + center;
        const rx = Math.round(radius * Math.cos(angle)) + center;
        if (ry >= 0 && ry < SIZE && rx >= 0 && rx < SIZE &&
            (Math.abs(ry - y) > 2 || Math.abs(rx - x) > 2)) {
          ringMags.push(mag[ry * SIZE + rx]);
        }
      }
      if (ringMags.length > 0) {
        ringMags.sort((a, b) => a - b);
        const median = ringMags[ringMags.length >> 1];
        magRatios.push(median > 0 ? mag[idx] / median : 0);
      }
    }

    if (phaseScores.length === 0) {
      return { detected: false, confidence: 0, codebookId: codebook.id };
    }

    const avgPhase = phaseScores.reduce((a, b) => a + b, 0) / phaseScores.length;
    const phaseMatched = phaseScores.filter((s) => s > t.phaseMatchCutoff).length;
    const avgMag = magRatios.length > 0
      ? magRatios.reduce((a, b) => a + b, 0) / magRatios.length
      : 0;

    const strong = phaseMatched >= t.strongPhaseMatched && avgMag > t.strongMagRatio;
    const moderate = phaseMatched >= t.moderatePhaseMatched &&
                     avgPhase > t.moderatePhaseAvg &&
                     avgMag > t.moderateMagRatio;
    const detected = strong || moderate;

    let confidence = 0;
    if (strong) {
      confidence = Math.min(0.95, 0.6 + avgPhase * 0.2 + Math.min(avgMag / 50, 0.15));
    } else if (moderate) {
      confidence = Math.min(0.75, 0.3 + avgPhase * 0.3 + Math.min(avgMag / 10, 0.15));
    }

    return {
      detected,
      confidence,
      codebookId: codebook.id,
      codebookLabel: codebook.label,
      phaseMatch: avgPhase,
      phaseMatched,
      magRatio: avgMag,
    };
  }

  // ── Public API ──

  /**
   * Detect SynthID watermark in an image.
   *
   * @param {HTMLImageElement|ArrayBuffer} input - Image element or raw bytes
   * @returns {Promise<{detected: boolean, confidence: number, source?: string, details?: object}>}
   */
  async function detectSynthIDWatermark(input) {
    let gray = null;

    if (input instanceof HTMLImageElement) {
      // Skip tiny images — watermark won't survive
      const w = input.naturalWidth || input.width;
      const h = input.naturalHeight || input.height;
      if (w < 128 || h < 128) {
        return { detected: false, confidence: 0, reason: "image_too_small" };
      }
      gray = imageToGrayscale(input);
    } else if (input instanceof ArrayBuffer || input instanceof Uint8Array) {
      const buf = input instanceof Uint8Array ? input.buffer : input;
      gray = await bufferToGrayscale(buf);
    }

    if (!gray) {
      return { detected: false, confidence: 0, reason: "decode_failed" };
    }

    // Test against all codebook versions, return best match
    let best = { detected: false, confidence: 0 };
    for (const codebook of CODEBOOKS) {
      const result = analyzeSpectrum(gray, codebook);
      if (result.detected && result.confidence > best.confidence) {
        best = result;
      }
    }

    if (best.detected) {
      return {
        detected: true,
        confidence: best.confidence,
        source: "Google (SynthID)",
        codebook: best.codebookLabel || best.codebookId,
        details: {
          phaseMatch: best.phaseMatch,
          phaseMatched: best.phaseMatched,
          magRatio: best.magRatio,
        },
      };
    }

    return { detected: false, confidence: 0 };
  }

  // Expose to content.js scope
  window.__acsSynthID = { detectSynthIDWatermark };
})();
