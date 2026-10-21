export const disclaimer =
  "Research and teaching prototype only. Model-predicted regions are not a diagnosis."
export const regions = [
  { id: "WT", name: "Whole tumour", color: "#db704b", volume: 0 },
  { id: "TC", name: "Tumour core", color: "#e9b56c", volume: 0 },
  { id: "ET", name: "Enhancing tumour", color: "#a8c7be", volume: 0 },
]
export type Point = [number, number, number]
export function tissue(x: number, y: number, z: number) {
  const r = (x * x) / 0.7 + (y * y) / 0.88 + (z * z) / 0.77
  if (r > 1) return 0
  const folds =
    Math.sin(x * 37 + Math.sin(y * 19) * 1.6 + z * 13) *
    Math.sin(y * 29 + z * 17 + Math.sin(x * 21))
  const ventricle =
    (Math.abs(x) - 0.11) ** 2 / 0.006 + (y * y) / 0.08 + (z * z) / 0.07 < 1
  return ventricle ? 0.1 : 0.43 + folds * 0.15 + (r > 0.78 ? 0.13 : 0)
}
export function regionAt(x: number, y: number, z: number) {
  const r =
    ((x - 0.34) / 0.28) ** 2 +
    ((y + 0.02) / 0.32) ** 2 +
    ((z - 0.13) / 0.3) ** 2
  return r < 0.19 ? 3 : r < 0.48 ? 2 : r < 1 ? 1 : 0
}
// Counts use the same nested masks and 1 mm isotropic voxel grid as the viewer.
const counts = [0, 0, 0]
for (let x = 0; x < 160; x++)
  for (let y = 0; y < 192; y++)
    for (let z = 0; z < 160; z++) {
      const r = regionAt(x / 79.5 - 1, y / 95.5 - 1, z / 79.5 - 1)
      if (r) {
        counts[0]++
        if (r >= 2) counts[1]++
        if (r === 3) counts[2]++
      }
    }
regions.forEach((r, i) => (r.volume = Number((counts[i] / 1000).toFixed(1))))
export const sequences = ["T1", "T1Gd", "T2", "FLAIR"] as const
export const sequenceCopy = [
  "The anatomical foundation. T1 reveals structure and the boundaries between brain tissues.",
  "Contrast, made visible. T1Gd highlights areas of contrast enhancement within the tumour region.",
  "Sensitive to water. T2 brings fluid-rich tissue and surrounding changes into view.",
  "The surrounding picture. FLAIR suppresses free fluid to help reveal tissue changes.",
]
