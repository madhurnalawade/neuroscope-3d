import { useEffect, useRef } from "react"
import { regionAt, regions, tissue, type Point } from "./data"

export function Brain({
  rotation = 0,
  zoom = 1,
  enabled = [true, true, true],
  opacity = 0.75,
  onRotate,
  hero = false,
}: {
  rotation?: number
  zoom?: number
  enabled?: boolean[]
  opacity?: number
  onRotate?: (n: number) => void
  hero?: boolean
}) {
  const ref = useRef<HTMLCanvasElement>(null)
  const drag = useRef<number | null>(null)
  useEffect(() => {
    const canvas = ref.current!
    const ctx = canvas.getContext("2d")!
    const size = 760
    canvas.width = size
    canvas.height = size
    ctx.clearRect(0, 0, size, size)
    const pts: {
      x: number
      y: number
      z: number
      shade: number
      region: number
    }[] = []
    const a = rotation - 0.35,
      ca = Math.cos(a),
      sa = Math.sin(a)
    for (let v = 0.035; v < Math.PI - 0.03; v += 0.012)
      for (let u = 0; u < Math.PI * 2; u += 0.014) {
        const sx = Math.sin(v) * Math.cos(u),
          sy = Math.cos(v),
          sz = Math.sin(v) * Math.sin(u)
        const groove =
          Math.sin(u * 19 + Math.sin(v * 11) * 1.8) *
          Math.sin(v * 27 + Math.sin(u * 9) * 1.2)
        const f = 1 + groove * 0.026
        let x = sx * 0.77 * f,
          y = sy * 0.91 * f,
          z = sz * 0.79 * f
        x += Math.sign(x) * 0.018
        y += 0.05 * Math.cos(u * 2)
        const rx = x * ca + z * sa,
          rz = -x * sa + z * ca
        const yy = y * 0.93 - rz * 0.27,
          zz = y * 0.27 + rz * 0.93
        const lighting =
          (0.36 + Math.max(0, -sx * 0.4 + sy * 0.2 + sz * 0.8) * 0.55) *
          (Math.abs(sx) < 0.025 ? 0.48 : 1)
        // Project the nested fixture onto the anterior surface as an illustrative cutaway.
        pts.push({
          x: rx,
          y: yy,
          z: zz,
          shade: lighting * (groove < -0.2 ? 0.66 : 1),
          region: sz > 0.3 ? regionAt(x, y, 0.13) : 0,
        })
      }
    pts.sort((a, b) => a.z - b.z)
    for (const p of pts) {
      const scale = 300 * zoom
      const light = Math.min(230, Math.floor(55 + p.shade * 174))
      const isRegion = p.region > 0 && enabled[p.region - 1]
      ctx.fillStyle = isRegion
        ? regions[p.region - 1].color
        : `rgb(${light + 5},${light + 3},${light})`
      ctx.globalAlpha = isRegion ? opacity : 1
      ctx.beginPath()
      ctx.ellipse(
        size / 2 + p.x * scale,
        size / 2 - p.y * scale,
        2.6 * zoom,
        3.1 * zoom,
        -0.3,
        0,
        Math.PI * 2,
      )
      ctx.fill()
    }
    ctx.globalAlpha = 1
  }, [rotation, zoom, enabled.join(","), opacity])
  return (
    <canvas
      ref={ref}
      className={`brain-canvas ${hero ? "hero-brain" : ""}`}
      role="img"
      aria-label="Synthetic sculptural brain illustration with nested tumour regions"
      onPointerDown={(e) => {
        if (onRotate) {
          drag.current = e.clientX
          e.currentTarget.setPointerCapture(e.pointerId)
        }
      }}
      onPointerMove={(e) => {
        if (drag.current !== null && onRotate) {
          onRotate(rotation + (e.clientX - drag.current) * 0.009)
          drag.current = e.clientX
        }
      }}
      onPointerUp={() => (drag.current = null)}
      onPointerCancel={() => (drag.current = null)}
    />
  )
}

export function Slice({
  axis = 2,
  point = [95, 95, 90],
  sequence = 1,
  enabled = [true, true, true],
  opacity = 0.65,
  onPoint,
  compact = false,
}: {
  axis?: number
  point?: Point
  sequence?: number
  enabled?: boolean[]
  opacity?: number
  onPoint?: (p: Point) => void
  compact?: boolean
}) {
  const ref = useRef<HTMLCanvasElement>(null)
  const dims = [160, 192, 160]
  const other = [0, 1, 2].filter((a) => a !== axis)
  useEffect(() => {
    const c = ref.current!,
      ctx = c.getContext("2d")!
    c.width = 256
    c.height = 256
    const img = ctx.createImageData(256, 256)
    for (let j = 0; j < 256; j++)
      for (let i = 0; i < 256; i++) {
        const q = [0, 0, 0]
        q[axis] = (point[axis] / (dims[axis] - 1)) * 2 - 1
        q[other[0]] = (i / 255) * 2 - 1
        q[other[1]] = 1 - (j / 255) * 2
        let t = tissue(q[0], q[1], q[2])
        const r = regionAt(q[0], q[1], q[2])
        if (sequence === 2) t = t ? 0.8 - t * 0.55 : 0
        if (sequence === 3) t = t ? t * 0.9 + (r ? 0.18 : 0) : 0
        if (sequence === 1 && r === 3) t = 0.92
        const n = (Math.sin(i * 13.37 + j * 83.1) * 43758.5) % 1
        const shade = t ? Math.max(0, Math.min(255, t * 260 + n * 12)) : 14
        const rgb =
          r && enabled[r - 1]
            ? regions[r - 1].color.match(/\w\w/g)!.map((v) => parseInt(v, 16))
            : null
        const k = (j * 256 + i) * 4
        for (let a = 0; a < 3; a++)
          img.data[k + a] = rgb
            ? shade * (1 - opacity) + rgb[a] * opacity
            : shade
        img.data[k + 3] = 255
      }
    ctx.putImageData(img, 0, 0)
    if (!compact) {
      const x = (point[other[0]] / (dims[other[0]] - 1)) * 256,
        y = (1 - point[other[1]] / (dims[other[1]] - 1)) * 256
      ctx.strokeStyle = "rgba(219,112,75,.8)"
      ctx.lineWidth = 0.65
      ctx.beginPath()
      ctx.moveTo(x, 0)
      ctx.lineTo(x, 256)
      ctx.moveTo(0, y)
      ctx.lineTo(256, y)
      ctx.stroke()
    }
  }, [axis, point.join(","), sequence, enabled.join(","), opacity, compact])
  return (
    <canvas
      ref={ref}
      className="slice-canvas"
      role="img"
      aria-label={`${["Sagittal", "Coronal", "Axial"][axis]} synthetic MRI slice`}
      onClick={(e) => {
        if (!onPoint) return
        const rect = e.currentTarget.getBoundingClientRect(),
          p = [...point] as Point
        p[other[0]] = Math.round(
          Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width)) *
            (dims[other[0]] - 1),
        )
        p[other[1]] = Math.round(
          Math.max(0, Math.min(1, 1 - (e.clientY - rect.top) / rect.height)) *
            (dims[other[1]] - 1),
        )
        onPoint(p)
      }}
    />
  )
}
