import { useEffect, useRef, useState } from "react"
import { Brain, Slice } from "./Imaging"
import {
  disclaimer,
  regions,
  sequences,
  sequenceCopy,
  type Point,
} from "./data"

function Arrow({ diagonal = false }: { diagonal?: boolean }) {
  return <span aria-hidden="true">{diagonal ? "↗" : "↗"}</span>
}
function Brand() {
  return (
    <a className="brand" href="#/">
      <span className="brand-icon">
        <i />
        <i />
        <i />
      </span>
      voxel<span className="brand-sub">RESEARCH IMAGING</span>
    </a>
  )
}
type Route = "landing" | "analyse" | "viewer"
const routeOf = (hash: string): Route =>
  hash === "#/analyse" ? "analyse" : hash === "#/viewer" ? "viewer" : "landing"
function App() {
  const [route, setRoute] = useState<Route>(routeOf(location.hash))
  const [ready, setReady] = useState(false)
  const [study, setStudy] = useState<Study | null>(null)
  const [attempt, setAttempt] = useState(0)
  useEffect(() => {
    const f = () => {
      setRoute(routeOf(location.hash))
      window.scrollTo(0, 0)
    }
    window.addEventListener("hashchange", f)
    return () => window.removeEventListener("hashchange", f)
  }, [])
  // The viewer needs a finished analysis; until there is one, it opens the analyse page.
  useEffect(() => {
    if (route === "viewer" && !ready) location.replace("#/analyse")
  }, [route, ready])
  const page = route === "viewer" && !ready ? "analyse" : route
  return (
    <>
      <div hidden={page !== "viewer"}>
        {study && (
          <Viewer
            study={study}
            onNew={() => {
              setReady(false)
              setAttempt(attempt + 1)
              location.hash = "#/analyse"
            }}
          />
        )}
      </div>
      <div hidden={page !== "analyse"}>
        <Analyse
          key={attempt}
          onComplete={(s) => {
            setStudy(s)
            setReady(true)
            location.hash = "#/viewer"
          }}
        />
      </div>
      <div hidden={page !== "landing"}>
        <Landing />
      </div>
    </>
  )
}
function Landing() {
  const [sequence, setSequence] = useState(1),
    [scroll, setScroll] = useState(0),
    [menu, setMenu] = useState(false)
  useEffect(() => {
    let frame = 0
    const onScroll = () => {
      cancelAnimationFrame(frame)
      frame = requestAnimationFrame(() =>
        setScroll(Math.min(window.scrollY / 1000, 1)),
      )
    }
    window.addEventListener("scroll", onScroll, { passive: true })
    return () => {
      window.removeEventListener("scroll", onScroll)
      cancelAnimationFrame(frame)
    }
  }, [])
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches
  return (
    <div className="landing">
      <header className="site-header">
        <Brand />
        <button
          className="mobile-menu"
          onClick={() => setMenu(!menu)}
          aria-expanded={menu}
        >
          Menu
        </button>
        <nav className={menu ? "open" : ""}>
          <a href="#science">The science</a>
          <a href="#workflow">The workflow</a>
          <a href="#boundaries">Our boundaries</a>
          <a href="#/analyse">Analyse a study</a>
        </nav>
        <a className="header-cta" href="#/viewer">
          Open viewer <Arrow />
        </a>
      </header>
      <main>
        <section className="hero">
          <div className="hero-topline">
            <span>
              <i className="status-dot" /> A NEW PERSPECTIVE ON RESEARCH MRI
            </span>
            <span>V.01 / INTERACTIVE PROTOTYPE</span>
          </div>
          <div className="hero-grid">
            <div className="hero-copy">
              <div className="eyebrow">COMPLEX DATA. CLEARER PERSPECTIVES.</div>
              <h1>
                See a brain
                <br />
                tumour in <em>3D.</em>
                <br />
                Measure it.
                <br />
                <span className="muted-title">Know its limits.</span>
              </h1>
              <p>
                Explore model-predicted tumour regions in research MRI scans.
                Inspect the 3D shape, linked slices, estimated volumes and model
                performance.
              </p>
              <div className="hero-actions">
                <a className="button primary" href="#/viewer">
                  Explore the viewer <Arrow />
                </a>
                <a className="text-link" href="#science">
                  How it works <span>↓</span>
                </a>
              </div>
              <div className="safety-line">
                <span className="small-cross">+</span> Research and teaching
                prototype only — not for diagnosis.
              </div>
            </div>
            <div className="anatomy-stage">
              <div className="stage-top">
                <span>FIG. 01 — SEGMENTED ANATOMY</span>
                <span>3D RECONSTRUCTION / DEMO</span>
              </div>
              <div className="anatomy-grid" />
              <div className="anatomy-ring ring-one" />
              <div className="anatomy-ring ring-two" />
              <span className="axis-label axis-s">S</span>
              <span className="axis-label axis-l">L</span>
              <span className="axis-label axis-r">R</span>
              <span className="axis-label axis-i">I</span>
              <div
                className="brain-wrap"
                style={{
                  transform: `translateX(${
                    reduced ? 0 : scroll * 18
                  }px) scale(${reduced ? 1 : 1 + scroll * 0.06})`,
                }}
              >
                <Brain hero rotation={reduced ? 0 : scroll * 0.45} />
              </div>
              <div className="brain-callout">
                <span className="callout-line" />
                <i /> PREDICTED REGIONS
                <span className="callout-sub">WT / TC / ET</span>
              </div>
              <div className="stage-bottom">
                <span>
                  SAMPLE-001
                  <br />
                  <b>SYNTHETIC TEACHING STUDY</b>
                </span>
                <span>
                  X 095 · Y 095 · Z 090
                  <br />
                  <b>1.0 × 1.0 × 1.0 mm</b>
                </span>
              </div>
            </div>
          </div>
          <div className="hero-footer">
            <span>FROM VOLUME TO UNDERSTANDING.</span>
            <span>
              SCROLL TO EXPLORE <span className="scroll-arrow">↓</span>
            </span>
            <span>RESEARCH, NOT DIAGNOSIS.</span>
          </div>
        </section>
        <section className="sequence-section section" id="science">
          <div className="section-heading">
            <span className="section-index">01 / THE INPUT</span>
            <h2>
              The scan,
              <br />
              <em>opened up.</em>
            </h2>
            <p>
              One study. Four complementary perspectives.
              <br />
              Each sequence makes something different visible.
            </p>
          </div>
          <div className="sequence-grid">
            {sequences.map((s, i) => (
              <button
                key={s}
                className={`sequence-card ${sequence === i ? "selected" : ""}`}
                onClick={() => setSequence(i)}
                aria-pressed={sequence === i}
              >
                <div className="sequence-label">
                  <span>0{i + 1}</span>
                  <b>{s}</b>
                  <span>{sequence === i ? "↗" : "+"}</span>
                </div>
                <Slice sequence={i} compact />
                <span className="sequence-type">
                  {
                    [
                      "ANATOMICAL STRUCTURE",
                      "CONTRAST ENHANCEMENT",
                      "FLUID-SENSITIVE",
                      "FLUID SUPPRESSION",
                    ][i]
                  }
                </span>
              </button>
            ))}
          </div>
          <div className="sequence-description">
            <span className="eyebrow">
              {sequences[sequence]} / A DIFFERENT LENS
            </span>
            <p>{sequenceCopy[sequence]}</p>
            <span className="mono">ILLUSTRATIVE SYNTHETIC SLICES</span>
          </div>
        </section>
        <section className="regions-section section" id="workflow">
          <div>
            <span className="section-index">02 / THE INTERPRETATION</span>
            <h2>
              From voxels
              <br />
              to <em>regions.</em>
            </h2>
            <p>
              A model labels individual voxels. Together, they form nested
              regions you can explore — not a diagnosis you should act on.
            </p>
            <div className="nested-formula">
              ET <span>⊆</span> TC <span>⊆</span> WT
            </div>
          </div>
          <div className="nested-diagram">
            <div className="region-wt">
              <span>WT / Whole tumour</span>
              <div className="region-tc">
                <span>TC / Tumour core</span>
                <div className="region-et">
                  ET
                  <br />
                  <small>Enhancing tumour</small>
                </div>
              </div>
            </div>
          </div>
        </section>
        <section className="inspect-section section">
          <div className="section-heading">
            <span className="section-index">03 / THE WORKSPACE</span>
            <h2>
              Look closer.
              <br />
              <em>Stay connected.</em>
            </h2>
            <p>
              One shared point. Every perspective.
              <br />
              Rotate the anatomy, move through slices, and see how the regions
              relate.
            </p>
          </div>
          <a href="#/viewer" className="workspace-preview">
            <div className="preview-bar">
              <span>
                <i className="status-dot" /> SAMPLE-001 / LINKED VIEWS
              </span>
              <span>OPEN THE INTERACTIVE VIEWER ↗</span>
            </div>
            <div className="preview-images">
              <div className="preview-brain">
                <Brain />
              </div>
              {[2, 1, 0].map((a) => (
                <div key={a}>
                  <span className="mono">
                    {["SAGITTAL", "CORONAL", "AXIAL"][a]}
                  </span>
                  <Slice axis={a} />
                </div>
              ))}
            </div>
          </a>
        </section>
        <section className="measure-section section">
          <div>
            <span className="section-index">04 / THE CONTEXT</span>
            <h2>
              Numbers need
              <br />
              <em>perspective.</em>
            </h2>
            <p>
              Estimated volume describes a region. Evaluation describes how
              predictions compare with expert labels. Neither tells the whole
              story.
            </p>
            <a className="text-link" href="#/viewer">
              Explore the measurements <Arrow />
            </a>
          </div>
          <div className="measurement">
            <span className="eyebrow">WHOLE TUMOUR / SAMPLE ESTIMATE</span>
            <div className="big-measure">
              {regions[0].volume}
              <small>mL</small>
            </div>
            <div className="measure-rule" />
            <div className="metric-row">
              <span>Illustrative Dice</span>
              <strong>0.89</strong>
              <span>Illustrative sensitivity</span>
              <strong>0.91</strong>
            </div>
            <p>
              Illustrative sample evaluation — not validated performance. No
              trained model is running in this prototype.
            </p>
          </div>
        </section>
        <section className="boundaries section" id="boundaries">
          <span className="section-index">05 / OUR BOUNDARIES</span>
          <h2>
            A tool for exploration.
            <br />
            <em>Not a clinical answer.</em>
          </h2>
          <div className="boundary-grid">
            <p>
              Built for research and teaching with de-identified data in mind.
              This prototype uses a synthetic teaching fixture only, never real
              patient information.
            </p>
            <p>
              No diagnosis. No treatment guidance.
              <br />
              No validated model performance claims.
              <br />
              Always understand the limits of the output.
            </p>
            <a className="button primary" href="#/viewer">
              Explore responsibly <Arrow />
            </a>
          </div>
        </section>
      </main>
      <footer className="site-footer">
        <Brand />
        <span>AN EXPERIMENT IN SEEING CLEARER.</span>
        <span>RESEARCH & TEACHING ONLY © 2026</span>
      </footer>
    </div>
  )
}

type StudyInput = {
  label: string
  name: string
  size: number
  hash: string
}
type Study = {
  sample: boolean
  inputs: StudyInput[]
  elapsed: number
}

// ------------------------------------------------------------------ analyse a study
type Role = typeof sequences[number] | "expert" | "study4d" | "ignore" | ""
type Header = {
  ndim?: number
  dim?: number[]
  pix?: number[]
  error?: string
}
type Item = {
  file: File
  head: Header
  role: Role
}
type StepState = "" | "run" | "ok" | "fail"
const MAX_BYTES = 100 * 1024 * 1024
const DECATHLON_ORDER = ["FLAIR", "T1", "T1Gd", "T2"]
const STEPS = [
  "Fingerprint files (SHA-256, in this browser)",
  "Structural checks",
  "Region segmentation — simulated",
  "Volumes and viewer",
]
const fmtMB = (b: number) =>
  (b / 1048576).toFixed(b < 1048576 * 10 ? 2 : 1) + " MB"
const wait = (ms: number) => new Promise((r) => setTimeout(r, ms))

// Reads only the NIfTI-1/2 header, in this browser; nothing is uploaded.
async function readHeader(file: File): Promise<Header> {
  let bytes: Uint8Array
  try {
    if (/\.gz$/i.test(file.name)) {
      const reader = file
        .stream()
        .pipeThrough(new DecompressionStream("gzip"))
        .getReader()
      const chunks: Uint8Array[] = []
      let size = 0
      try {
        while (size < 560) {
          const { value, done } = await reader.read()
          if (done) break
          chunks.push(value)
          size += value.length
        }
      } catch {
        return { error: "not a valid gzip file" }
      }
      reader.cancel().catch(() => {})
      bytes = new Uint8Array(size)
      let o = 0
      for (const c of chunks) {
        bytes.set(c, o)
        o += c.length
      }
    } else {
      bytes = new Uint8Array(await file.slice(0, 560).arrayBuffer())
    }
  } catch {
    return { error: "cannot be read" }
  }
  if (bytes.length < 348) return { error: "too short to be NIfTI" }
  const dv = new DataView(bytes.buffer)
  for (const le of [true, false]) {
    const sz = dv.getInt32(0, le)
    if (sz === 348) {
      const nd = dv.getInt16(40, le)
      const dim = [1, 2, 3, 4].map((k) => dv.getInt16(40 + 2 * k, le))
      const pix = [1, 2, 3].map((k) => dv.getFloat32(76 + 4 * k, le))
      return { ndim: nd, dim: dim.slice(0, nd), pix }
    }
    if (sz === 540 && bytes.length >= 540) {
      const nd = Number(dv.getBigInt64(16, le))
      const dim = [1, 2, 3, 4].map((k) =>
        Number(dv.getBigInt64(16 + 8 * k, le)),
      )
      const pix = [1, 2, 3].map((k) => dv.getFloat64(104 + 8 * k, le))
      return { ndim: nd, dim: dim.slice(0, nd), pix }
    }
  }
  return { error: "no NIfTI header found" }
}

// Same filename token rules as the NeuroScope server's sequence detector.
function detectRole(name: string): Role {
  const n = name.replace(/\.nii(\.gz)?$/i, "").replace(/\.gz$/i, "")
  const b = (re: string) =>
    new RegExp(`(?:^|[_\\-.\\s])(${re})(?:$|[_\\-.\\s])`, "i").test(n)
  if (b("seg|label|labels|mask|gt|expert")) return "expert"
  if (b("t1(?:ce|gd|c|post|\\+c|\\+gd|-ce|-gd|-post|_ce|_gd|_post)"))
    return "T1Gd"
  if (b("t2[-_]?flair|flair")) return "FLAIR"
  if (b("t2(?:w|_w|-w|weighted|_weighted)?")) return "T2"
  if (b("t1(?:w|_w|-w|weighted|_weighted)?")) return "T1"
  return ""
}
function planStudy(items: Item[], order: string[]) {
  const live = items.filter((it) => it.role && it.role !== "ignore")
  const four = live.filter((it) => it.role === "study4d")
  const seqs = live.filter((it) =>
    (sequences as readonly string[]).includes(it.role),
  )
  const experts = live.filter((it) => it.role === "expert")
  const problems: string[] = []
  for (const it of live) {
    const n = `“${it.file.name}”`,
      h = it.head
    if (h.error) problems.push(`${n} cannot be read as NIfTI (${h.error}).`)
    else if (it.role !== "study4d") {
      const nd = h.ndim === 4 && h.dim?.[3] === 1 ? 3 : h.ndim
      if (nd !== 3)
        problems.push(
          `${n} is ${nd}D (${h.dim?.join(" × ")}); each sequence must be a 3D volume.`,
        )
    }
  }
  if (items.some((it) => !it.role))
    problems.push("Assign a sequence to every file (or mark it “ignore”).")
  if (experts.length > 1) problems.push("Only one expert mask can be attached.")
  if (four.length > 1) problems.push("Drop only one 4-channel study at a time.")
  if (four.length && seqs.length)
    problems.push(
      "Use either one 4-channel file or four sequence files, not both.",
    )
  if (four.length === 1) {
    if (new Set(order).size !== 4)
      problems.push("Each channel must map to a different sequence.")
    return {
      mode: "4d" as const,
      four: four[0],
      seqs,
      expert: experts[0],
      problems,
    }
  }
  const missing = sequences.filter((s) => !seqs.some((it) => it.role === s))
  const dupes = sequences.filter(
    (s) => seqs.filter((it) => it.role === s).length > 1,
  )
  if (missing.length) problems.push(`Missing: ${missing.join(", ")}.`)
  if (dupes.length)
    problems.push(`More than one file for: ${dupes.join(", ")}.`)
  const grids = new Set(
    seqs
      .filter((it) => it.head.dim && (it.head.ndim ?? 0) >= 3)
      .map((it) => it.head.dim!.slice(0, 3).join("×")),
  )
  if (grids.size > 1)
    problems.push(
      `The sequences have different grids (${[...grids].join(" vs ")}); they must be co-registered.`,
    )
  return {
    mode: "4files" as const,
    four: undefined,
    seqs,
    expert: experts[0],
    problems,
  }
}
async function sha256(file: File) {
  try {
    const digest = await crypto.subtle.digest(
      "SHA-256",
      await file.arrayBuffer(),
    )
    return Array.from(new Uint8Array(digest))
      .map((b) => b.toString(16).padStart(2, "0"))
      .join("")
  } catch {
    return "Unavailable"
  }
}
function Analyse({ onComplete }: { onComplete: (s: Study) => void }) {
  const [items, setItems] = useState<Item[]>([]),
    [order, setOrder] = useState<string[]>([...DECATHLON_ORDER]),
    [sample, setSample] = useState(false),
    [notice, setNotice] = useState(""),
    [over, setOver] = useState(false),
    [steps, setSteps] = useState<StepState[]>(["", "", "", ""]),
    [times, setTimes] = useState<string[]>(["", "", "", ""]),
    [running, setRunning] = useState(false)
  const token = useRef(0)
  useEffect(() => () => void token.current++, [])
  const p = planStudy(items, order)
  const canRun =
    !running && (sample || (items.length > 0 && !p.problems.length))
  const mark = (k: number, s: StepState, t = "") => {
    setSteps((old) => old.map((v, i) => (i === k ? s : v)))
    setTimes((old) => old.map((v, i) => (i === k ? t : v)))
  }
  const stop = () => {
    token.current++
    setRunning(false)
    setSteps(["", "", "", ""])
    setTimes(["", "", "", ""])
  }
  const clear = () => {
    stop()
    setItems([])
    setOrder([...DECATHLON_ORDER])
    setSample(false)
    setNotice("")
  }
  async function addFiles(list: File[]) {
    stop()
    setSample(false)
    const skipped: string[] = []
    const added: Item[] = []
    for (const f of list) {
      if (!/\.nii(\.gz)?$/i.test(f.name)) {
        skipped.push(`“${f.name}” is not a .nii or .nii.gz file.`)
        continue
      }
      if (!f.size || f.size > MAX_BYTES) {
        skipped.push(`“${f.name}” must be nonempty and no larger than 100 MB.`)
        continue
      }
      if (
        [...items, ...added].some(
          (it) => it.file.name === f.name && it.file.size === f.size,
        )
      )
        continue
      const head = await readHeader(f)
      const is4 = head.ndim === 4 && head.dim?.[3] === 4
      added.push({ file: f, head, role: is4 ? "study4d" : detectRole(f.name) })
    }
    setItems((old) => [...old, ...added])
    setNotice(skipped.join(" "))
  }
  const loadSample = () => {
    clear()
    setSample(true)
  }
  async function run() {
    if (!canRun) return
    const me = ++token.current
    const t0 = performance.now()
    const secs = (t: number) =>
      ((performance.now() - t) / 1000).toFixed(1) + " s"
    setRunning(true)
    setSteps(["run", "", "", ""])
    setTimes(["", "", "", ""])
    let inputs: StudyInput[]
    if (sample) {
      inputs = sequences.map((s) => ({
        label: s,
        name: `SAMPLE-001_${s}.nii.gz`,
        size: 0,
        hash: `Fixture identifier: SAMPLE-001-${s}`,
      }))
      await wait(400)
    } else {
      const used = [
        ...(p.four ? [p.four] : p.seqs),
        ...(p.expert ? [p.expert] : []),
      ]
      const hashes = new Map<File, string>()
      for (const it of used) hashes.set(it.file, await sha256(it.file))
      const entry = (label: string, it: Item) => ({
        label,
        name: it.file.name,
        size: it.file.size,
        hash: hashes.get(it.file)!,
      })
      inputs = p.four
        ? sequences.map((s) => entry(`${s} (CH ${order.indexOf(s)})`, p.four!))
        : sequences.map((s) => entry(s, p.seqs.find((it) => it.role === s)!))
      if (p.expert) inputs.push(entry("EXPERT MASK", p.expert))
    }
    if (me !== token.current) return
    mark(0, "ok", secs(t0))
    const t1 = performance.now()
    mark(1, "run")
    await wait(500)
    if (me !== token.current) return
    mark(1, "ok", secs(t1))
    const t2 = performance.now()
    mark(2, "run")
    await wait(2200)
    if (me !== token.current) return
    mark(2, "ok", secs(t2))
    const t3 = performance.now()
    mark(3, "run")
    await wait(600)
    if (me !== token.current) return
    mark(3, "ok", `${secs(t3)} · total ${secs(t0)}`)
    setRunning(false)
    onComplete({ sample, inputs, elapsed: (performance.now() - t0) / 1000 })
  }
  const roleName = (r: string) =>
    (({
      study4d: "4-channel study",
      expert: "Expert mask",
      ignore: "ignore",
    }) as Record<string, string>)[r] || r
  const current = steps.findIndex((s) => s === "run")
  const status = running
    ? `${STEPS[current] ?? "Finishing"}…`
    : items.length
      ? p.problems.length
        ? p.problems.join(" ")
        : p.mode === "4d"
          ? `Ready: one 4-channel study${
              p.expert ? " + expert mask" : ""
            }, split into four sequences by channel.`
          : `Ready: T1, T1Gd, T2 and FLAIR${p.expert ? " + expert mask" : ""}.`
      : sample
        ? "Synthetic sample loaded. Ready to run."
        : "No study loaded. Drop files or load the sample study."
  return (
    <div className="viewer analyse">
      <header className="viewer-header">
        <Brand />
        <a href="#/" className="back-link">
          ← Back to overview
        </a>
        <span className="mono">ANALYSE A STUDY / V.01</span>
      </header>
      <div className="viewer-safety">
        <span>+</span>
        {disclaimer}
        <span className="sample-badge">LOCAL ONLY · SIMULATED RUN</span>
      </div>
      <div className="workspace-title">
        <div>
          <span className="eyebrow">ANALYSE A STUDY</span>
          <h1>Start with a study.</h1>
        </div>
        <div className="workspace-actions">
          <button className="button primary" onClick={loadSample}>
            Load sample study <Arrow />
          </button>
        </div>
      </div>
      <div className="setup-layout">
        <div className="setup-main">
          <div className="panel-heading">
            <span>01 / ONE STUDY, ONE DROP</span>
            <span>NIfTI · .nii / .nii.gz</span>
          </div>
          <p className="setup-intro">Bring the perspectives together.</p>
          <p className="setup-note">
            One 4-channel NIfTI, or the four sequences as separate files (T1,
            T1Gd, T2, FLAIR). An expert mask is optional. Headers are read in
            this browser to check the study, and files stay on your device.
            Prototype workflow: the prediction shown next is the synthetic
            sample, not computed from your files.
          </p>
          <label
            className={`analyse-drop ${over ? "over" : ""}`}
            onDragEnter={(e) => {
              e.preventDefault()
              setOver(true)
            }}
            onDragOver={(e) => {
              e.preventDefault()
              setOver(true)
            }}
            onDragLeave={() => setOver(false)}
            onDrop={(e) => {
              e.preventDefault()
              setOver(false)
              void addFiles([...e.dataTransfer.files])
            }}
          >
            <span className="drop-big">Drop the study here</span>
            <span className="drop-small">
              <b>one file</b> with the four sequences stacked (X × Y × Z × 4) ·{" "}
              <b>or four files</b> named *_t1, *_t1ce, *_t2, *_flair · optional
              *_seg mask
            </span>
            <span className="mono">OR CLICK TO SELECT ↗</span>
            <input
              type="file"
              multiple
              accept=".nii,.gz"
              aria-label="Select NIfTI files for one study"
              onChange={(e) => {
                void addFiles([...(e.target.files ?? [])])
                e.target.value = ""
              }}
            />
          </label>
          {items.length > 0 && (
            <div className="analyse-manifest">
              <table>
                <thead>
                  <tr>
                    <th>ROLE</th>
                    <th>FILE</th>
                    <th>GRID · SPACING (HEADER)</th>
                    <th>SIZE</th>
                    <th />
                  </tr>
                </thead>
                <tbody>
                  {items.map((it, k) => {
                    const h = it.head,
                      is4 = h.ndim === 4 && h.dim?.[3] === 4
                    return (
                      <tr key={`${it.file.name}-${it.file.size}`}>
                        <td>
                          <select
                            value={it.role}
                            aria-label={`Role of ${it.file.name}`}
                            onChange={(e) => {
                              stop()
                              setItems(
                                items.map((x, j) =>
                                  j === k
                                    ? { ...x, role: e.target.value as Role }
                                    : x,
                                ),
                              )
                            }}
                          >
                            {!it.role && <option value="">choose…</option>}
                            {(is4
                              ? ["study4d", "ignore"]
                              : [...sequences, "expert", "ignore"]
                            )
                              .map((r) => (
                                <option key={r} value={r}>
                                  {roleName(r)}
                                </option>
                              ))}
                          </select>
                        </td>
                        <td className="file-cell">{it.file.name}</td>
                        <td className="mono">
                          {h.error ? (
                            <span className="bad">{h.error}</span>
                          ) : (
                            `${h.dim?.join(" × ")}${
                              h.pix
                                ? ` · ${h.pix.map((v) => v.toFixed(2)).join(" × ")} mm`
                                : ""
                            }`
                          )}
                        </td>
                        <td className="mono">{fmtMB(it.file.size)}</td>
                        <td>
                          <button
                            className="remove"
                            aria-label={`Remove ${it.file.name}`}
                            onClick={() => {
                              stop()
                              setItems(items.filter((_, j) => j !== k))
                            }}
                          >
                            ×
                          </button>
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
              {p.mode === "4d" && (
                <div className="channel-map">
                  <p>
                    Which sequence is in each channel? Defaults follow the
                    Medical Segmentation Decathlon (FLAIR, T1, T1Gd, T2).
                  </p>
                  <div>
                    {[0, 1, 2, 3].map((c) => (
                      <label key={c}>
                        CHANNEL {c}
                        <select
                          value={order[c]}
                          onChange={(e) => {
                            stop()
                            setOrder(
                              order.map((v, i) =>
                                i === c ? e.target.value : v,
                              ),
                            )
                          }}
                        >
                          {sequences.map((s) => (
                            <option key={s}>{s}</option>
                          ))}
                        </select>
                      </label>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
          <div
            className={`status-message ${
              items.length && p.problems.length && !running ? "has-problem" : ""
            }`}
            role="status"
            aria-live="polite"
          >
            {status}
          </div>
          {notice && <p className="error-label">{notice}</p>}
          <div className="setup-bottom">
            <span className="mono">100 MB MAX / FILE · LOCAL ONLY</span>
            <div className="setup-buttons">
              {(items.length > 0 || sample) && !running && (
                <button className="button outline" onClick={clear}>
                  Clear
                </button>
              )}
              {running ? (
                <button className="button outline" onClick={stop}>
                  Cancel processing
                </button>
              ) : (
                <button
                  className="button primary"
                  onClick={run}
                  disabled={!canRun}
                >
                  Run analysis <Arrow />
                </button>
              )}
            </div>
          </div>
          <ol className="run-steps" aria-label="Run log">
            {STEPS.map((s, k) => (
              <li key={s} className={steps[k]}>
                <span>{s}</span>
                <time>{times[k]}</time>
              </li>
            ))}
          </ol>
        </div>
        <aside className="setup-aside">
          <div className="mini-brain">
            <Brain />
          </div>
          <span className="eyebrow">NO FILES? NO PROBLEM.</span>
          <h2>
            Explore a<br />
            <em>synthetic study.</em>
          </h2>
          <p>
            A coordinated teaching fixture, designed to make the workflow
            tangible. No patient data. No real inference.
          </p>
          <button className="text-link" onClick={loadSample}>
            Load SAMPLE-001 <Arrow />
          </button>
          <div className="sample-specs mono">
            160 × 192 × 160 VOXELS
            <br />1 mm ISOTROPIC
            <br />
            FOUR SYNTHETIC SEQUENCES
          </div>
        </aside>
      </div>
      <footer className="viewer-footer">
        <span>VOXEL / ANALYSE WITH CONTEXT</span>
        <span>FILES STAY ON THIS DEVICE · SYNTHETIC PREDICTION ONLY</span>
      </footer>
    </div>
  )
}

// ------------------------------------------------------------------ viewer
type ViewerProps = {
  study: Study
  onNew: () => void
}
function Viewer({ study, onNew }: ViewerProps) {
  const [point, setPoint] = useState<Point>([95, 95, 90]),
    [sequence, setSequence] = useState(1),
    [enabled, setEnabled] = useState([true, true, true]),
    [opacity, setOpacity] = useState(0.7),
    [rotation, setRotation] = useState(0),
    [zoom, setZoom] = useState(1),
    [mobileAxis, setMobileAxis] = useState(2)
  const report = () => {
    const data = {
      study: "SAMPLE-001",
      synthetic: true,
      model: "Demo segmentation model v0.1 — simulated",
      regions,
      settings: {
        point,
        sequence: sequences[sequence],
        enabled,
        opacity,
        rotation,
        zoom,
      },
      coordinates:
        "Illustration origin: grid corner; 1 mm isotropic, not clinical registration",
      selectedInputs: {
        unrelatedToSyntheticPrediction: !study.sample,
        files: study.inputs,
      },
      runtime: {
        seconds: study.elapsed,
        type: "Prototype elapsed time, not inference runtime",
      },
      evaluation: {
        dice: 0.89,
        sensitivity: 0.91,
        label: "Illustrative sample evaluation — not validated performance",
      },
      disclaimer,
    }
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }),
    )
    const a = document.createElement("a")
    a.href = url
    a.download = "VOXEL-SAMPLE-001.json"
    a.click()
    setTimeout(() => URL.revokeObjectURL(url), 1000)
  }
  return (
    <div className="viewer">
      <header className="viewer-header">
        <Brand />
        <a href="#/" className="back-link">
          ← Back to overview
        </a>
        <span className="mono">RESEARCH WORKSPACE / V.01</span>
      </header>
      <div className="viewer-safety">
        <span>+</span>
        {disclaimer}
        <span className="sample-badge">SYNTHETIC DATA</span>
      </div>
      <div className="workspace-title">
        <div>
          <span className="eyebrow">INTERACTIVE IMAGING</span>
          <h1>A closer perspective.</h1>
        </div>
        <div className="workspace-actions">
          <button className="button outline" onClick={onNew}>
            New study
          </button>
          <button className="button primary" onClick={report}>
            Export sample report ↓
          </button>
        </div>
      </div>
      <>
        <div className="study-bar">
          <span>
            <i className="status-dot" /> SAMPLE-001{" "}
            <b>Sample prediction complete</b>
          </span>
          <span>1 mm ISOTROPIC · {sequences[sequence]} · SYNTHETIC</span>
        </div>
        <div className="viewer-grid">
          <div className="imaging-main">
            <section className="brain-panel">
              <div className="panel-heading">
                <span>01 / SPATIAL VIEW</span>
                <span>DRAG TO ROTATE</span>
              </div>
              <Brain
                rotation={rotation}
                zoom={zoom}
                enabled={enabled}
                opacity={opacity}
                onRotate={setRotation}
              />
              <div className="brain-panel-label mono">
                SIMULATED 3D ANATOMY
                <br />
                NOT AN MRI RECONSTRUCTION
              </div>
              <div className="spatial-controls">
                <button
                  onClick={() => setRotation(rotation - 0.2)}
                  aria-label="Rotate brain left"
                >
                  ↶
                </button>
                <button
                  onClick={() => setRotation(rotation + 0.2)}
                  aria-label="Rotate brain right"
                >
                  ↷
                </button>
                <label>
                  ZOOM{" "}
                  <input
                    type="range"
                    min=".65"
                    max="1.3"
                    step=".05"
                    value={zoom}
                    onChange={(e) => setZoom(+e.target.value)}
                  />
                </label>
                <button
                  onClick={() => {
                    setRotation(0)
                    setZoom(1)
                  }}
                >
                  Reset
                </button>
              </div>
            </section>
            <div className="slice-toolbar">
              <span className="mono">02 / LINKED SLICES</span>
              <div>
                {sequences.map((s, i) => (
                  <button
                    key={s}
                    aria-pressed={sequence === i}
                    className={sequence === i ? "active" : ""}
                    onClick={() => setSequence(i)}
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>
            <div className="mobile-slice-tabs">
              {[2, 1, 0].map((a) => (
                <button
                  key={a}
                  className={mobileAxis === a ? "active" : ""}
                  onClick={() => setMobileAxis(a)}
                >
                  {["Sagittal", "Coronal", "Axial"][a]}
                </button>
              ))}
            </div>
            <div className="slices-grid">
              {[2, 1, 0].map((a) => (
                <div
                  className={`slice-panel ${
                    mobileAxis === a ? "mobile-active" : ""
                  }`}
                  key={a}
                >
                  <div className="panel-heading">
                    <span>{["SAGITTAL", "CORONAL", "AXIAL"][a]}</span>
                    <span>{point[a].toString().padStart(3, "0")}</span>
                  </div>
                  <div className="slice-wrap">
                    <Slice
                      axis={a}
                      point={point}
                      sequence={sequence}
                      enabled={enabled}
                      opacity={opacity}
                      onPoint={setPoint}
                    />
                    <span className="orient top">{a === 2 ? "A" : "S"}</span>
                    <span className="orient left">{a === 0 ? "P" : "R"}</span>
                    <span className="orient right">{a === 0 ? "A" : "L"}</span>
                    <span className="orient bottom">{a === 2 ? "P" : "I"}</span>
                  </div>
                  <input
                    aria-label={`${["Sagittal", "Coronal", "Axial"][a]} slice`}
                    type="range"
                    min="0"
                    max={a === 1 ? 191 : 159}
                    value={point[a]}
                    onChange={(e) =>
                      setPoint(
                        point.map((v, i) =>
                          i === a ? +e.target.value : v,
                        ) as Point,
                      )
                    }
                  />
                </div>
              ))}
            </div>
            <div className="coordinate-bar">
              <span>SHARED CROSSHAIR</span>
              {point.map((p, i) => (
                <label key={i}>
                  {["X", "Y", "Z"][i]}
                  <input
                    type="number"
                    min="0"
                    max={i === 1 ? 191 : 159}
                    value={p}
                    onChange={(e) =>
                      setPoint(
                        point.map((v, j) =>
                          i === j
                            ? Math.max(
                                0,
                                Math.min(
                                  i === 1 ? 191 : 159,
                                  Math.round(+e.target.value),
                                ),
                              )
                            : v,
                        ) as Point,
                      )
                    }
                  />
                </label>
              ))}
              <span>VOXEL = mm*</span>
            </div>
            <p className="coordinate-note">
              *Illustration origin at grid corner. Not clinical scanner
              coordinates.
            </p>
          </div>
          <aside className="viewer-rail">
            <section>
              <div className="panel-heading">REGIONS & VOLUME ESTIMATES</div>
              {regions.map((r, i) => (
                <label className="region-control" key={r.id}>
                  <input
                    type="checkbox"
                    checked={enabled[i]}
                    onChange={() =>
                      setEnabled(enabled.map((v, j) => (i === j ? !v : v)))
                    }
                  />
                  <span
                    className="region-dot"
                    style={{ background: r.color }}
                  />
                  <span>
                    <b>{r.id}</b>
                    <small>{r.name}</small>
                  </span>
                  <strong>
                    {r.volume}
                    <small> mL</small>
                  </strong>
                </label>
              ))}
              <label className="opacity-control">
                Overlay opacity <span>{Math.round(opacity * 100)}%</span>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step=".05"
                  value={opacity}
                  onChange={(e) => setOpacity(+e.target.value)}
                />
              </label>
              <p className="rail-note">
                Nested regions: ET ⊆ TC ⊆ WT.
                <br />
                Synthetic sample estimates, not clinical measurements.
              </p>
            </section>
            <section>
              <div className="panel-heading">ILLUSTRATIVE EVALUATION</div>
              <div className="evaluation">
                <div>
                  <span>Dice</span>
                  <strong>0.89</strong>
                </div>
                <div>
                  <span>Sensitivity</span>
                  <strong>0.91</strong>
                </div>
              </div>
              <p className="rail-note">
                Held-out labelled-case evaluation example only. Not validated
                performance. No trained model is running.
              </p>
            </section>
            <section>
              <details open>
                <summary>MODEL & PROVENANCE</summary>
                <dl>
                  <dt>MODEL</dt>
                  <dd>Demo segmentation model v0.1 — simulated</dd>
                  <dt>PREPROCESSING (ILLUSTRATIVE)</dt>
                  <dd>Alignment · resampling · normalization</dd>
                  <dt>ELAPSED PROTOTYPE TIME</dt>
                  <dd>{study.elapsed.toFixed(2)} s — not inference runtime</dd>
                  <dt>INPUT IDENTIFIERS</dt>
                  {study.inputs.map((f) => (
                    <dd className="hash" key={f.label}>
                      {f.label}: {f.hash}
                    </dd>
                  ))}
                </dl>
                <p className="rail-note">
                  Selected files are unrelated to this synthetic prediction. No
                  file data leaves this browser.
                </p>
              </details>
            </section>
          </aside>
        </div>
      </>
      <footer className="viewer-footer">
        <span>VOXEL / EXPLORE WITH CONTEXT</span>
        <span>PUBLIC-DATA RESEARCH INTENT · SYNTHETIC FIXTURE ONLY</span>
      </footer>
    </div>
  )
}
export default App
