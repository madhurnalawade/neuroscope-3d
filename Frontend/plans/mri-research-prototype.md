# MRI research prototype — implementation plan

## Goal and agreed scope

Build a polished, responsive React prototype of an editorial MRI research site and linked imaging workspace, following the attached “Figma Prompt.txt.”

Confirmed choices:
- **High-fidelity interactive prototype**, not real NIfTI visualization or machine-learning inference.
- **Editorial vermilion** direction: warm bone, graphite, restrained vermilion, square geometry, thin rules, generous whitespace.

Success means a convincing, complete journey from the landing page through sample setup and simulated processing into a usable viewer, with honest safety language and working controls. No backend, patient data, clinical claims, or external upload service.

## Repository grounding

- Clean React 19 / Vite 8 / TypeScript scaffold; `src/App.tsx` currently contains only an empty shell.
- Tailwind v4 is imported in `src/index.css`; global fonts belong there.
- No existing design system, icon package, router, imaging library, or assets are declared.
- Preserve the existing toolchain and supervised development server. Do not start another server.
- Available scripts: `build` (`vite build`) and `format` (`oxfmt`); no test script exists.

## Visual system

- Working product title: **VOXEL / Research imaging**, explicitly presented as a prototype rather than an institution.
- Define semantic CSS variables for paper, ink, muted text, rules, workspace, signal, and region colors.
- Typography: Google Fonts **Bodoni Moda** for expressive display, **IBM Plex Sans** for body/UI, and **IBM Plex Mono** for technical readouts; provide appropriate serif/sans/monospace fallbacks.
- Build small reusable local primitives for buttons, labels, controls, section headings, and status messages. Avoid a generic card dashboard or a large UI framework.
- Use vermilion for primary actions and the selected segmentation, with restrained amber and pale teal for distinct nested region overlays. Always pair colors with names, outlines, or legends.
- Use custom small inline SVG symbols with accessible text, not emojis.
- Anatomy is a purpose-built scientific illustration, not stock photography: layered SVG/canvas geometry with sulci, directional shading, technical ticks, and segmented regions.

## Information architecture and navigation

- Two application views, driven by URL hash: `#/` landing and `#/viewer` workspace.
- Landing CTA “Explore the viewer” opens setup in the viewer; “How it works” scrolls to the MRI sequence explanation.
- Viewer includes a clear route back to the landing page and a direct “Load sample study” path.
- Keep the active study in React memory while navigating between these views. Reloading starts clean; no local storage of files or metadata.
- Footer/navigation links target actual sections rather than placeholder destinations.

## Landing page

1. **Header:** compact wordmark, section links, and viewer action; mobile uses a compact, accessible navigation toggle.
2. **Hero:** exact headline “See a brain tumour in 3D. Measure it. Know its limits.” and supplied supporting copy; two required actions; visible “Research and teaching prototype only — not for diagnosis.”
3. **Brain centerpiece:** large, carefully shaded pseudo-3D anatomy, region labels, orientation marks, and scan coordinates. A sticky visual stage responds to scroll with subtle rotation, scale, and lateral movement, yielding space to the next section. Use requestAnimationFrame and bounded progress; reduced-motion mode shows a stable composition without scroll transforms.
4. **The scan, opened up:** four coordinated sequence thumbnails for T1, T1Gd, T2, and FLAIR; selection updates the larger explanation/preview.
5. **From voxels to regions:** nested diagram explaining WT (whole tumour), TC (tumour core), and ET (enhancing tumour), explicitly showing ET ⊆ TC ⊆ WT.
6. **Inspect the result:** linked-view preview with an actionable path to load the sample workspace.
7. **Measure, with context:** sample volume display and a clearly marked illustrative evaluation panel, never presented as verified model performance.
8. **Research boundaries:** public de-identified research-data intent, prototype limitations, no diagnosis or treatment guidance, and a final viewer CTA.

## Sample study and illustration model

- Use a deterministic synthetic teaching fixture, named `SAMPLE-001`, with no real patient information and no claim that it comes from a specific public dataset.
- Shared grid: 160 × 192 × 160 voxels; 1 mm isotropic spacing. Coordinates use an explicitly documented illustration origin, not clinical scanner/world registration.
- Store region definitions, sample volumes, display colors, and illustrative evaluation values in a typed fixture module.
- Generate coordinated procedural anatomy and nested masks from the same normalized 3D coordinate model. Canvas slice rendering samples this model for the selected orientation, sequence, crosshair, and enabled overlays.
- Use downsampled surface points/layered projected contours for the rotatable 3D illustration, with anatomy clipping and depth shading. This is a simulated 3D rendering, not a reconstructed MRI mesh.
- Display rounded region volume estimates derived consistently from the fixture masks and voxel spacing; mark them as sample estimates.
- All metrics carry “Illustrative sample evaluation — not validated performance”; distinguish Dice/sensitivity from per-study volume.

## Setup and validation flow

- Four individually labelled input slots: T1, T1Gd, T2, FLAIR; accept `.nii` and `.nii.gz`, with file selection and drag-and-drop.
- Prominent explanation: “Prototype workflow. Selected files are not parsed or used for prediction. Use the synthetic sample to explore the viewer.”
- For selected files, perform only truthful client-side checks: required slots, supported filename extension, nonempty file, duplicate assignment, and a 100 MB per-file prototype limit.
- Do not claim to verify readability, anatomy, grid alignment, or sequence correctness from the filename.
- Provide a separate, clearly marked **validation demo** scenario selector: valid sample, missing sequence, unreadable sample, and mismatched sample geometry. This demonstrates required error states without pretending to inspect real NIfTI contents.
- “Load sample study” fills fixture metadata and enables “Run sample prediction.” A complete set of basic file checks also allows this action, but it still loads the same clearly labelled synthetic result.
- Processing progresses through simulated validation, preprocessing, and segmentation stages over a short timed sequence. Cancel/reset clears timers and returns to setup. Prevent duplicate runs.
- Finish with “Sample prediction complete”; announce status updates accessibly. Replacing inputs invalidates the current run/result.
- File objects remain local in memory and are never transmitted. File hashes are SHA-256 via Web Crypto when supported; if hashing fails, show “Unavailable,” never an invented hash. Sample hashes are labelled as fixture identifiers rather than hashes of uploaded content.

## Viewer layout and behavior

Desktop workspace:
- Compact study/status header and persistent safety strip.
- Main area: prominent 3D view plus three linked slice views (axial, coronal, sagittal).
- Aligned side rail for region controls, estimated volumes, performance, and expandable provenance.
- Consistent dark graphite surfaces and fine dividers, with readable text and strong control contrast.

Interactions:
- 3D drag-to-rotate, keyboard-accessible rotation controls, zoom range, reset view, independent WT/TC/ET visibility toggles, and overlay opacity control.
- Slice panes expose orientation labels, active-sequence selection, a slice range control, crosshair, and voxel/mm readouts.
- Clicking a slice updates the shared 3D crosshair coordinates; the other two slices and displayed coordinates update together. Coordinates clamp to grid bounds.
- Keyboard-accessible X/Y/Z controls offer an alternative to pointer-only crosshair placement.
- Sequence changes affect all slice panes; toggles and opacity apply consistently across views.
- Provenance shows “Demo segmentation model v0.1 — simulated,” illustrative preprocessing, sample/input identifiers or actual available file hashes, and measured elapsed prototype runtime explicitly not inference runtime.
- Performance panel identifies Dice and sensitivity as illustrative held-out-case evaluation examples, with a note that no validated model is running.
- “Export sample report” downloads a client-generated JSON report containing the synthetic study marker, region values, current viewer settings, provenance, illustrative metrics, and the full disclaimer. Exclude raw files and identify any selected-file metadata as unrelated to the synthetic prediction.
- Persistent exact disclaimer: “Research and teaching prototype only. Model-predicted regions are not a diagnosis.”

## Responsive and accessibility requirements

- Desktop: editorial split hero and dense, orderly workspace.
- Tablet: stack narrative content and move viewer details below the visual workspace.
- Mobile: retain a visible brain illustration, stack setup inputs, use one large slice pane with axial/coronal/sagittal tabs rather than shrinking all three; retain all controls and shared state.
- Accessible labels, semantic headings, visible keyboard focus, sufficient contrast, live status announcements, reduced-motion support, and sensible touch targets.
- No hover-only essential information. Canvas/SVG views have textual summaries and accessible control alternatives.
- Loading, error, empty, selected, disabled, and completed states must be intentionally designed.

## Implementation structure

- `src/App.tsx`: hash navigation, application state, and page composition.
- `src/index.css`: font imports, semantic tokens, shared visual primitives, layout, responsive styles, reduced-motion rules.
- `src/components/`: landing sections, reusable anatomy renderer, upload/setup flow, slice canvas, viewer controls and panels.
- `src/data/sampleStudy.ts`: typed synthetic fixture, region definitions, evaluation examples.
- `src/lib/`: coordinate transforms, procedural anatomy sampling, file checks/hashing, report serialization.
- Keep dependencies minimal; React, browser canvas/SVG, Web Crypto, and native downloads suffice. No imaging or 3D engine is required for this agreed prototype scope.

## Delivery sequence

1. Establish tokens, typography, local controls, fixture model, and anatomy renderer.
2. Implement landing composition and restrained scroll-linked brain behavior.
3. Add setup slots, honest validation, demonstration error scenarios, and cancellable processing.
4. Implement linked viewer state, rendering, controls, metrics/provenance, and export.
5. Complete mobile layouts, accessibility, and all interaction states.
6. Verify and fix issues within scope.

## Verification and acceptance

- Format changed source using the repository `format` script with targeted paths.
- Run the repository `build` script for this broader implementation; do not claim it passed unless it succeeds.
- Use the existing supervised preview for interaction checks; do not launch another development server.
- Confirm landing navigation, scroll behavior, reduced motion, sequence selection, sample loading, all demo validation scenarios, cancellation, completion, input replacement, region visibility/opacity, rotation/zoom/reset, linked slices, coordinate bounds, report contents, and navigation state retention.
- Check narrow mobile, tablet, and desktop layouts for readable anatomy, usable controls, and no unintended horizontal overflow.
- Verify keyboard access, focus visibility, status announcements, and exact disclaimers.
- Verify no network requests send selected file content, no real MRI-processing claim appears, and illustrative metrics cannot be mistaken for validated results.
- No test framework is currently installed; avoid adding one solely for this prototype. Use focused deterministic checks for coordinate mapping and validation helpers if browser tooling supports them; report any unverified checks honestly.

## Explicit exclusions

Real NIfTI parsing, clinical orientation/registration, real model inference, validated accuracy claims, true patient-derived 3D reconstruction, backend storage, authentication, DICOM/PACS integration, diagnostic reporting, treatment guidance, and production medical-device readiness.
