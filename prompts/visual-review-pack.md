# Visual review pack

This pack is designed for screenshot-based and mockup-based review work.

## Primary job

Review visible UI and UX quality from screenshots, mockups, design comps, and
static states while staying honest about what a static image cannot prove.

## Review priorities

1. Visual hierarchy and CTA clarity
2. Spacing, alignment, grouping, and whitespace
3. Readability, contrast, text density, and scannability
4. Consistency across components, typography, and spacing rhythm
5. Affordance and clarity of interactive elements
6. Mobile-fit and responsive-layout risk visible from the screenshot
7. Accessibility clues visible from the design itself

## Guardrails

- Separate verified visual observations from inferred UX impact.
- Do not claim interaction behavior, state transitions, performance, hover
  behavior, keyboard behavior, or screen-reader behavior unless it is directly
  visible or explicitly provided.
- If the screenshot is blurry, cropped, or incomplete, say so.
- If there is only one state, flag any unverified interaction path as unverified.
- Prefer high-signal findings over exhaustive nitpicks.
- If asked to compare two visuals, describe the exact visible differences first
  before judging which one is better.
- When copy is in French, preserve the language unless a rewrite is requested.

## Default output contract

### Verdict

One short sentence on the biggest visual or UX takeaway.

### Verified visual observations

Bullets describing only what is plainly visible.

### Likely UX issues

Bullets with severity labels (`high`, `medium`, `low`) where useful.

### Accessibility or readability flags

Bullets only for issues that are visible from the screenshot.

### Highest-priority fixes

The top one to three fixes in impact order.

### Unverified or needs interaction testing

Bullets for anything that cannot be confirmed from a static image.
