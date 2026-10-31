# Ichkeul Habitat Time Machine: design system

Global source of truth for the website (`web/`). Page-specific overrides go in `design-system/ichkeul/pages/`.

## 1. Brief and audience

| | |
|---|---|
| Product | Single-page research feature with interactive maps, presenting a remote-sensing study |
| Audience | Earth-observation and ecology researchers, park managers, internship reviewers, the interested public |
| Reading mode | Long-form, figure-led; readers scan findings first, then inspect figures and methods |
| Tone | Scientific journal / atlas, not a product landing page |
| Stack | Vite + TypeScript, MapLibre GL 6, Apache ECharts 6 (static, GitHub Pages) |

## 2. How the direction was chosen

The direction comes from the UI/UX Pro Max design database, queried per the skill's contract (one intent per query, verify the fit, retry once).

| Query | Result | Decision |
|---|---|---|
| `--design-system` "scientific research environmental data storytelling" | Enterprise Gateway pattern, red/blue bio palette, Exo | Rejected: a sales funnel pattern does not fit a research study |
| `--design-system` retry, "environmental monitoring geospatial data" | Operations dashboard, OLED dark, Orbitron/JetBrains Mono | Rejected: cyberpunk/HUD tone contradicts a scholarly piece |
| `--domain style` "editorial print magazine" | **Editorial Grid / Magazine** (#1) | Adopted: print-inspired hierarchy, figures, column layout |
| `--domain typography` "editorial academic research serif" | **Academic/Research: Crimson Pro + Atkinson Hyperlegible** (#1) | Adopted, with Atkinson Hyperlegible *Next* and IBM Plex Mono for data |
| `--domain color` "nature earth environmental scientific" | **Agriculture/Farm Tech: #15803D / #A16207 / #14532D** | Adapted: earth green + harvest gold, moved onto warm paper |
| `--domain chart` "time series trend comparison" | Line chart; *never distinguish series by hue alone* | Adopted: markers + dash styles + hatching everywhere |

Neither full design system was used, because both failed the fit check; the system above is assembled from verified domain matches only.

### Why a light "paper" theme replaced the dark theme

- Dark, glowing UIs with teal accents read as a generic tech/AI aesthetic. Paper, hairline rules and serif display type read as research and publishing, which is what this project is.
- Satellite maps and habitat colours keep their meaning on a neutral warm ground; on near-black, mid-tone classes (sand, olive) lose separation.
- Long captions and methods text are easier to read as dark text on light paper.

## 3. Colour

### 3.1 Interface tokens (WCAG contrast measured)

| Token | Hex | Role | On paper `#F5F1E8` | On band `#ECE6D8` |
|---|---|---|---|---|
| `--paper` | `#F5F1E8` | page background (salt-flat paper) | – | – |
| `--paper-band` | `#ECE6D8` | alternate section band | – | – |
| `--surface` | `#FBF9F4` | figure panels | – | – |
| `--ink` | `#16241F` | body and headings | 14.26 : 1 | 12.92 : 1 |
| `--ink-2` | `#36443E` | secondary text | 9.07 : 1 | 8.22 : 1 |
| `--muted` | `#56635C` | captions, metadata | 5.59 : 1 | 5.06 : 1 |
| `--reed` | `#166534` | primary accent, links, wins | 6.33 : 1 | 5.73 : 1 |
| `--gold` | `#8A5A06` | section numbers, figure labels | 5.25 : 1 | 4.76 : 1 |
| `--lagoon` | `#1D4E89` | water data, focus ring | 7.44 : 1 | 6.74 : 1 |
| `--alert` | `#9A3324` | losses, negative trends | 6.48 : 1 | 5.87 : 1 |

Every text token passes WCAG AA (4.5 : 1) on every background it is used on. White on `--ink` is 16.1 : 1 (footer).

### 3.2 Habitat classes (cartographic logic)

| Class | Hex | Why this colour |
|---|---|---|
| Permanent open water | `#1D4E89` | deep water blue |
| Seasonal shallow water | `#7FB3D5` | light water blue: same hue family, "less" water |
| Freshwater emergent marsh | `#3C8D5A` | reed green |
| Halophytic marsh | `#B5523B` | *Salicornia* turns red-brown in autumn |
| Bare saline flat / mudflat | `#E6D5A8` | salt-crust sand |
| Cropland & grassland | `#B9B46C` | dry olive farmland |
| Shrub & woodland | `#1F4A2C` | dark forest green (Djebel Ichkeul) |
| Built & roads | `#5E5A57` | neutral graphite |

Colour-vision check (CIELAB ΔE between the closest pair, Machado simulation):

| Vision | Minimum ΔE | Closest pair |
|---|---|---|
| Typical | 20.1 | saline flat / cropland |
| Deuteranopia | 15.0 | woodland / built |
| Protanopia | 12.7 | freshwater / halophytic marsh |
| Tritanopia | 12.9 | permanent water / woodland |

All pairs stay above ΔE 12 (clearly distinguishable). The map legend also lists names and areas, so no class depends on colour alone.

## 4. Typography

| Role | Font | Notes |
|---|---|---|
| Display, headings, big numbers | Crimson Pro (variable, 300–900, italic) | scholarly serif; italic for the counter-claim in the headline |
| Body, UI | Atkinson Hyperlegible Next (variable) | designed for low-vision legibility; its slashed zero is kept deliberately (0 vs O in data) |
| Data, labels, metadata | IBM Plex Mono 400/500 | axis labels, table numbers, figure labels, kickers |

All fonts are self-hosted through `@fontsource` (no third-party font requests). Body text is 17 px (16 px on phones) at a 1.6 line height. Tables use tabular lining figures.

## 5. Layout and components

- **Editorial grid**: 1200 px max width; section header = large gold section number on a 3 px ink rule + kicker + serif title + lead paragraph.
- **Hero**: headline, standfirst, byline strip and a numbered *Key findings* list beside a framed Figure 1 (animated map + caption + legend).
- **Figures**: every chart, map and table is a `<figure>` numbered automatically with CSS counters (`FIG. n`, `TABLE n`), with a caption and a `Source:` line where data is external.
- **Tables**: booktabs style (2 px top/bottom rules, 1 px header rule, no zebra), numbers right-aligned in mono, best row tinted reed.
- **Controls**: outlined segmented buttons, active = ink fill; all controls ≥ 44 px high.
- **Callouts**: "Interpretation" boxes with a gold left rule, used for cautious reading of results.
- **Footer**: journal-style *How to cite*, *Data availability*, *Code availability*, *References*.

## 6. Charts

- Paper tooltips with an ink border; dashed light grid; ink axis lines with ticks; mono axis labels.
- Series are told apart by **colour + marker + dash**: wet vs dry = circle/solid vs diamond/dashed; Landsat vs Sentinel-2 = dashed grey/open circle vs solid blue; each model has its own symbol and dash style.
- Emphasis is never colour alone: below-average rainfall years and published-interview bird figures are **hatched** as well as coloured.
- Heatmap labels switch to paper colour on dark cells to keep 4.5 : 1.

## 7. Interaction and accessibility

- Skip link, visible 3 px focus ring (`--focus`), `aria-pressed` on every toggle, `role="img"` + labels on charts and maps.
- The comparison divider is a keyboard slider (←/→, Shift for big steps, Home/End) with `aria-valuenow`.
- Legend classes are keyboard-toggleable buttons.
- SVG icons only (play/pause, divider chevrons); no emoji.
- `prefers-reduced-motion`: no reveal animation, no smooth scroll, the hero map shows a static last year.
- Checked at 375, 768, 1024 and 1440 px: no horizontal scroll, no control under 44 px.

## 8. Anti-patterns to avoid on this project

- Dark neon / glow / gradient cards, glassmorphism, emoji icons.
- Hue-only series encoding; rainbow ramps for ordered data.
- Rounded "SaaS" cards with large radii; keep 3 px corners and hairlines.
- Headline numbers without their test, p-value or source nearby.
