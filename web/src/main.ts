import "@fontsource-variable/inter";
import "@fontsource-variable/fraunces";
import "./style.css";
import { C, axis, chart, legend, tooltip } from "./charts";
import { MODEL_COLOR, MODEL_LABEL, dataUrl, fmt, loadSite, pval, signed, type Season, type Site } from "./data";
import { Swipe, addBasin, addLabels, makeMap, onReady, preload, setImage } from "./maps";

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
const SEASON_LABEL: Record<Season, string> = { wet: "Wet season · Jan–Mar", dry: "Dry season · Jul–Sep" };
const WETLAND_KEYS = ["perm_water", "seas_water", "fresh_marsh", "halo_marsh", "saline_flat"];
const PRETTY: Record<string, string> = {
  open_water: "Open water (all)", marsh: "Marsh (all)", perm_water: "Permanent open water", seas_water: "Seasonal shallow water",
  fresh_marsh: "Freshwater marsh", halo_marsh: "Halophytic marsh", saline_flat: "Saline flat / mudflat",
  crop_grass: "Cropland & grassland", shrub_wood: "Shrub & woodland", built: "Built & roads",
};

function segmented(id: string, onChange: (v: string) => void) {
  const el = $(id);
  el.querySelectorAll("button").forEach((b) =>
    b.addEventListener("click", () => {
      el.querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
      onChange(b.dataset.v!);
    }));
}

/** Habitat PNG recoloured client-side so individual classes can be hidden. */
const classCanvasCache = new Map<string, ImageData>();
async function maskedMapUrl(site: Site, url: string, hidden: Set<number>): Promise<string> {
  if (!hidden.size) return url;
  let data = classCanvasCache.get(url);
  const cv = document.createElement("canvas");
  if (!data) {
    const im = new Image();
    im.src = dataUrl(url);
    await im.decode();
    cv.width = im.width; cv.height = im.height;
    const ctx = cv.getContext("2d")!;
    ctx.drawImage(im, 0, 0);
    data = ctx.getImageData(0, 0, im.width, im.height);
    classCanvasCache.set(url, data);
  }
  const out = new ImageData(new Uint8ClampedArray(data.data), data.width, data.height);
  const cols = site.classes.filter((c) => hidden.has(c.id)).map((c) => [1, 3, 5].map((i) => parseInt(c.color.slice(i, i + 2), 16)));
  const d = out.data;
  for (let i = 0; i < d.length; i += 4) {
    if (d[i + 3] === 0) continue;
    for (const [r, g, b] of cols) if (d[i] === r && d[i + 1] === g && d[i + 2] === b) { d[i + 3] = 0; break; }
  }
  cv.width = out.width; cv.height = out.height;
  cv.getContext("2d")!.putImageData(out, 0, 0);
  return cv.toDataURL("image/png");
}

// ------------------------------------------------------------------ hero
function hero(site: Site) {
  const map = makeMap("hero-map", site, { interactive: false, attributionControl: { compact: true } });
  const wet = site.seasons.filter((s) => s.season === "wet");
  preload(wet.map((s) => s.map));
  $("hero-stats").innerHTML = site.headline.slice(0, 4)
    .map((h) => `<div class="hero-stat"><span>${h.label}</span><b class="num">${h.value}</b><span>${h.sub}</span></div>`).join("");
  $("hero-legend").innerHTML = `<div class="legend">${site.classes.map((c) =>
    `<label><span class="sw" style="background:${c.color}"></span>${c.name}</label>`).join("")}</div>`;
  onReady(map, () => {
    const [w, s, e, n] = site.meta.bounds;
    map.fitBounds([[w, s], [e, n]], { padding: { top: 60, bottom: 40, left: innerWidth > 900 ? innerWidth * 0.3 : 10, right: 10 }, duration: 0 });
    setImage(map, "h0", wet[0].map, site, 0.82);
    setImage(map, "h1", wet[0].map, site, 0);
    for (const id of ["h0", "h1"]) map.setPaintProperty(id, "raster-opacity-transition", { duration: 900, delay: 0 });
    addLabels(map);
    let k = 0, front = 0;
    const show = (i: number) => {
      const s = wet[i];
      const back = 1 - front;
      setImage(map, `h${back}`, s.map, site);
      setTimeout(() => {
        map.setPaintProperty(`h${back}`, "raster-opacity", 0.82);
        map.setPaintProperty(`h${front}`, "raster-opacity", 0);
        front = back;
      }, 120);
      $("hero-year").textContent = String(s.year);
    };
    if (reduceMotion) { show(wet.length - 1); return; }
    const tick = () => {
      k = (k + 1) % wet.length;
      show(k);
      setTimeout(tick, k === wet.length - 1 ? 3800 : 1700);
    };
    setTimeout(tick, 1800);
  });
}

// ------------------------------------------------------------------ context
function context(site: Site) {
  $("timeline").innerHTML = site.timeline.map((t) =>
    `<div class="tl-item ${t.bad ? "bad" : ""}"><div class="y">${t.year}</div><p>${t.text}</p></div>`).join("");
  chart("birds-chart").render({
    grid: { left: 150, right: 40, top: 10, bottom: 30 },
    tooltip: tooltip({ trigger: "item", formatter: (p: any) => {
      const b = site.birds[p.dataIndex];
      return `<b>${b.label}</b><br/>${b.kind === "upper bound" ? "fewer than " : "~"}${b.value.toLocaleString()} waterbirds<br/><span style="color:${C.muted}">${b.source}</span>`;
    } }),
    xAxis: { type: "value", ...axis({ axisLabel: { color: C.muted, formatter: (v: number) => `${v / 1000}k` } }) },
    yAxis: { type: "category", inverse: true, data: site.birds.map((b) => b.label), ...axis({ splitLine: { show: false } }) },
    series: [{
      type: "bar", barWidth: 14, data: site.birds.map((b, i) => ({
        value: b.value, itemStyle: { color: i === 2 ? C.danger : i >= 3 ? "#56b4e9" : C.accent, borderRadius: 4 } })),
      label: { show: true, position: "right", color: C.text, fontSize: 11,
        formatter: (p: any) => (site.birds[p.dataIndex].kind === "upper bound" ? "< " : "") + `${Math.round(p.value / 1000)}k` },
    }],
  });
}

// ------------------------------------------------------------------ time machine
function timeMachine(site: Site) {
  const a = makeMap("tm-a", site);
  const b = makeMap("tm-b", site);
  const swipe = new Swipe(a, b, $("tm-box"), $("tm-handle"));
  void swipe;
  const years = site.meta.years;
  const st = { mode: "swipe", season: "wet" as Season, year: years[years.length - 1], yearB: years[0], opacity: 0.9, hidden: new Set<number>() };
  const yb = $<HTMLSelectElement>("tm-yearB");
  yb.innerHTML = years.map((y) => `<option value="${y}">Right: ${y}</option>`).join("");
  yb.value = String(st.yearB);
  const slider = $<HTMLInputElement>("tm-year");
  slider.min = String(years[0]); slider.max = String(years[years.length - 1]); slider.value = String(st.year);
  $("tm-ticks").innerHTML = years.map((y) => `<span>${y}</span>`).join("");
  const entry = (y: number, s: Season) => site.seasons.find((e) => e.year === y && e.season === s)!;
  let ready = 0;
  const legendEl = $("tm-legend");

  async function update() {
    if (ready < 2) return;
    const e = entry(st.year, st.season);
    const eb = entry(st.yearB, st.season);
    const mapA = await maskedMapUrl(site, e.map, st.hidden);
    if (st.mode === "swipe") {
      setImage(a, "s2", e.s2, site, 1, "labels");
      setImage(a, "hab", mapA, site, 0, "labels");
      setImage(b, "s2", e.s2, site, 1, "labels");
      setImage(b, "hab", mapA, site, st.opacity, "labels");
      $("tm-tag-a").textContent = `Sentinel-2 · ${st.year} ${st.season}`;
      $("tm-tag-b").textContent = `Habitat map · ${st.year} ${st.season}`;
    } else {
      const mapB = await maskedMapUrl(site, eb.map, st.hidden);
      setImage(a, "s2", e.s2, site, 0.55, "labels");
      setImage(a, "hab", mapA, site, st.opacity, "labels");
      setImage(b, "s2", eb.s2, site, 0.55, "labels");
      setImage(b, "hab", mapB, site, st.opacity, "labels");
      $("tm-tag-a").textContent = `Habitat · ${st.year} ${st.season}`;
      $("tm-tag-b").textContent = `Habitat · ${st.yearB} ${st.season}`;
    }
    for (const m of [a, b]) m.setPaintProperty("hab", "raster-opacity", m === a && st.mode === "swipe" ? 0 : st.opacity);
    $("tm-year-label").textContent = String(st.year);
    // legend with areas
    const row = site.areas.find((r) => r.year === st.year && r.season === st.season)!;
    const max = Math.max(...site.classes.map((c) => Number(row[c.key])));
    legendEl.innerHTML = site.classes.map((c) => {
      const v = Number(row[c.key]);
      return `<label data-id="${c.id}" style="opacity:${st.hidden.has(c.id) ? 0.35 : 1}"><span class="sw" style="background:${c.color}"></span>
        <span style="flex:1">${c.name}<div class="bar"><i style="width:${(100 * v) / max}%;background:${c.color}"></i></div></span>
        <span class="area">${fmt(v, 1)} km²</span></label>`;
    }).join("");
    legendEl.querySelectorAll("label").forEach((l) => l.addEventListener("click", () => {
      const id = Number((l as HTMLElement).dataset.id);
      st.hidden.has(id) ? st.hidden.delete(id) : st.hidden.add(id);
      update();
    }));
    const ow = Number(row.perm_water) + Number(row.seas_water);
    $("tm-meta").innerHTML = `<b>${st.year} · ${SEASON_LABEL[st.season]}</b>
      <p class="muted" style="margin:6px 0 0">Open water <b style="color:${C.text}">${fmt(ow, 1)} km²</b> ·
      ${e.scenes} scenes · ${fmt(e.obs, 1)} clear observations per pixel on average.</p>
      <p class="muted small" style="margin:6px 0 0">Scenes: ${e.dates.map((d) => `${d.slice(6)}/${d.slice(4, 6)}`).join(", ")}</p>
      <p class="muted small" style="margin:6px 0 0">Map model: ${site.meta.map_model_label}</p>`;
    // warm the cache for neighbours
    preload(site.seasons.filter((s) => Math.abs(s.year - st.year) <= 1 && s.season === st.season).flatMap((s) => [s.s2, s.map]));
  }
  for (const m of [a, b]) onReady(m, () => { addLabels(m); addBasin(m, site); ready++; update(); });
  segmented("tm-mode", (v) => { st.mode = v; yb.hidden = v !== "years"; update(); });
  segmented("tm-season", (v) => { st.season = v as Season; update(); });
  slider.addEventListener("input", () => { st.year = Number(slider.value); update(); });
  yb.addEventListener("change", () => { st.yearB = Number(yb.value); update(); });
  $<HTMLInputElement>("tm-opacity").addEventListener("input", (e) => { st.opacity = Number((e.target as HTMLInputElement).value); update(); });
  let timer = 0;
  $("tm-play").addEventListener("click", () => {
    if (timer) { clearInterval(timer); timer = 0; $("tm-play").textContent = "▶"; return; }
    $("tm-play").textContent = "❚❚";
    timer = window.setInterval(() => {
      const i = years.indexOf(st.year);
      st.year = years[(i + 1) % years.length];
      slider.value = String(st.year);
      update();
    }, 1400);
  });
}

// ------------------------------------------------------------------ breathing lake
function breathing(site: Site) {
  const map = makeMap("br-map", site);
  let cur = "inundation_all";
  const render = () => {
    const L = site.layers[cur];
    setImage(map, "layer", L.url, site, 1, "labels");
    $("br-legend").innerHTML = `<h3 style="font-size:1rem">${L.title}</h3>
      <div class="ramp" style="background:linear-gradient(90deg,${L.ramp!.join(",")})"></div>
      <div class="ramp-labels">${L.labels!.map((l) => `<span>${l}</span>`).join("")}</div>`;
    const inn = site.inundation;
    const pt = site.pixel_trends;
    const info: Record<string, string> = {
      inundation_all: `<b>${fmt(inn.always_water_km2, 1)} km²</b> were water in all 20 seasons and <b>${fmt(inn.ever_water_km2, 1)} km²</b> at least once.`,
      inundation_wet: `<b>${fmt(inn.wet_only_km2, 1)} km²</b> are flooded in most winters but dry in most summers: the seasonal pulse.`,
      inundation_dry: `In summer the lake shrinks back to its core. The seasonal pulse is ${fmt(inn.breathing_ratio * 100, 0)} % of the summer water area.`,
      sav: `Candidate submerged-vegetation patches: lake pixels whose winter NDAVI exceeds that winter's lake median by ${fmt(site.sav.ndavi_threshold, 2)}. Only <b>${fmt((site.sav as any).persistent_km2, 1)} km²</b> recur in at least half of the winters. Exploratory, not calibrated with field data.`,
      trend_mndwi_wet: `Pixels with a significant (|z| > 1.96) 2017–2026 trend in wet-season MNDWI: <b>${fmt(pt.mndwi_wet.sig_decrease_km2, 1)} km²</b> getting drier, <b>${fmt(pt.mndwi_wet.sig_increase_km2, 1)} km²</b> wetter.`,
      trend_ndvi_dry: `Significant dry-season NDVI trend: <b>${fmt(pt.ndvi_dry.sig_decrease_km2, 1)} km²</b> browning, <b>${fmt(pt.ndvi_dry.sig_increase_km2, 1)} km²</b> greening.`,
    };
    $("br-stats").innerHTML = `<p class="muted" style="margin:0">${info[cur]}</p>`;
  };
  let ok = false;
  onReady(map, () => { addLabels(map); addBasin(map, site); ok = true; render(); });
  segmented("br-layer", (v) => { cur = v; if (ok) render(); });
}

// ------------------------------------------------------------------ change
function change(site: Site) {
  const areas = chart("areas-chart");
  const sankey = chart("sankey-chart");
  const draw = (season: Season) => {
    const rows = site.areas.filter((r) => r.season === season).sort((a, b) => Number(a.year) - Number(b.year));
    const years = rows.map((r) => String(r.year));
    const selected: Record<string, boolean> = {};
    site.classes.forEach((c) => (selected[c.name] = WETLAND_KEYS.includes(c.key)));
    areas.render({
      tooltip: tooltip({ trigger: "axis", valueFormatter: (v: number) => `${fmt(v, 1)} km²` }),
      legend: legend({ top: 0, selected }),
      grid: { left: 52, right: 20, top: 56, bottom: 36 },
      xAxis: { type: "category", data: years, boundaryGap: false, ...axis() },
      yAxis: { type: "value", name: "km²", ...axis() },
      series: site.classes.map((c, i) => ({
        name: c.name, type: "line", stack: "a", smooth: 0.25, symbol: "circle", symbolSize: 5,
        lineStyle: { width: 1, color: c.color }, itemStyle: { color: c.color }, areaStyle: { opacity: 0.85, color: c.color },
        emphasis: { focus: "series" }, data: rows.map((r) => Number(r[c.key])),
        ...(i === 0 ? { markLine: { symbol: "none", silent: true, lineStyle: { color: C.danger, type: "dashed" },
          label: { color: C.danger, formatter: "Jan 2020: < 6,000 waterbirds", fontSize: 11 },
          data: season === "wet" ? [{ xAxis: "2020" }] : [] } } : {}),
      })),
    });
    const t = site.trends[season];
    const keys = ["open_water", "marsh", ...site.classes.map((c) => c.key)];
    $("trend-table").innerHTML = `<tr><th>Class</th><th class="r">Sen's slope<br>km²/yr</th><th class="r">95 % CI</th><th class="r">τ</th><th class="r">p</th><th></th></tr>` +
      keys.map((k) => {
        const r = t[k];
        const cls = r.trend === "increasing" ? "up" : r.trend === "decreasing" ? "down" : "";
        return `<tr><td>${PRETTY[k] ?? k}</td><td class="r">${signed(r.sen_slope, 2)}</td>
          <td class="r small muted">${signed(r.slope_ci95[0], 2)} … ${signed(r.slope_ci95[1], 2)}</td>
          <td class="r">${fmt(r.tau, 2)}</td><td class="r">${pval(r.p)}</td>
          <td><span class="pill ${cls}">${r.trend === "no significant trend" ? "n.s." : r.trend}</span></td></tr>`;
      }).join("");
    const tr = site.transitions[season];
    const names = site.classes.map((c) => c.name);
    const yl = (ys: number[]) => `${ys[0]}–${String(ys[ys.length - 1]).slice(2)}`;
    const nodes = [...names.map((n, i) => ({ name: `${n} ·${yl(tr.from_years)}`, itemStyle: { color: site.classes[i].color } })),
      ...names.map((n, i) => ({ name: `${n} ·${yl(tr.to_years)}`, itemStyle: { color: site.classes[i].color } }))];
    const links: any[] = [];
    tr.matrix_km2.forEach((row, i) => row.forEach((v, j) => {
      if (i === j && i <= 1) return; // unchanged water would dwarf every marsh flow
      const wet = WETLAND_KEYS.includes(site.classes[i].key) || WETLAND_KEYS.includes(site.classes[j].key);
      if (v > 0.15 && wet) links.push({ source: nodes[i].name, target: nodes[8 + j].name, value: Number(v.toFixed(2)) });
    }));
    const used = new Set(links.flatMap((l) => [l.source, l.target]));
    $("sankey-sub").textContent = `Majority class ${yl(tr.from_years)} → ${yl(tr.to_years)}, ${season} seasons, flows > 0.15 km² touching a wetland class (unchanged water omitted).`;
    sankey.render({
      tooltip: tooltip({ trigger: "item", formatter: (p: any) => p.dataType === "edge"
        ? `${p.data.source.split(" ·")[0]} → ${p.data.target.split(" ·")[0]}<br/><b>${fmt(p.data.value, 2)} km²</b>` : p.name }),
      series: [{
        type: "sankey", left: 6, right: 120, top: 10, bottom: 10, nodeWidth: 12, nodeGap: 9, draggable: false,
        data: nodes.filter((n) => used.has(n.name)), links,
        label: { color: C.text, fontSize: 11, formatter: (p: any) => p.name.replace(" ·", "  ") },
        lineStyle: { color: "gradient", opacity: 0.38, curveness: 0.5 }, emphasis: { focus: "adjacency" },
      }],
    });
  };
  segmented("ch-season", (v) => draw(v as Season));
  draw("dry");
  const rob = (site as any).robustness as Record<string, any>;
  const mine = rob[site.meta.map_model];
  const pct = 100 * (mine.dry_marsh_last3 / mine.dry_marsh_first3 - 1);
  $("change-title").textContent = pct < 0 ? `Summer marshes lost ${fmt(-pct, 0)} % in a decade` : "What the lake and marshes gained and lost";
  const cell = (t: any) => `<td class="r">${signed(t.sen_slope, 2)} <span class="small muted">(p ${pval(t.p)})</span></td>`;
  $("robust-table").innerHTML = `<tr><th>Map model</th><th class="r">Summer marsh</th><th class="r">Summer open water</th><th class="r">Winter open water</th><th class="r">Summer cropland</th><th class="r">Woodland (stable?)</th><th class="r">Built (stable?)</th></tr>` +
    Object.entries(rob).map(([m, r]) => `<tr class="${m === site.meta.map_model ? "best" : ""}"><td>${MODEL_LABEL[m] ?? m}${m === site.meta.map_model ? " · used for maps" : ""}</td>
      ${cell(r.marsh.dry)}${cell(r.open_water.dry)}${cell(r.open_water.wet)}${cell(r.crop_grass.dry)}${cell(r.shrub_wood.dry)}${cell(r.built.dry)}</tr>`).join("") +
    `<tr><td colspan="7" class="small muted">Sen's slopes in km²/yr, Mann–Kendall p in brackets, 10 seasons each.</td></tr>`;
}

// ------------------------------------------------------------------ drivers
function drivers(site: Site) {
  const d = site.drivers;
  const table = d.table as any[];
  const ser = (s: Season, color: string) => ({
    name: s === "wet" ? "Wet seasons (rain Sep–Feb)" : "Dry seasons (rain Sep–Jun)", type: "scatter", symbolSize: 11,
    itemStyle: { color, borderColor: "#071014", borderWidth: 1 },
    label: { show: true, position: "top", color: C.muted, fontSize: 9, formatter: (p: any) => String(p.value[2]).slice(2) },
    data: table.filter((r) => r.season === s).map((r) => [r.rain_antecedent, r.open_water, r.year]),
  });
  const line = (s: Season, color: string) => {
    const p = d.per_season.open_water[s];
    const xs = table.filter((r) => r.season === s).map((r) => r.rain_antecedent);
    const x0 = Math.min(...xs), x1 = Math.max(...xs);
    return { name: `fit ${s}`, type: "line", symbol: "none", lineStyle: { color, type: "dashed", width: 1.5 }, tooltip: { show: false },
      data: [[x0, p.intercept + (p.slope_km2_per_100mm / 100) * x0], [x1, p.intercept + (p.slope_km2_per_100mm / 100) * x1]] };
  };
  chart("scatter-chart").render({
    tooltip: tooltip({ trigger: "item", formatter: (p: any) => p.value?.[2] ? `${p.value[2]}<br/>rain ${fmt(p.value[0], 0)} mm<br/>open water ${fmt(p.value[1], 1)} km²` : "" }),
    legend: legend({ top: 0, data: ["Wet seasons (rain Sep–Feb)", "Dry seasons (rain Sep–Jun)"] }),
    grid: { left: 52, right: 18, top: 40, bottom: 44 },
    xAxis: { type: "value", name: "antecedent rainfall (mm)", nameLocation: "middle", nameGap: 28, scale: true, ...axis() },
    yAxis: { type: "value", name: "open water km²", scale: true, ...axis() },
    series: [ser("wet", "#56b4e9"), ser("dry", C.warn), line("wet", "#56b4e9"), line("dry", C.warn)],
  });
  const rain = site.rain.filter((r) => r.rain_hydro_year != null);
  const mean = rain.reduce((s, r) => s + (r.rain_hydro_year ?? 0), 0) / rain.length;
  chart("rain-chart").render({
    tooltip: tooltip({ trigger: "axis", valueFormatter: (v: number) => `${fmt(v, 0)} mm` }),
    grid: { left: 52, right: 18, top: 24, bottom: 36 },
    xAxis: { type: "category", data: rain.map((r) => `${r.year - 1}/${String(r.year).slice(2)}`), ...axis() },
    yAxis: { type: "value", name: "mm (Sep–Aug)", ...axis() },
    series: [{ type: "bar", barWidth: "55%", data: rain.map((r) => ({ value: r.rain_hydro_year,
      itemStyle: { color: (r.rain_hydro_year ?? 0) < mean * 0.85 ? C.danger : "#56b4e9", borderRadius: [4, 4, 0, 0] } })),
      markLine: { symbol: "none", silent: true, lineStyle: { color: C.muted, type: "dashed" },
        label: { color: C.muted, formatter: `mean ${fmt(mean, 0)} mm`, fontSize: 11, position: "insideEndTop" }, data: [{ yAxis: mean }] } }],
  });
  const ols = d.ols.open_water;
  const ur = ols.unique_r2;
  const lab: Record<string, string> = { rain_antecedent: "Antecedent rain", rain_prev_hydro_year: "Previous year's rain", year: "Time trend (non-climatic)" };
  chart("r2-chart").render({
    tooltip: tooltip({ trigger: "item", valueFormatter: (v: number) => fmt(v, 3) }),
    grid: { left: 170, right: 40, top: 10, bottom: 30 },
    xAxis: { type: "value", max: 1, ...axis() },
    yAxis: { type: "category", inverse: true, data: ["Full model R²", ...Object.keys(ur).map((k) => lab[k])], ...axis({ splitLine: { show: false } }) },
    series: [{ type: "bar", barWidth: 14, data: [{ value: ols.r2, itemStyle: { color: C.accent } },
      ...Object.values(ur).map((v: any) => ({ value: v, itemStyle: { color: "#56b4e9" } }))],
      label: { show: true, position: "right", color: C.text, fontSize: 11, formatter: (p: any) => fmt(p.value, 2) } }],
  });
  const ws = d.per_season.open_water.wet, ds = d.per_season.open_water.dry;
  $("r2-text").innerHTML = `OLS on all 20 seasons with a season term: R² = ${fmt(ols.r2, 2)} (adjusted ${fmt(ols.adj_r2, 2)}). Bars show the R² lost when each factor is dropped. Within seasons, +100 mm of antecedent rain goes with ${signed(ws.slope_km2_per_100mm, 1)} km² of winter water (R² ${fmt(ws.r2, 2)}, p ${pval(ws.p)}) and ${signed(ds.slope_km2_per_100mm, 1)} km² of summer water (R² ${fmt(ds.r2, 2)}, p ${pval(ds.p)}).`;
  const g = d.gbm_shap;
  const flab: Record<string, string> = { is_wet: "Season (wet/dry)", rain_antecedent: "Antecedent rain", rain_prev_hydro_year: "Previous year's rain", year: "Year" };
  chart("shap-chart").render({
    tooltip: tooltip({ trigger: "item", valueFormatter: (v: number) => `${fmt(v, 2)} km²` }),
    grid: { left: 150, right: 40, top: 10, bottom: 30 },
    xAxis: { type: "value", name: "mean |SHAP| (km²)", nameLocation: "middle", nameGap: 22, ...axis() },
    yAxis: { type: "category", inverse: true, data: g.features.map((f: string) => flab[f]), ...axis({ splitLine: { show: false } }) },
    series: [{ type: "bar", barWidth: 14, data: g.mean_abs_shap, itemStyle: { color: C.warn, borderRadius: 4 },
      label: { show: true, position: "right", color: C.text, fontSize: 11, formatter: (p: any) => fmt(p.value, 1) } }],
  });
  $("shap-text").textContent = `LightGBM (depth-limited, 200 trees) on the same features; leave-one-year-out R² = ${fmt(g.loyo_r2, 2)}. With 20 points this ranks drivers, it does not forecast.`;
  const yr = ols.coef.year, yp = ols.p.year;
  $("drivers-note").innerHTML = `<b>Reading this carefully:</b> the time-trend coefficient is ${signed(yr, 2)} km² of open water per year after accounting for rainfall (p = ${pval(yp)}). ` +
    `A residual trend is consistent with growing upstream water capture, but it is not proof: no dam-release or storage series was available to test it directly.`;
}

// ------------------------------------------------------------------ forty years
function forty(site: Site) {
  const c = chart("landsat-chart");
  const ev = [
    { x: 1983.5, t: "Joumine + Ghezala dams" }, { x: 1994, t: "Sejnane dam" }, { x: 2016, t: "Off Montreux Record" },
  ];
  const draw = (s: Season) => {
    const rows = site.landsat.filter((r) => r.season === s && r.water_km2 != null);
    const ls = rows.filter((r) => r.source === "Landsat").map((r) => [r.year, r.water_km2]);
    const s2 = rows.filter((r) => r.source === "Sentinel-2").map((r) => [r.year, r.water_km2]);
    c.render({
      tooltip: tooltip({ trigger: "axis", valueFormatter: (v: number) => `${fmt(v, 1)} km²` }),
      legend: legend({ top: 0 }),
      grid: { left: 52, right: 20, top: 40, bottom: 40 },
      xAxis: { type: "value", min: 1982, max: 2027, ...axis({ axisLabel: { color: C.muted, formatter: (v: number) => String(v) } }) },
      yAxis: { type: "value", name: "lake water km²", scale: true, ...axis() },
      series: [
        { name: "Landsat 5/7/8 (TOA MNDWI)", type: "line", data: ls, symbol: "circle", symbolSize: 6, lineStyle: { color: "#9fb4bb", width: 1.5 }, itemStyle: { color: "#9fb4bb" },
          markArea: { silent: true, itemStyle: { color: "rgba(239,122,106,0.10)" }, label: { color: C.danger, fontSize: 11 },
            data: [[{ name: "World Heritage in Danger", xAxis: 1996 }, { xAxis: 2006 }]] },
          markLine: { symbol: "none", silent: true, lineStyle: { color: C.faint, type: "dotted" }, label: { color: C.muted, fontSize: 10, formatter: "{b}" },
            data: ev.map((e) => ({ name: e.t, xAxis: e.x })) } },
        { name: "Sentinel-2 habitat maps", type: "line", data: s2, symbol: "circle", symbolSize: 7, lineStyle: { color: C.accent, width: 2.5 }, itemStyle: { color: C.accent } },
      ],
    });
    const t = (site as any).landsat_trend[s];
    $("landsat-text").innerHTML = `Mean ${s}-season lake water: ${fmt(t.mean_1984_1995, 1)} km² (1984–95), ${fmt(t.mean_1996_2006, 1)} km² (1996–2006), ${fmt(t.mean_2007_2016, 1)} km² (2007–16), ${fmt(t.mean_2017_2026, 1)} km² (2017–26). ` +
      `Whole-series Mann–Kendall: Sen's slope ${signed(t.sen_slope, 2)} km²/yr, p = ${pval(t.p)}. A single Otsu threshold (MNDWI ${fmt(site.landsat_overlap.threshold, 3)}) is applied to every Landsat composite. Seasons with less than 90 % clear basin pixels are omitted.`;
  };
  segmented("ls-season", (v) => draw(v as Season));
  draw("wet");
  const sv = site.sav;
  chart("sav-chart").render({
    tooltip: tooltip({ trigger: "axis" }),
    grid: { left: 46, right: 46, top: 24, bottom: 30 },
    xAxis: { type: "category", data: sv.series.map((r) => String(r.year)), ...axis() },
    yAxis: [{ type: "value", name: "km²", ...axis() }, { type: "value", name: "% of lake", max: 100, ...axis({ splitLine: { show: false } }) }],
    series: [
      { name: "Submerged vegetation", type: "bar", barWidth: "55%", data: sv.series.map((r) => +r.sav_km2.toFixed(2)), itemStyle: { color: "#41ab5d", borderRadius: [4, 4, 0, 0] } },
      { name: "Share of lake", type: "line", yAxisIndex: 1, data: sv.series.map((r) => +(r.sav_share * 100).toFixed(1)), itemStyle: { color: C.text }, lineStyle: { color: C.text, width: 1.5 } },
    ],
  });
  $("sav-text").innerHTML = `<i>Potamogeton</i> beds, the main winter food of ducks and coots, show up as patches greener than the surrounding water (NDAVI more than ${fmt(sv.ndavi_threshold, 2)} above that winter's lake median). Lake-wide NDAVI shifts strongly with turbidity, so this series is exploratory and uncalibrated. Year-to-year swings are large and there is no significant trend (p = ${pval(sv.trend.p)}).`;
  const ov = site.landsat_overlap as any;
  const mx = Math.max(...ov.pairs.flatMap((p: any) => [p.landsat, p.s2])) * 1.05;
  chart("overlap-chart").render({
    tooltip: tooltip({ trigger: "item", formatter: (p: any) => p.value?.[2] ? `${p.value[2]}<br/>Landsat ${fmt(p.value[0], 1)} · S2 ${fmt(p.value[1], 1)} km²` : "" }),
    grid: { left: 50, right: 20, top: 16, bottom: 42 },
    xAxis: { type: "value", name: "Landsat km²", nameLocation: "middle", nameGap: 26, min: 0, max: Math.ceil(mx), ...axis() },
    yAxis: { type: "value", name: "Sentinel-2 km²", min: 0, max: Math.ceil(mx), ...axis() },
    series: [
      { type: "scatter", symbolSize: 10, data: ov.pairs.map((p: any) => [p.landsat, p.s2, `${p.year} ${p.season}`]), itemStyle: { color: C.accent } },
      { type: "line", symbol: "none", silent: true, data: [[0, 0], [mx, mx]], lineStyle: { color: C.faint, type: "dashed" } },
    ],
  });
  $("overlap-text").textContent = `${ov.n} overlapping seasons: Pearson r = ${fmt(ov.r, 3)}, mean absolute difference ${fmt(ov.mean_abs_diff_km2, 1)} km², bias (Landsat − S2) ${signed(ov.bias_km2, 1)} km².`;
}

// ------------------------------------------------------------------ model lab
function lab(site: Site) {
  const M = site.models;
  const models = [...new Set(M.curves.map((r) => r.model))];
  const fr = [0.05, 0.1, 0.25, 0.5, 1];
  const FR_LABELS = ["5 %", "10 %", "25 %", "50 %", "100 %"];
  const series: any[] = [];
  for (const m of models) {
    const rows = fr.map((f) => M.curves.find((r) => r.model === m && r.fraction === f));
    if (rows.filter(Boolean).length < 2) continue;
    const col = MODEL_COLOR[m] ?? C.text;
    series.push({ name: MODEL_LABEL[m] ?? m, type: "line", symbol: "circle", symbolSize: 7, lineStyle: { color: col, width: 2.5 }, itemStyle: { color: col },
      data: rows.map((r, i) => (r ? [FR_LABELS[i], +r.miou_mean.toFixed(4), r.miou_std] : null)).filter(Boolean) });
    series.push({ name: MODEL_LABEL[m] ?? m, type: "custom", silent: true, z: 1, data: rows.map((r, i) => (r ? [i, r.miou_mean - r.miou_std, r.miou_mean + r.miou_std] : null)).filter(Boolean),
      renderItem: (_: any, api: any) => {
        const x = api.coord([api.value(0), api.value(1)]);
        const y = api.coord([api.value(0), api.value(2)]);
        return { type: "line", shape: { x1: x[0], y1: x[1], x2: y[0], y2: y[1] }, style: { stroke: col, lineWidth: 2, opacity: 0.6 } };
      } });
  }
  chart("lec-chart").render({
    tooltip: tooltip({ trigger: "item", formatter: (p: any) => p.seriesType === "line"
      ? `${p.seriesName}<br/>${p.value[0]} of training chips<br/>mIoU <b>${fmt(p.value[1], 3)}</b> ± ${fmt(p.value[2], 3)}` : "" }),
    legend: legend({ top: 0, data: models.map((m) => MODEL_LABEL[m] ?? m) }),
    grid: { left: 56, right: 24, top: 56, bottom: 46 },
    xAxis: { type: "category", data: FR_LABELS, boundaryGap: false, name: "share of training chips", nameLocation: "middle", nameGap: 28, ...axis() },
    yAxis: { type: "value", name: "test mIoU", scale: true, ...axis() },
    series,
  });
  const full = [...M.rows].sort((a, b) => b.miou_mean - a.miou_mean);
  $("model-table").innerHTML = `<tr><th>Model</th><th class="r">mIoU</th><th class="r">mIoU 2017–25<br>(matched)</th><th class="r">Overall acc.</th><th class="r">Macro F1</th><th class="r">κ</th><th class="r">Seeds</th></tr>` +
    full.map((r) => `<tr class="${r.model === M.best ? "best" : ""}"><td>${r.label}</td><td class="r">${fmt(r.miou_mean, 3)}${r.n > 1 ? ` ± ${fmt(r.miou_std, 3)}` : ` <span class="small muted">(1 seed)</span>`}</td><td class="r">${fmt((r as any).miou_matched, 3)}</td>
      <td class="r">${fmt(r.oa_mean * 100, 1)} %</td><td class="r">${fmt(r.f1_mean, 3)}</td><td class="r">${fmt(r.kappa_mean, 3)}</td><td class="r">${r.n}</td></tr>`).join("");
  $("model-note").innerHTML = (M.gpu_pending
    ? "GPU runs (U-Net, Prithvi-EO-2.0, TerraMind) are queued on a Kaggle T4. Rows appear here as soon as their results are added to <span class=\"mono\">results/gpu/</span>. "
    : "") + `Test set: ${site.meta.test_locations} held-out 2.24 km chip locations × 20 seasons, every second pixel; mean ± standard deviation over seeds. AlphaEarth embeddings stop at 2025, so that baseline is tested on 2017–2025 only; the matched column scores every model on those same seasons.` +
    (M.compute.length ? `<br/>Compute: ${M.compute.join("; ")}` : "");
  chart("f1-chart").render({
    tooltip: tooltip({ trigger: "axis", valueFormatter: (v: number) => fmt(v, 3) }),
    legend: legend({ top: 0 }),
    grid: { left: 150, right: 16, top: 50, bottom: 30 },
    xAxis: { type: "value", max: 1, ...axis() },
    yAxis: { type: "category", inverse: true, data: site.classes.map((c) => c.name), ...axis({ splitLine: { show: false } }) },
    series: full.map((r) => ({ name: r.label, type: "bar", barGap: "10%", barCategoryGap: "30%", itemStyle: { color: MODEL_COLOR[r.model], borderRadius: 2 }, data: r.per_class_f1 })),
  });
  const cm = M.confusion;
  $("cm-model").textContent = MODEL_LABEL[M.confusion_model] ?? M.confusion_model;
  const data: [number, number, number][] = [];
  cm.forEach((row, i) => { const s = row.reduce((a, b) => a + b, 0) || 1; row.forEach((v, j) => data.push([j, i, +(100 * v / s).toFixed(1)])); });
  const short = ["Perm. water", "Seas. water", "Fresh marsh", "Halo. marsh", "Saline flat", "Crop/grass", "Shrub/wood", "Built"];
  chart("cm-chart").render({
    tooltip: tooltip({ formatter: (p: any) => `reference <b>${short[p.value[1]]}</b><br/>predicted <b>${short[p.value[0]]}</b><br/>${p.value[2]} % of row` }),
    grid: { left: 90, right: 16, top: 10, bottom: 70 },
    xAxis: { type: "category", data: short, name: "predicted", nameLocation: "middle", nameGap: 54, ...axis({ axisLabel: { color: C.muted, rotate: 40, fontSize: 10 } }) },
    yAxis: { type: "category", data: short, inverse: true, name: "", ...axis({ axisLabel: { color: C.muted, fontSize: 10 } }) },
    visualMap: { show: false, min: 0, max: 100, inRange: { color: ["#0b171c", "#13474a", "#2bb3a3", "#c8f5ec"] } },
    series: [{ type: "heatmap", data, label: { show: true, color: C.text, fontSize: 10, formatter: (p: any) => (p.value[2] >= 1 ? p.value[2].toFixed(0) : "") },
      itemStyle: { borderColor: "#071014", borderWidth: 1 } }],
  });
  // chip comparison
  const sel = $<HTMLSelectElement>("chip-select");
  sel.innerHTML = site.chips.map((c, i) => `<option value="${i}">Test location ${c.id.replace("_", " ")} · ${c.year} ${c.season}</option>`).join("");
  const drawChip = async () => {
    const c = site.chips[Number(sel.value)];
    const cells: [string, string][] = [["Sentinel-2", c.s2], ["Weak labels (white = unlabeled)", c.label],
      ...Object.entries(c.preds).map(([m, u]) => [MODEL_LABEL[m] ?? m, u] as [string, string])];
    $("chips").innerHTML = cells.map(([t], i) => `<div class="chip-cell"><canvas id="chip-${i}" width="224" height="224"></canvas>${t}</div>`).join("");
    cells.forEach(async ([, u], i) => {
      const im = new Image(); im.src = dataUrl(u); await im.decode();
      const ctx = ($(`chip-${i}`) as HTMLCanvasElement).getContext("2d")!;
      ctx.fillStyle = i === 1 ? "#f3f6f7" : "#0a1418"; ctx.fillRect(0, 0, 224, 224);
      ctx.drawImage(im, 0, 0, 224, 224);
    });
    $("chip-caption").textContent = "Same 2.24 km test chip, never seen in training.";
  };
  sel.addEventListener("change", drawChip);
  if (site.chips.length) drawChip();
}

// ------------------------------------------------------------------ method, footer
function method(site: Site) {
  $("rules-table").innerHTML = `<tr><th>Class</th><th>Rule (all conditions)</th></tr>` +
    site.rules.map((r, i) => `<tr><td style="white-space:nowrap"><span class="sw" style="display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:8px;background:${site.classes[i].color}"></span>${r.cls}</td><td class="muted small">${r.rule}</td></tr>`).join("");
  $("data-table").innerHTML = `<tr><th>Dataset</th><th>Use</th><th>Licence</th></tr>` +
    site.data_sources.map((d) => `<tr><td>${d.name}</td><td class="muted small">${d.use}</td><td class="muted small">${d.licence}</td></tr>`).join("");
  $("limits-list").innerHTML = site.limitations.map((l) => `<li>${l}</li>`).join("");
  $("credits").innerHTML = `<b style="color:var(--text)">Credits</b><br/>Contains modified Copernicus Sentinel data 2017–2026 (ESA). ESA WorldCover (CC BY 4.0), JRC Global Surface Water, Copernicus DEM, AlphaEarth Foundations embeddings (Google, CC BY 4.0), PERSIANN-CDR (NOAA), Landsat (USGS). Basemap imagery © Esri and partners, labels © OpenStreetMap contributors, © CARTO. Results generated ${site.meta.generated}.`;
  $("sources").innerHTML = `<b style="color:var(--text)">Sources</b><ul class="clean">${site.sources.map((s) => `<li><a href="${s.url}" rel="noopener">${s.name}</a></li>`).join("")}</ul>`;
}

function chrome() {
  const io = new IntersectionObserver((es) => es.forEach((e) => e.isIntersecting && e.target.classList.add("in")), { threshold: 0.12 });
  document.querySelectorAll(".reveal").forEach((el) => io.observe(el));
  const links = [...document.querySelectorAll<HTMLAnchorElement>(".nav .links a")];
  const secs = links.map((a) => document.querySelector(a.getAttribute("href")!)!);
  addEventListener("scroll", () => {
    const y = scrollY + 120;
    let k = -1;
    secs.forEach((s, i) => { if ((s as HTMLElement).offsetTop <= y) k = i; });
    links.forEach((a, i) => a.classList.toggle("active", i === k));
  }, { passive: true });
}

async function main() {
  chrome();
  const site = await loadSite();
  hero(site);
  context(site);
  timeMachine(site);
  breathing(site);
  change(site);
  drivers(site);
  forty(site);
  lab(site);
  method(site);
}
main().catch((e) => {
  console.error(e);
  document.body.insertAdjacentHTML("afterbegin", `<p style="padding:90px 20px;color:#ef7a6a">Could not load results: ${e}</p>`);
});
