import * as maplibregl from "maplibre-gl";
import { setWorkerUrl, type Map as MLMap, type StyleSpecification } from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import { dataUrl, type Site } from "./data";

// MapLibre 6 loads its worker as a module (copied to <base>/maplibre/ by vite.config.ts)
setWorkerUrl(new URL(import.meta.env.BASE_URL + "maplibre/maplibre-gl-worker.mjs", location.href).href);

export function baseStyle(dim = 0.55): StyleSpecification {
  return {
    version: 8,
    sources: {
      imagery: {
        type: "raster", tileSize: 256, maxzoom: 18,
        tiles: ["https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"],
        attribution: "Imagery © Esri, Maxar, Earthstar Geographics · Sentinel-2 © Copernicus (ESA)",
      },
      labels: {
        type: "raster", tileSize: 256,
        tiles: ["https://basemaps.cartocdn.com/dark_only_labels/{z}/{x}/{y}.png"],
        attribution: "© OpenStreetMap contributors © CARTO",
      },
    },
    layers: [
      { id: "bg", type: "background", paint: { "background-color": "#0a1418" } },
      { id: "imagery", type: "raster", source: "imagery",
        paint: { "raster-brightness-max": dim, "raster-saturation": -0.35, "raster-contrast": 0.05 } },
    ],
  };
}

export function makeMap(container: string | HTMLElement, site: Site, opts: Partial<maplibregl.MapOptions> = {}): MLMap {
  const [w, s, e, n] = site.meta.bounds;
  const map = new maplibregl.Map({
    container, style: baseStyle(), bounds: [[w, s], [e, n]], fitBoundsOptions: { padding: 10 },
    attributionControl: { compact: true }, maxZoom: 16, minZoom: 9, dragRotate: false, pitchWithRotate: false,
    ...opts,
  });
  map.touchZoomRotate.disableRotation();
  return map;
}

/** Runs fn once the style is parsed, without waiting for basemap tiles (a slow tile server must not block overlays). */
export function onReady(map: MLMap, fn: () => void) {
  let done = false;
  const run = () => { if (!done) { done = true; fn(); } };
  if (map.isStyleLoaded()) run();
  else { map.once("style.load", run); map.once("load", run); }
}

export function addLabels(map: MLMap) {
  if (map.getLayer("labels")) return;
  map.addLayer({ id: "labels", type: "raster", source: "labels", paint: { "raster-opacity": 0.8 } });
}

/** Adds (or updates) a georeferenced PNG/JPEG covering the study area. */
export function setImage(map: MLMap, id: string, url: string, site: Site, opacity = 1, before?: string) {
  const src = map.getSource(id) as maplibregl.ImageSource | undefined;
  const full = url.startsWith("data:") ? url : dataUrl(url);
  if (src) {
    src.updateImage({ url: full, coordinates: site.meta.corners as any });
  } else {
    map.addSource(id, { type: "image", url: full, coordinates: site.meta.corners as any });
    map.addLayer({ id, type: "raster", source: id,
      paint: { "raster-opacity": opacity, "raster-resampling": "nearest", "raster-fade-duration": 0 } },
      before && map.getLayer(before) ? before : undefined);
  }
}

export function addBasin(map: MLMap, site: Site, color = "#ffffff") {
  if (map.getSource("basin")) return;
  map.addSource("basin", { type: "geojson", data: site.basin as any });
  map.addLayer({ id: "basin", type: "line", source: "basin",
    paint: { "line-color": color, "line-width": 1.2, "line-opacity": 0.55, "line-dasharray": [3, 2] } });
}

const cache = new Set<string>();
export function preload(urls: string[]) {
  for (const u of urls) {
    if (cache.has(u)) continue;
    cache.add(u);
    const img = new Image();
    img.src = dataUrl(u);
  }
}

/** Two synchronised maps, the right one clipped by a draggable divider. */
export class Swipe {
  pos = 0.5;
  private syncing = false;
  constructor(public a: MLMap, public b: MLMap, private box: HTMLElement, private handle: HTMLElement) {
    const sync = (from: MLMap, to: MLMap) => () => {
      if (this.syncing) return;
      this.syncing = true;
      to.jumpTo({ center: from.getCenter(), zoom: from.getZoom(), bearing: from.getBearing(), pitch: from.getPitch() });
      this.syncing = false;
    };
    a.on("move", sync(a, b));
    b.on("move", sync(b, a));
    let drag = false;
    const move = (x: number) => {
      const r = box.getBoundingClientRect();
      this.set(Math.min(0.98, Math.max(0.02, (x - r.left) / r.width)));
    };
    handle.addEventListener("pointerdown", (e) => { drag = true; handle.setPointerCapture(e.pointerId); });
    handle.addEventListener("pointermove", (e) => drag && move(e.clientX));
    handle.addEventListener("pointerup", () => (drag = false));
    new ResizeObserver(() => this.set(this.pos)).observe(box);
    this.set(0.5);
  }
  set(p: number) {
    this.pos = p;
    const w = this.box.clientWidth;
    this.handle.style.left = `${p * w}px`;
    (this.b.getContainer() as HTMLElement).style.clipPath = `inset(0 0 0 ${p * w}px)`;
  }
}
