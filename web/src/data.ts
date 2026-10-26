// Types and loader for public/data/site.json (written by src/ichkeul/export.py).
export type Season = "wet" | "dry";

export interface ClassDef { id: number; key: string; name: string; color: string }
export interface SeasonEntry { year: number; season: Season; s2: string; map: string; scenes: number; obs: number; dates: string[] }
export interface Trend { n: number; tau: number; p: number; sen_slope: number; slope_ci95: [number, number]; intercept: number; trend: string }
export interface ModelRow {
  model: string; label: string; fraction: number; n: number;
  miou_mean: number; miou_std: number; oa_mean: number; oa_std: number; f1_mean: number; kappa_mean: number;
  per_class_f1: number[]; n_train_chips: number;
}

export interface Site {
  meta: {
    generated: string; map_model: string; map_model_label: string;
    corners: [number, number][]; center: [number, number]; bounds: [number, number, number, number];
    basin_km2: number; years: number[]; test_locations: number; train_locations: number; val_locations: number;
  };
  classes: ClassDef[];
  seasons: SeasonEntry[];
  layers: Record<string, { url: string; legend: string; title: string; ramp?: string[]; labels?: string[] }>;
  basin: any;
  areas: Record<string, number | string>[];
  trends: Record<Season, Record<string, Trend>>;
  transitions: Record<Season, { from_years: number[]; to_years: number[]; matrix_km2: number[][] }>;
  inundation: Record<string, number>;
  pixel_trends: Record<string, { sig_increase_km2: number; sig_decrease_km2: number }>;
  sav: { ndavi_threshold: number; series: { year: number; sav_km2: number; lake_km2: number; sav_share: number }[]; trend: Trend };
  drivers: any;
  rain: { year: number; rain_wet_antecedent: number; rain_dry_antecedent: number; rain_prev_hydro_year: number; rain_hydro_year: number | null }[];
  landsat: { year: number; season: Season; water_km2: number | null; source: string; scenes: number }[];
  landsat_overlap: { r: number; mean_abs_diff_km2: number; bias_km2: number; n: number; threshold: number };
  models: { rows: ModelRow[]; curves: ModelRow[]; best: string; confusion: number[][]; confusion_model: string; gpu_pending: boolean; compute: string[] };
  chips: { id: string; year: number; season: Season; s2: string; label: string; preds: Record<string, string> }[];
  birds: { label: string; value: number; kind: string; source: string }[];
  timeline: { year: string; text: string; bad?: boolean }[];
  headline: { label: string; value: string; sub: string }[];
  rules: { cls: string; rule: string }[];
  data_sources: { name: string; use: string; licence: string }[];
  limitations: string[];
  sources: { name: string; url: string }[];
}

export const MODEL_LABEL: Record<string, string> = {
  rf_indices: "Random Forest · spectral indices",
  rf_alphaearth: "Random Forest · AlphaEarth embeddings",
  unet: "U-Net (ResNet-34)",
  unet_cpu: "U-Net · random init (CPU)",
  prithvi: "Prithvi-EO-2.0-300M",
  prithvi_frozen: "Prithvi-EO-2.0 · frozen encoder",
  prithvi_mt: "Prithvi-EO-2.0 · 3 dates",
  terramind: "TerraMind-1.0-base",
};
export const MODEL_COLOR: Record<string, string> = {
  rf_indices: "#f2b84b", rf_alphaearth: "#c792ea", unet: "#ef7a6a", unet_cpu: "#d98b7c", prithvi: "#4fd8c4",
  prithvi_frozen: "#2a8f80", prithvi_mt: "#9be7da", terramind: "#56b4e9",
};

export const dataUrl = (p: string) => new URL(import.meta.env.BASE_URL + "data/" + p, location.href).href;

export async function loadSite(): Promise<Site> {
  const r = await fetch(dataUrl("site.json"));
  if (!r.ok) throw new Error("site.json missing");
  return r.json();
}

export const fmt = (v: number | null | undefined, d = 1) =>
  v == null || !isFinite(v) ? "–" : v.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
export const signed = (v: number, d = 2) => (v > 0 ? "+" : v < 0 ? "−" : "±") + fmt(Math.abs(v), d);
export const pval = (p: number) => (p < 0.001 ? "< 0.001" : fmt(p, 3));
