import "@fontsource-variable/inter";
import "@fontsource-variable/fraunces";
import "./style.css";
import { dataUrl, loadSite, type Site } from "./data";

interface Pt { id: number; year: number; season: string; row: number; col: number; lon: number; lat: number }
const KEY = "ichkeul-gold-labels-v1";
const HALF = 32;

async function main() {
  const site: Site = await loadSite();
  const { points } = (await (await fetch(dataUrl("gold_points.json"))).json()) as { points: Pt[] };
  let labels: Record<string, number | null> = {};
  try { labels = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch { /* private mode */ }
  const save = () => { try { localStorage.setItem(KEY, JSON.stringify(labels)); } catch { /* ignore */ } };
  let i = Math.max(0, points.findIndex((p) => !(String(p.id) in labels)));

  const imgs = new Map<string, Promise<HTMLImageElement>>();
  const img = (u: string) => {
    if (!imgs.has(u)) imgs.set(u, new Promise((res, rej) => { const im = new Image(); im.onload = () => res(im); im.onerror = rej; im.src = dataUrl(u); }));
    return imgs.get(u)!;
  };
  const season = (y: number, s: string) => site.seasons.find((e) => e.year === y && e.season === s)!;

  const keys = document.getElementById("keys")!;
  site.classes.forEach((c) => {
    const b = document.createElement("button");
    b.className = "btn";
    b.innerHTML = `<span class="sw" style="background:${c.color}"></span>${c.name}<kbd>${c.id + 1}</kbd>`;
    b.onclick = () => answer(c.id);
    keys.appendChild(b);
  });

  async function draw() {
    const p = points[i];
    const views: [string, string][] = [
      [season(p.year, "wet").s2, "c0"], [`img/fc_${p.year}_wet.jpg`, "c1"],
      [season(p.year, "dry").s2, "c2"], [`img/fc_${p.year}_dry.jpg`, "c3"],
    ];
    for (const [u, id] of views) {
      const cv = document.getElementById(id) as HTMLCanvasElement;
      const ctx = cv.getContext("2d")!;
      ctx.fillStyle = "#0a1418"; ctx.fillRect(0, 0, 64, 64);
      try {
        const im = await img(u);
        ctx.imageSmoothingEnabled = false;
        ctx.drawImage(im, p.col - HALF, p.row - HALF, 64, 64, 0, 0, 64, 64);
      } catch { /* false-colour image only exists for sampled years */ }
      ctx.strokeStyle = "#ff3b3b"; ctx.lineWidth = 1;
      ctx.strokeRect(HALF - 1.5, HALF - 1.5, 3, 3);
    }
    const done = Object.keys(labels).length;
    document.getElementById("status")!.textContent =
      `Point ${i + 1} / ${points.length} · ${p.year} wet season · labelled ${done}` +
      (String(p.id) in labels ? ` · current answer: ${labels[p.id] == null ? "skipped" : site.classes[labels[p.id]!].name}` : "");
    (document.getElementById("bar") as HTMLElement).style.width = `${(100 * done) / points.length}%`;
    document.getElementById("coords")!.textContent = `lat ${p.lat.toFixed(5)}, lon ${p.lon.toFixed(5)}`;
  }
  function answer(k: number | null) {
    labels[points[i].id] = k; save();
    if (i < points.length - 1) i++;
    draw();
  }
  document.getElementById("prev")!.onclick = () => { if (i > 0) i--; draw(); };
  document.getElementById("next")!.onclick = () => { if (i < points.length - 1) i++; draw(); };
  document.getElementById("skip")!.onclick = () => answer(null);
  document.getElementById("export")!.onclick = () => {
    const blob = new Blob([JSON.stringify({ labels, exported: new Date().toISOString() }, null, 1)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob); a.download = "gold_labels.json"; a.click();
  };
  addEventListener("keydown", (e) => {
    const n = Number(e.key);
    if (n >= 1 && n <= site.classes.length) answer(n - 1);
    else if (e.key === "ArrowLeft") document.getElementById("prev")!.click();
    else if (e.key === "ArrowRight") document.getElementById("next")!.click();
    else if (e.key === "0" || e.key === "s") answer(null);
  });
  draw();
}
main();
