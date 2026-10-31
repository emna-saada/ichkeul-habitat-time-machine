import * as echarts from "echarts/core";
import { BarChart, CustomChart, HeatmapChart, LineChart, SankeyChart, ScatterChart } from "echarts/charts";
import {
  DataZoomComponent, GridComponent, LegendComponent, MarkAreaComponent, MarkLineComponent,
  TooltipComponent, VisualMapComponent,
} from "echarts/components";
import { CanvasRenderer } from "echarts/renderers";

echarts.use([BarChart, CustomChart, HeatmapChart, LineChart, SankeyChart, ScatterChart, DataZoomComponent, GridComponent,
  LegendComponent, MarkAreaComponent, MarkLineComponent, TooltipComponent, VisualMapComponent, CanvasRenderer]);

export const C = {
  text: "#16241F", ink2: "#36443E", muted: "#56635C", faint: "#8C8676", line: "#D5CCB9", paper: "#FBF9F4",
  accent: "#166534", warn: "#8A5A06", danger: "#9A3324", water: "#1D4E89", waterLight: "#7FB3D5", grey: "#7A7468",
};
const FONT = "Atkinson Hyperlegible Next Variable, system-ui, sans-serif";
const MONO = "IBM Plex Mono, ui-monospace, monospace";

export const axis = (extra: Record<string, unknown> = {}) => ({
  axisLine: { lineStyle: { color: C.text } }, axisTick: { show: true, length: 4, lineStyle: { color: C.text } },
  axisLabel: { color: C.ink2, fontSize: 11, fontFamily: MONO }, splitLine: { lineStyle: { color: C.line, type: [3, 3] } },
  nameTextStyle: { color: C.muted, fontSize: 11, fontFamily: MONO }, ...extra,
});
export const tooltip = (extra: Record<string, unknown> = {}) => ({
  backgroundColor: "#FBF9F4", borderColor: C.text, borderWidth: 1, padding: [8, 12],
  textStyle: { color: C.text, fontSize: 12, fontFamily: FONT },
  extraCssText: "border-radius:2px;box-shadow:0 6px 18px rgba(22,36,31,.14)", ...extra,
});
export const legend = (extra: Record<string, unknown> = {}) => ({
  textStyle: { color: C.ink2, fontSize: 11, fontFamily: FONT }, itemWidth: 22, itemHeight: 10, itemGap: 14, ...extra,
});
/** Hatching so emphasis never depends on colour alone. */
export const HATCH = { symbol: "rect", symbolSize: 1, dashArrayX: [1, 0], dashArrayY: [2, 4], rotation: -Math.PI / 4, color: "rgba(251,249,244,.55)" };

const charts: echarts.ECharts[] = [];
const BASE = { textStyle: { fontFamily: FONT }, animationDuration: 700, animationEasing: "cubicOut" as const };

/** A chart initialised lazily, when its container first scrolls into view; render() always takes a full option. */
export function chart(id: string) {
  const el = document.getElementById(id)!;
  let inst: echarts.ECharts | null = null;
  let last: echarts.EChartsCoreOption | null = null;
  new IntersectionObserver((es, ob) => {
    if (!es.some((e) => e.isIntersecting)) return;
    ob.disconnect();
    inst = echarts.init(el, undefined, { renderer: "canvas" });
    charts.push(inst);
    if (last) inst.setOption({ ...BASE, ...last }, true);
  }, { rootMargin: "300px" }).observe(el);
  return {
    render(o: echarts.EChartsCoreOption) {
      last = o;
      inst?.setOption({ ...BASE, ...o }, true);
    },
  };
}
addEventListener("resize", () => charts.forEach((c) => c.resize()));
