import * as echarts from "echarts/core";
import { BarChart, CustomChart, HeatmapChart, LineChart, SankeyChart, ScatterChart } from "echarts/charts";
import {
  DataZoomComponent, GridComponent, LegendComponent, MarkAreaComponent, MarkLineComponent,
  TooltipComponent, VisualMapComponent,
} from "echarts/components";
import { CanvasRenderer } from "echarts/renderers";

echarts.use([BarChart, CustomChart, HeatmapChart, LineChart, SankeyChart, ScatterChart, DataZoomComponent, GridComponent,
  LegendComponent, MarkAreaComponent, MarkLineComponent, TooltipComponent, VisualMapComponent, CanvasRenderer]);

export const C = { text: "#e7f0f2", muted: "#8ea6ae", faint: "#5d7880", line: "#1f3740", accent: "#4fd8c4", warn: "#f2b84b", danger: "#ef7a6a" };

export const axis = (extra: Record<string, unknown> = {}) => ({
  axisLine: { lineStyle: { color: C.line } }, axisTick: { show: false },
  axisLabel: { color: C.muted, fontSize: 11 }, splitLine: { lineStyle: { color: "rgba(31,55,64,0.55)" } },
  nameTextStyle: { color: C.muted, fontSize: 11 }, ...extra,
});
export const tooltip = (extra: Record<string, unknown> = {}) => ({
  backgroundColor: "rgba(11,23,28,0.96)", borderColor: C.line, textStyle: { color: C.text, fontSize: 12 },
  extraCssText: "border-radius:10px;box-shadow:0 10px 30px rgba(0,0,0,.4)", ...extra,
});
export const legend = (extra: Record<string, unknown> = {}) => ({
  textStyle: { color: C.muted, fontSize: 11 }, icon: "roundRect", itemWidth: 12, itemHeight: 8, ...extra,
});

const charts: echarts.ECharts[] = [];
const BASE = { textStyle: { fontFamily: "Inter Variable, system-ui, sans-serif" }, animationDuration: 900 };

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
