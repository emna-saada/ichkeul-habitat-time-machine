import { resolve } from "node:path";
import { defineConfig } from "vite";
import { viteStaticCopy } from "vite-plugin-static-copy";

export default defineConfig({
  base: "./", // works under any GitHub Pages sub-path
  plugins: [
    viteStaticCopy({
      // MapLibre 6 worker module and the shared chunk it imports (see src/maps.ts)
      targets: ["maplibre-gl-worker.mjs", "maplibre-gl-shared.mjs"].map((f) => ({
        src: `node_modules/maplibre-gl/dist/${f}`, dest: "maplibre", rename: { stripBase: true } })),
    }),
  ],
  build: {
    chunkSizeWarningLimit: 2500,
    rollupOptions: { input: { main: resolve(__dirname, "index.html"), label: resolve(__dirname, "label.html") } },
  },
});
