// hooks/useCmemsLegend.ts
import { useQuery } from "@tanstack/react-query";
import type { WmtsLayer, WmtsStyle } from "./useWmtsLayer";

export type LegendJson = {
  continuous: {
    clamp: boolean;
    logScale: boolean;
    valueMin: number;
    valueMax: number;
    cmap: {
      brightnessType?: string;
      colorMap: number[][];
      colorMapStrings: string[];
    };
    cmapName?: string;
    units?: string;
    variableId?: string;
    variableName?: string;
  };
  // Some servers also repeat below at root; we read from continuous primarily:
  cmapName?: string;
  logScale?: boolean;
  units?: string;
  valueMin?: number;
  valueMax?: number;
  variableId?: string;
  variableName?: string;
};

export type UseCmemsLegendParams = {
  layer: WmtsLayer;
  styleName: WmtsStyle;
  maxStops?: number; // default 128
};

const WMTS_BASE = "https://wmts.marine.copernicus.eu/teroWmts";
const PRODUCT_PATH =
  "GLOBAL_ANALYSISFORECAST_WAV_001_027/cmems_mod_glo_wav_anfc_0.083deg_PT3H-i_202411";

function buildLegendUrl(
  layer: string,
  styleName: string
) {
  const params = new URLSearchParams({
    SERVICE: "WMTS",
    REQUEST: "GetLegend",
    LAYER: PRODUCT_PATH + "/" + layer,
    STYLE: styleName,
    FORMAT: "application/json",
  });
  return `${WMTS_BASE}?${params.toString()}`;
}

function sampleStops<T>(arr: T[], maxStops = 128): T[] {
  if (arr.length <= maxStops) return arr;
  const out: T[] = [];
  for (let i = 0; i < maxStops; i++) {
    const idx = Math.round((i * (arr.length - 1)) / (maxStops - 1));
    out.push(arr[idx]);
  }
  return out;
}

export function useCmemsLegend({
  layer,
  styleName,
  maxStops = 128,
}: UseCmemsLegendParams) {
  const enabled = !!layer;

  const query = useQuery({
    queryKey: ["cmems-legend", layer, styleName, maxStops],
    enabled,
    queryFn: async (): Promise<{
      legend: LegendJson;
      gradientCss: string;
      min: number;
      max: number;
      units: string | undefined;
      variableName: string | undefined;
    }> => {
      if (!layer) throw new Error("Missing capabilities/layer");

      const url = buildLegendUrl(layer, styleName);
      const res = await fetch(url, { mode: "cors" });
      if (!res.ok) throw new Error(`GetLegend failed: ${res.status}`);
      const legend: LegendJson = await res.json();

      const cont = legend.continuous ?? ({} as LegendJson["continuous"]);
      const cm = cont.cmap?.colorMapStrings || [];
      const stops = sampleStops(cm, maxStops);

      // Compose a smooth continuous gradient
      const gradientCss = `linear-gradient(to right, ${stops.join(", ")})`;

      const min = cont.valueMin ?? legend.valueMin ?? 0;
      const max = cont.valueMax ?? legend.valueMax ?? 1;
      const units = cont.units ?? legend.units;
      const variableName = cont.variableName ?? legend.variableName;

      return { legend, gradientCss, min, max, units, variableName };
    },
    staleTime: 10 * 60 * 1000,
  });

  return query;
}
