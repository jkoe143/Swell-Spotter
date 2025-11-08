import { useMemo } from "react";
import { TileLayer } from "@deck.gl/geo-layers";
import { BitmapLayer } from "@deck.gl/layers";

export type WmtsLayer =
  | "VHM0"
  | "VCMX"
  | "VHM0_SW1"
  | "VHM0_SW2"
  | "VHM0_WW"
  | "VMDR"
  | "VMDR_SW1"
  | "VMDR_SW2"
  | "VMDR_WW"
  | "VMXL"
  | "VPED"
  | "VSDX"
  | "VSDY"
  | "VTM01_SW1"
  | "VTM01_SW2"
  | "VTM01_WW"
  | "VTM02"
  | "VTM10"
  | "VTPK";

export type WmtsStyle =
  | "cmap:amp"
  | "cmap:amp,logScale"
  | "cmap:tempo"
  | "cmap:tempo,logScale"
  | "cmap:cyclic"
  | "cmap:cyclic,logScale";

type Props = {
  id?: string;
  timeISO: string | null; // e.g., "2025-11-01T18:00:00Z"
  layer: WmtsLayer;
  styleName: WmtsStyle;
  opacity?: number;
};

function makeWmtsTemplate(layer: string, styleName: string, timeISO: string) {
  return (
    "https://wmts.marine.copernicus.eu/teroWmts" +
    "?SERVICE=WMTS" +
    "&REQUEST=GetTile" +
    "&VERSION=1.0.0" +
    `&LAYER=GLOBAL_ANALYSISFORECAST_WAV_001_027/cmems_mod_glo_wav_anfc_0.083deg_PT3H-i_202411/${layer}` +
    `&STYLE=${encodeURIComponent(styleName)}` +
    "&FORMAT=image/png" +
    "&TILEMATRIXSET=EPSG:3857" +
    "&TILEMATRIX={z}&TILEROW={y}&TILECOL={x}" +
    `&TIME=${encodeURIComponent(timeISO)}`
  );
}

export function useCmemsWmtsLayer({
  layer,
  styleName,
  timeISO,
  opacity = 0.8,
}: Props) {
  return useMemo(() => {
    if (!timeISO) return null;
    const template = makeWmtsTemplate(layer, styleName, timeISO);

    return new TileLayer({
      id: `cmems-wmts-${layer}`,
      data: template,
      minZoom: 0,
      maxZoom: 10,
      tileSize: 256,
      wrapLongitude: true,
      refinementStrategy: "best-available",
      getTileData: async ({ index: { x, y, z }, signal }) => {
        const n = 1 << z; // number of tiles horizontally at zoom z
        const xWrapped = ((x % n) + n) % n; // wrap x to [0, n-1]
        const url = template
          .replace("{z}", String(z))
          .replace("{x}", String(xWrapped))
          .replace("{y}", String(y));
        const res = await fetch(url, { signal, mode: "cors" });
        if (!res.ok) throw new Error("WMTS tile fetch failed");
        const blob = await res.blob();
        return await createImageBitmap(blob);
      },
      renderSubLayers: (props) => {
        const { tile, data: image } = props;
        if (!image) return null;
        const {
          boundingBox: [[west, south], [east, north]],
        } = tile;
        return new BitmapLayer({
          id: `${props.id}-bmp-${tile.id}`,
          bounds: [west, south, east, north],
          image,
          textureParameters: {
            magFilter: "linear",
            minFilter: "linear",
            wrapS: "clamp-to-edge",
            wrapT: "clamp-to-edge",
          },
          opacity,
        });
      },
      maxRequests: 24,
    });
  }, [layer, styleName, timeISO, opacity]);
}
