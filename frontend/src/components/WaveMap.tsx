import { useMemo, useState } from "react";
import DeckGL from "@deck.gl/react";
import { BitmapLayer, PathLayer, ScatterplotLayer } from "@deck.gl/layers";
import { TileLayer } from "@deck.gl/geo-layers";
import Map, { NavigationControl } from "react-map-gl/maplibre";
import "maplibre-gl/dist/maplibre-gl.css";

export type BuoyPoint = {
  id: string;
  lat: number;
  lon: number;
  waveHeight: number; // meters
};

type Props = {
  points: BuoyPoint[];
  mapStyle?: string;
  tileUrl?: string;
  tileTime?: string;
};

interface Route {
  name: string;
  path: [longitude: number, latitude: number][];
}

const route: Route = {
  name: "Sample Route",
  path: [
    [-125, 50],
    [-120, 45],
  ],
};

const stops = [
  [68, 1, 84],
  [59, 82, 139],
  [33, 145, 140],
  [94, 201, 97],
  [253, 231, 37],
];

function makeViridisLUT(steps = 256): Uint8Array {
  const lut = new Uint8Array(steps * 4);
  for (let i = 0; i < steps; i++) {
    const t = i / (steps - 1);
    const j = Math.min(stops.length - 2, Math.floor(t * (stops.length - 1)));
    const f = t * (stops.length - 1) - j;
    const c0 = stops[j];
    const c1 = stops[j + 1];
    const r = Math.round(c0[0] + (c1[0] - c0[0]) * f);
    const g = Math.round(c0[1] + (c1[1] - c0[1]) * f);
    const b = Math.round(c0[2] + (c1[2] - c0[2]) * f);
    lut[i * 4 + 0] = r;
    lut[i * 4 + 1] = g;
    lut[i * 4 + 2] = b;
    lut[i * 4 + 3] = 220; // alpha
  }
  return lut;
}

const calculateMinMax = (points: BuoyPoint[]) => {
  let minH = Infinity;
  let maxH = -Infinity;
  for (const p of points) {
    const h = p.waveHeight;
    if (h < minH) minH = h;
    if (h > maxH) maxH = h;
  }
  if (!isFinite(minH) || !isFinite(maxH)) {
    minH = 0;
    maxH = 1;
  }
  if (maxH === minH) maxH = minH + 1e-6;
  return { minH, maxH };
};

const calculateColors = (points: BuoyPoint[], maxH: number, minH: number) => {
  const out = new Uint8Array(points.length * 4);
  const range = maxH - minH;
  for (let i = 0; i < points.length; i++) {
    const h = points[i].waveHeight;
    const t = Math.max(0, Math.min(1, (h - minH) / range));
    const idx = (t * 255) | 0;
    out[i * 4 + 0] = VIRIDIS[idx * 4 + 0];
    out[i * 4 + 1] = VIRIDIS[idx * 4 + 1];
    out[i * 4 + 2] = VIRIDIS[idx * 4 + 2];
    out[i * 4 + 3] = VIRIDIS[idx * 4 + 3];
  }
  return out;
};

function buildTileTemplate(base: string, time?: string) {
  if (!time) return base;
  const sep = base.includes("?") ? "&" : "?";
  return `${base}${sep}time=${encodeURIComponent(time)}`;
}

// Converts template http://.../{z}/{x}/{y}.png to a URL
function tileUrlFromTemplate(
  template: string,
  x: number,
  y: number,
  z: number
) {
  return template
    .replace("{z}", String(z))
    .replace("{x}", String(x))
    .replace("{y}", String(y));
}

async function fetchTileAsImage(
  url: string,
  signal?: AbortSignal
): Promise<ImageBitmap | HTMLImageElement> {
  const res = await fetch(url, { mode: "cors", credentials: "omit", signal });
  if (!res.ok) {
    throw new Error(`Tile fetch failed ${res.status} ${res.statusText}`);
  }
  const blob = await res.blob();
  try {
    // Faster to upload to GPU
    // @ts-ignore: createImageBitmap is available in modern browsers
    const bmp = await createImageBitmap(blob);
    return bmp;
  } catch {
    return await new Promise((resolve, reject) => {
      const img = new Image();
      img.crossOrigin = "anonymous";
      img.onload = () => resolve(img);
      img.onerror = (e) => reject(e);
      img.src = URL.createObjectURL(blob);
    });
  }
}

const VIRIDIS = makeViridisLUT(256);

export function WaveMap({ points, mapStyle, tileUrl, tileTime }: Props) {
  const [hoverInfo, setHoverInfo] = useState<{
    x: number;
    y: number;
    object: BuoyPoint;
  } | null>(null);
  const { minH, maxH } = useMemo(() => calculateMinMax(points), [points]);

  const colors = useMemo(
    () => calculateColors(points, maxH, minH),
    [points, minH, maxH]
  );

  const template = useMemo(
    () =>
      buildTileTemplate(
        tileUrl ?? "http://localhost:5000/tiles/waves/{z}/{x}/{y}.png",
        tileTime
      ),
    [tileUrl, tileTime]
  );

  const wavesTileLayer = useMemo(() => {
    return new TileLayer({
      id: "wave-tiles",
      data: template, // keep template here for cache key stability
      minZoom: 0,
      maxZoom: 12,
      tileSize: 256,
      refinementStrategy: "best-available",
      getTileData: async ({ index: { x, y, z }, signal }) => {
        const url = tileUrlFromTemplate(template, x, y, z);
        try {
          const img = await fetchTileAsImage(url, signal);
          return img;
        } catch (e) {
          // Return transparent image on error (keeps map smooth)
          console.warn("Wave tile error", { x, y, z, e });
          const canvas = new OffscreenCanvas(256, 256);
          const ctx = canvas.getContext("2d");
          if (ctx) {
            ctx.clearRect(0, 0, 256, 256);
          }
          return canvas.transferToImageBitmap
            ? canvas.transferToImageBitmap()
            : canvas;
        }
      },
      renderSubLayers: (props) => {
        const { tile, data: image } = props as any;
        if (!image) return null;

        const {
          bbox: { west, south, east, north },
        } = tile;

        return new BitmapLayer({
          id: `${props.id}-bmp-${tile.id}`,
          bounds: [west, south, east, north],
          image,
          textureParameters: {
            magFilter: "linear",
            minFilter: "linear",
          },
          opacity: 0.75,
          pickable: false,
        });
      },
      onTileLoad: (tile) => {},
      onTileError: (err) => {
        console.warn("tile load error", err);
      },
      maxRequests: 24,
    });
  }, [template]);

  const pointLayer = useMemo(() => {
    return new ScatterplotLayer<BuoyPoint>({
      id: "buoy-points",
      data: points,
      getPosition: (d) => [d.lon, d.lat],
      getFillColor: [255, 100, 0, 255],
      radiusUnits: "pixels",
      getRadius: 5,
      stroked: false,
      pickable: true,
      onHover: (info: any) => setHoverInfo(info.object ? info : null),
    });
  }, [points, colors]);

  const routeLayer = useMemo(() => {
    return new PathLayer<Route>({
      id: "route",
      data: [route],
      getPath: (d) => d.path,
      widthMinPixels: 2,
      getColor: [0, 122, 255, 220],
      getWidth: 4,
      widthUnits: "pixels",
      pickable: true,
    });
  }, []);

  const layers = useMemo(
    () => [wavesTileLayer, pointLayer, routeLayer],
    [wavesTileLayer, pointLayer, routeLayer]
  );

  const initialViewState = useMemo(() => {
    return { latitude: 39.5, longitude: -98.35, zoom: 3, bearing: 0, pitch: 0 };
  }, []);

  return (
    <div className="relative h-[calc(100vh-64px)] w-full">
      <DeckGL layers={layers} initialViewState={initialViewState} controller>
        <Map
          reuseMaps
          mapStyle={
            mapStyle ??
            "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"
          }
        >
          <div className="absolute right-2 top-2 z-10">
            <NavigationControl visualizePitch />
          </div>
        </Map>
      </DeckGL>

      <div className="pointer-events-auto absolute left-2 top-2 z-10 flex gap-2"></div>

      <div className="absolute bottom-3 right-3 z-10 rounded bg-white/90 p-3 text-xs shadow">
        <div className="mb-1 font-semibold">Wave height (m)</div>
        <div className="flex items-center gap-2">
          <span>{minH.toFixed(2)}</span>
          <div className="h-2 w-40 bg-gradient-to-r from-[#440154] via-[#20A386] to-[#FDE725]" />
          <span>{maxH.toFixed(2)}</span>
        </div>
      </div>
      {hoverInfo && (
        <div
          className="absolute z-20 pointer-events-none bg-gray-900 text-white px-3 py-2 rounded shadow-lg text-sm"
          style={{
            left: hoverInfo.x + 10,
            top: hoverInfo.y + 10,
          }}
        >
          <div className="font-semibold mb-1">Buoy {hoverInfo.object.id}</div>
          <div>Lat: {hoverInfo.object.lat.toFixed(4)}°</div>
          <div>Lon: {hoverInfo.object.lon.toFixed(4)}°</div>
          <div>Wave Height: {hoverInfo.object.waveHeight.toFixed(2)}m</div>
        </div>
      )}
    </div>
  );
}
