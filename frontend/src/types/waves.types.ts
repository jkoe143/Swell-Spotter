export type BuoyPoint = {
  id: string;
  lat: number;
  lon: number;
  waveHeight: number;
};

export type Route = {
  name: string;
  path: [longitude: number, latitude: number][];
};

export type HoverInfo = {
  x: number;
  y: number;
  object: any;
  layer?: any;
};

export type ViewState = {
  latitude: number;
  longitude: number;
  zoom: number;
  bearing: number;
  pitch: number;
};

export type TileJSON = {
  tilejson: string;
  name: string;
  description?: string;
  version: string;
  scheme: string;
  tiles: string[];
  minzoom: number;
  maxzoom: number;
  bounds: [number, number, number, number];
  center?: [number, number, number];
  format: string;
  encoding?: {
    type: string;
    scale: number;
    offset: number;
    units: string;
    formula: string;
  };
  vector_layers?: Array<{
    id: string;
    description: string;
    fields: Record<string, string>;
    minzoom: number;
    maxzoom: number;
  }>;
  quantization?: {
    scale: number;
    offset: number;
    units: string;
  };
  attribution: string;
  dataset_id: string;
  variable: string;
  step_hours: number;
};

export type Capabilities = {
  products: {
    analytic: string;
    isobands: string;
    grid: string;
  };
  tilejson: {
    analytic: string;
    isobands: string;
    grid: string;
  };
  dataset_id: string;
  variables: string[];
  step_hours: number;
  coverage: {
    lon: [number, number];
    lat: [number, number];
  };
};

export type VisualizationMode = "surface" | "isobands" | "grid";
