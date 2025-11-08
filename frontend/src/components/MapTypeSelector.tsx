import type { MapType } from "./WaveMap";
import { Card, CardContent } from "./ui/card";
import { Label } from "./ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select";
import type { WmtsLayer, WmtsStyle } from "@/hooks/useWmtsLayer";

const SUPPORTED_STYLES: readonly WmtsStyle[] = [
  "cmap:amp",
  "cmap:amp,logScale",
  "cmap:tempo",
  "cmap:tempo,logScale",
  "cmap:cyclic",
  "cmap:cyclic,logScale",
] as const;

function isWmtsStyle(s: string): s is WmtsStyle {
  return (SUPPORTED_STYLES as readonly string[]).includes(s);
}

const SUPPORTED_LAYERS: readonly WmtsLayer[] = [
  "VHM0",
  "VCMX",
  "VHM0_SW1",
  "VHM0_SW2",
  "VHM0_WW",
  "VMDR",
  "VMDR_SW1",
  "VMDR_SW2",
  "VMDR_WW",
  "VMXL",
  "VPED",
  "VSDX",
  "VSDY",
  "VTM01_SW1",
  "VTM01_SW2",
  "VTM01_WW",
  "VTM02",
  "VTM10",
  "VTPK",
];

function isWmtsLayer(l: string): l is WmtsLayer {
  return (SUPPORTED_LAYERS as readonly string[]).includes(l);
}

type Props = {
  mapType: MapType;
  setMapType: (mode: MapType) => void;
};

export function MapTypeSelector({ mapType, setMapType }: Props) {
  const handleStyleChange = (nextStyle: string) => {
    if (!isWmtsStyle(nextStyle)) return;
    if (nextStyle !== mapType.styleName) {
      setMapType({ layer: mapType.layer, styleName: nextStyle });
    }
  };

  const handleLayerChange = (nextLayer: string) => {
    if (!isWmtsLayer(nextLayer)) return;
    if (nextLayer !== mapType.layer) {
      setMapType({ layer: nextLayer, styleName: mapType.styleName });
    }
  };

  return (
    <div className="absolute top-2 left-2 z-20 flex gap-2">
      <Card className="p-2">
        <CardContent className="flex items-center gap-2 px-1">
          <Label htmlFor="wmts-layer" className="text-xs">
            Layer
          </Label>

          <Select value={mapType.layer} onValueChange={handleLayerChange}>
            <SelectTrigger id="wmts-layer" className="h-9">
              <SelectValue placeholder="Select layer" />
            </SelectTrigger>
            <SelectContent>
              {SUPPORTED_LAYERS.map((l) => (
                <SelectItem key={l} value={l}>
                  {l}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </CardContent>
      </Card>

      <Card className="p-2">
        <CardContent className="flex items-center gap-2 px-1">
          <Label htmlFor="wmts-style" className="text-xs">
            Style
          </Label>

          <Select value={mapType.styleName} onValueChange={handleStyleChange}>
            <SelectTrigger id="wmts-style" className="h-9">
              <SelectValue placeholder="Select style" />
            </SelectTrigger>
            <SelectContent>
              {SUPPORTED_STYLES.map((s) => (
                <SelectItem key={s} value={s}>
                  {s}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </CardContent>
      </Card>
    </div>
  );
}
