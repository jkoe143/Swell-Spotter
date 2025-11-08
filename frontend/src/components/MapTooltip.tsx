import type { HoverInfo } from "@/types/waves.types";

type Props = {
  hoverInfo: HoverInfo;
};

export function MapTooltip({ hoverInfo }: Props) {
  return (
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
  );
}
