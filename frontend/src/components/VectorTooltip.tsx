import type { HoverInfo } from "@/types/waves.types";

type Props = {
  hoverInfo: HoverInfo;
};

export function VectorTooltip({ hoverInfo }: Props) {
  const props = hoverInfo.object?.properties;
  if (!props) return null;

  return (
    <div
      className="absolute z-20 pointer-events-none bg-gray-900/95 text-white px-3 py-2 rounded shadow-lg text-sm max-w-xs"
      style={{
        left: hoverInfo.x + 10,
        top: hoverInfo.y + 10,
      }}
    >
      {/* Isobands layer */}
      {props.label && <div className="font-semibold mb-1">{props.label}</div>}
      {props.min !== undefined && props.max !== undefined && (
        <div>
          Range: {props.min.toFixed(2)} - {props.max.toFixed(2)} m
        </div>
      )}

      {/* Grid layer */}
      {props.value !== undefined && (
        <div className="font-semibold">
          Wave Height: {props.value.toFixed(2)} m
        </div>
      )}
      {props.h_q !== undefined && (
        <div className="text-xs text-gray-400 mt-1">
          Quantized: {props.h_q} / 255
        </div>
      )}
      {(props.i !== undefined || props.j !== undefined) && (
        <div className="text-xs text-gray-400">
          Grid: [{props.i ?? "?"}, {props.j ?? "?"}]
        </div>
      )}

      {/* Band ID for isobands */}
      {props.band_id !== undefined && (
        <div className="text-xs text-gray-400 mt-1">Band: {props.band_id}</div>
      )}
    </div>
  );
}
