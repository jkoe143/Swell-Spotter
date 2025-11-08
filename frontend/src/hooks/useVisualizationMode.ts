import type { VisualizationMode } from "@/types/waves.types";
import { useState, useMemo } from "react";

export function useVisualizationMode(currentZoom: number) {
  const [mode, setMode] = useState<VisualizationMode>("surface");

  // Auto-switch based on zoom if in auto mode
  const effectiveMode = useMemo(() => {
    // You can add auto-switching logic here
    // For now, respect user choice
    return mode;
  }, [mode, currentZoom]);

  return {
    mode: effectiveMode,
    setMode,
  };
}
