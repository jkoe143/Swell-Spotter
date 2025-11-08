import { useDebouncedValue } from "@/hooks/useDebouncedValue";
import { Play, Pause, SkipBack, SkipForward, Loader2 } from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";
import { Slider } from "./ui/slider";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "./ui/tooltip";
import { Card, CardContent } from "./ui/card";
import { Button } from "./ui/button";

type Props = {
  availableTimes: string[];
  selectedTime: string;
  isPlaying: boolean;
  isLoading?: boolean;
  onTimeChange: (time: string) => void;
  onPlayPause: () => void;
  onStepForward: () => void;
  onStepBackward: () => void;
};

function clampIndex(n: number, max: number) {
  if (max < 0) return 0;
  return Math.min(Math.max(Math.round(n), 0), max);
}

export function TimeControls({
  availableTimes,
  selectedTime,
  isPlaying,
  isLoading = false,
  onTimeChange,
  onPlayPause,
  onStepForward,
  onStepBackward,
}: Props) {
  const hasTimes = availableTimes.length > 0;
  const maxIndex = availableTimes.length - 1;

  const currentIndex = useMemo(() => {
    if (!hasTimes || !selectedTime) return -1;
    return availableTimes.indexOf(selectedTime);
  }, [availableTimes, hasTimes, selectedTime]);

  const initialIndex = useMemo(() => {
    if (!hasTimes) return 0;
    if (currentIndex >= 0) return currentIndex;
    return 0;
  }, [hasTimes, currentIndex]);

  const [previewIndex, setPreviewIndex] = useState<number>(initialIndex);
  const debouncedIndex = useDebouncedValue(previewIndex, 250);

  // Keep local previewIndex in sync with external selectedTime or times list
  useEffect(() => {
    if (!hasTimes) return;
    const idx =
      currentIndex >= 0 ? currentIndex : clampIndex(previewIndex, maxIndex); // safe fallback
    setPreviewIndex(idx);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hasTimes, currentIndex, maxIndex]);

  // Apply debounced time change while scrubbing
  useEffect(() => {
    if (!hasTimes) return;
    if (debouncedIndex !== currentIndex && availableTimes[debouncedIndex]) {
      onTimeChange(availableTimes[debouncedIndex]);
    }
  }, [debouncedIndex, currentIndex, hasTimes, availableTimes, onTimeChange]);

  const onSliderChange = useCallback(
    (vals: number[]) => {
      const next = clampIndex(vals[0] ?? 0, maxIndex);
      setPreviewIndex(next);
    },
    [maxIndex]
  );

  const isPreviewing =
    hasTimes && currentIndex >= 0 && previewIndex !== currentIndex;
  const previewTime = hasTimes
    ? availableTimes[clampIndex(previewIndex, maxIndex)]
    : selectedTime ?? "";

  const fmtRange = useMemo(
    () =>
      new Intl.DateTimeFormat(undefined, {
        month: "short",
        day: "numeric",
      }),
    []
  );

  const fmtClock = useMemo(
    () =>
      new Intl.DateTimeFormat(undefined, {
        month: "short",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
      }),
    []
  );

  const disableTransport = !hasTimes;
  const canStepBack = hasTimes && currentIndex > 0 && !isPlaying && !isLoading;
  const canStepFwd =
    hasTimes &&
    currentIndex >= 0 &&
    currentIndex < maxIndex &&
    !isPlaying &&
    !isLoading;
  const canPlayPause = hasTimes && !isLoading;

  return (
    <TooltipProvider>
      <Card
        className="absolute bottom-4 left-4 right-4 z-20 md:left-[20rem] max-w-3xl"
        role="group"
        aria-label="Time controls"
        aria-busy={isLoading}
        tabIndex={0}
      >
        <CardContent className="flex items-center gap-2">
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={onStepBackward}
                  disabled={!canStepBack}
                  aria-label="Step backward"
                >
                  <SkipBack className="size-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>Step backward</TooltipContent>
            </Tooltip>

            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="default"
                  size="icon"
                  onClick={onPlayPause}
                  disabled={!canPlayPause}
                  aria-label={isPlaying ? "Pause" : "Play"}
                >
                  {isLoading ? (
                    <Loader2 className="size-4animate-spin" />
                  ) : isPlaying ? (
                    <Pause className="size-4" />
                  ) : (
                    <Play className="size-4" />
                  )}
                </Button>
              </TooltipTrigger>
              <TooltipContent>{isPlaying ? "Pause" : "Play"}</TooltipContent>
            </Tooltip>

            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={onStepForward}
                  disabled={!canStepFwd}
                  aria-label="Step forward"
                >
                  <SkipForward className="size-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>Step forward</TooltipContent>
            </Tooltip>

            <div className="flex-1 px-2">
              <Slider
                min={0}
                max={Math.max(0, maxIndex)}
                step={1}
                value={[clampIndex(previewIndex, maxIndex)]}
                onValueChange={onSliderChange}
                disabled={disableTransport || isPlaying || isLoading}
                aria-label="Scrub time"
              />

              <div className="mt-1 flex items-center justify-between text-xs">
                <span className="text-muted-foreground">
                  {hasTimes
                    ? fmtRange.format(new Date(availableTimes[0]))
                    : "--"}
                </span>

                <span
                  className={[
                    "font-semibold transition-colors",
                    isPreviewing ? "text-primary" : "text-foreground",
                  ].join(" ")}
                >
                  {previewTime ? fmtClock.format(new Date(previewTime)) : "--"}
                  {isPreviewing && (
                    <span className="ml-1 text-xs text-primary/80">
                      (preview)
                    </span>
                  )}
                </span>

                <span className="text-muted-foreground">
                  {hasTimes
                    ? fmtRange.format(
                        new Date(availableTimes[availableTimes.length - 1])
                      )
                    : "--"}
                </span>
              </div>
            </div>

            <div className="min-w-[86px] text-right font-mono text-xs text-muted-foreground">
              {hasTimes
                ? `${(currentIndex >= 0 ? currentIndex : 0) + 1} / ${
                    availableTimes.length
                  }`
                : "0 / 0"}
            </div>
        </CardContent>
      </Card>
    </TooltipProvider>
  );
}
