import { useState, useEffect, useRef, useCallback, useMemo } from "react";

type UseTimePlaybackProps = {
  availableTimes: string[];
  selectedTime: string | null;
  onTimeChange: (time: string) => void;
};

const DEFAULT_INTERVAL_MS = 300;

export function useTimePlayback({
  availableTimes,
  selectedTime,
  onTimeChange,
}: UseTimePlaybackProps) {
  const [isPlaying, setIsPlaying] = useState(false);

  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const indexRef = useRef<number>(-1);

  // Normalize times to epoch ms for stable matching.
  const timesMs = useMemo(
    () =>
      availableTimes
        .map((t) => Date.parse(t))
        .filter((n) => Number.isFinite(n)),
    [availableTimes]
  );

  const findNearestIndex = useCallback(
    (targetMs: number): number => {
      if (!timesMs.length) return -1;
      let bestIdx = 0;
      let bestDiff = Math.abs(timesMs[0] - targetMs);
      for (let i = 1; i < timesMs.length; i++) {
        const d = Math.abs(timesMs[i] - targetMs);
        if (d < bestDiff) {
          bestDiff = d;
          bestIdx = i;
        }
      }
      return bestIdx;
    },
    [timesMs]
  );

  const resolveSelectedIndex = useCallback(() => {
    if (!selectedTime || !timesMs.length) return -1;
    const target = Date.parse(selectedTime);
    const exact = timesMs.indexOf(target);
    if (exact !== -1) return exact;
    return findNearestIndex(target);
  }, [selectedTime, timesMs, findNearestIndex]);

  // Sync internal index when selectedTime or list changes.
  useEffect(() => {
    indexRef.current = resolveSelectedIndex();
  }, [resolveSelectedIndex]);

  // Clear interval util.
  const clearTimer = useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  // Cleanup on unmount.
  useEffect(() => {
    return () => {
      clearTimer();
    };
  }, [clearTimer]);

  // Pause if there are no times.
  useEffect(() => {
    if (!availableTimes.length) {
      clearTimer();
      setIsPlaying(false);
      indexRef.current = -1;
    } else if (indexRef.current >= availableTimes.length) {
      indexRef.current = availableTimes.length - 1;
    }
  }, [availableTimes, clearTimer]);

  const stepBy = useCallback(
    (delta: number, wrap: boolean) => {
      const len = availableTimes.length;
      if (!len) return;

      let idx = indexRef.current;
      if (idx < 0) {
        idx = resolveSelectedIndex();
        if (idx < 0) idx = 0;
      }

      let next = idx + delta;
      if (wrap) {
        next = ((next % len) + len) % len;
      } else {
        next = Math.max(0, Math.min(len - 1, next));
      }

      if (next === idx) return;

      indexRef.current = next;
      const nextTime = availableTimes[next];
      if (nextTime) onTimeChange(nextTime);
    },
    [availableTimes, onTimeChange, resolveSelectedIndex]
  );

  const play = useCallback(() => {
    if (timerRef.current || !availableTimes.length) return;
    // Immediate advance for responsiveness.
    stepBy(1, true);
    timerRef.current = setInterval(() => {
      stepBy(1, true);
    }, DEFAULT_INTERVAL_MS);
    setIsPlaying(true);
  }, [availableTimes, stepBy]);

  const pause = useCallback(() => {
    clearTimer();
    setIsPlaying(false);
  }, [clearTimer]);

  const togglePlayPause = useCallback(() => {
    if (isPlaying) {
      pause();
    } else {
      play();
    }
  }, [isPlaying, play, pause]);

  const stepForward = useCallback(() => {
    // Pause before stepping for predictable UX.
    pause();
    stepBy(1, false);
  }, [pause, stepBy]);

  const stepBackward = useCallback(() => {
    pause();
    stepBy(-1, false);
  }, [pause, stepBy]);

  return {
    isPlaying,
    togglePlayPause,
    stepForward,
    stepBackward,
    play,
    pause,
  };
}
