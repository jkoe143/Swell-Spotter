import { fetchAvailableTimes } from "@/api/times";
import { useQuery } from "@tanstack/react-query";
import { useState, useEffect } from "react";

export function useAvailableTimes(
  apiBaseUrl: string,
  daysBack: number = 7
) {
  const [selectedTime, setSelectedTime] = useState<string | null>(null);

  const { data: availableTimes, isLoading } = useQuery({
    queryKey: ["available-times", apiBaseUrl, daysBack],
    queryFn: async () => {
      const end = new Date();
      const start = new Date(end.getTime() - daysBack * 24 * 60 * 60 * 1000);
      return fetchAvailableTimes(
        apiBaseUrl,
        start.toISOString(),
        end.toISOString(),
        200
      );
    },
    staleTime: 5 * 60 * 1000, // 5 minutes
  });

  // Initialize with most recent time
  useEffect(() => {
    if (availableTimes && availableTimes.length > 0 && !selectedTime) {
      setSelectedTime(availableTimes[availableTimes.length - 1]);
    }
  }, [availableTimes, selectedTime]);

  return {
    availableTimes: availableTimes || [],
    selectedTime,
    setSelectedTime,
    isLoading,
  };
}