import { fetchCapabilities, fetchTileJSON } from "@/api/capabilities";
import { useQuery } from "@tanstack/react-query";

export function useCapabilities(apiBaseUrl: string) {
  const { data: capabilities, isLoading: capabilitiesLoading } = useQuery({
    queryKey: ["capabilities", apiBaseUrl],
    queryFn: () => fetchCapabilities(apiBaseUrl),
    staleTime: 10 * 60 * 1000, // 10 minutes
  });

  const { data: analyticTileJSON, isLoading: analyticLoading } = useQuery({
    queryKey: ["tilejson", "analytic", apiBaseUrl],
    queryFn: () =>
      fetchTileJSON(`${apiBaseUrl}/tilejson/waves-analytic.json`),
    enabled: !!capabilities,
    staleTime: 10 * 60 * 1000,
  });

  const { data: isobandsTileJSON, isLoading: isobandsLoading } = useQuery({
    queryKey: ["tilejson", "isobands", apiBaseUrl],
    queryFn: () =>
      fetchTileJSON(`${apiBaseUrl}/tilejson/waves-isobands.json`),
    enabled: !!capabilities,
    staleTime: 10 * 60 * 1000,
  });

  const { data: gridTileJSON, isLoading: gridLoading } = useQuery({
    queryKey: ["tilejson", "grid", apiBaseUrl],
    queryFn: () => fetchTileJSON(`${apiBaseUrl}/tilejson/waves-grid.json`),
    enabled: !!capabilities,
    staleTime: 10 * 60 * 1000,
  });

  return {
    capabilities,
    analyticTileJSON,
    isobandsTileJSON,
    gridTileJSON,
    isLoading:
      capabilitiesLoading ||
      analyticLoading ||
      isobandsLoading ||
      gridLoading,
  };
}