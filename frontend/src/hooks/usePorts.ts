import { useEffect, useMemo, useState } from "react";
import type { PortWithId } from "@/types/ports.types";

const LOCAL_DATABASE_PATHS = [
  "/database.json",
  "/ports.json",
];

const REMOTE_FALLBACK =
  "https://raw.githubusercontent.com/tayljordan/ports/master/database.json";

export function usePorts() {
  const [ports, setPorts] = useState<PortWithId[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      setIsLoading(true);
      setError(null);

      // Try local public paths first
      const urls = [...LOCAL_DATABASE_PATHS, REMOTE_FALLBACK];

      for (const url of urls) {
        try {
          const res = await fetch(url, { cache: "no-store" });
          if (!res.ok) {
            continue;
          }
          const json = await res.json();
          if (!Array.isArray(json)) continue;

          const withIds: PortWithId[] = json
            .filter((p: any) =>
              typeof p?.LATITUDE === "number" && typeof p?.LONGITUDE === "number"
            )
            .map((p: any, idx: number) => ({
              CITY: String(p?.CITY ?? ""),
              STATE: p?.STATE ? String(p.STATE) : undefined,
              COUNTRY: String(p?.COUNTRY ?? ""),
              LATITUDE: Number(p.LATITUDE),
              LONGITUDE: Number(p.LONGITUDE),
              id: `${p?.CITY ?? ""}-${p?.STATE ?? ""}-${p?.COUNTRY ?? ""}-${idx}`,
            }));

          if (!cancelled) {
            setPorts(withIds);
            setIsLoading(false);
          }
          return;
        } catch (e: any) {
          // Try next URL
          continue;
        }
      }

      if (!cancelled) {
        setError("Failed to load ports dataset");
        setIsLoading(false);
      }
    }

    load();
    return () => {
      cancelled = true;
    };
  }, []);

  const byNameIndex = useMemo(() => {
    const map = new Map<string, PortWithId[]>();
    for (const p of ports) {
      const key = p.CITY.toLowerCase();
      const list = map.get(key) ?? [];
      list.push(p);
      map.set(key, list);
    }
    return map;
  }, [ports]);

  return { ports, byNameIndex, isLoading, error };
}

