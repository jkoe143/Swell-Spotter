import { useEffect, useMemo, useRef, useState } from "react";
import { Input } from "@/components/ui/input";
import type { PortWithId } from "@/types/ports.types";

type Props = {
  ports: PortWithId[];
  onSelectFrom: (p: PortWithId) => void;
  onSelectTo: (p: PortWithId) => void;
};

function formatPort(p: PortWithId) {
  return [p.CITY, p.STATE, p.COUNTRY].filter(Boolean).join(", ");
}

export function PortSearch({ ports, onSelectFrom, onSelectTo }: Props) {
  const [fromQuery, setFromQuery] = useState("");
  const [toQuery, setToQuery] = useState("");
  const [active, setActive] = useState<"from" | "to" | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);

  // Close dropdowns on outside click
  useEffect(() => {
    function onDocClick(e: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setActive(null);
      }
    }
    document.addEventListener("mousedown", onDocClick);
    return () => document.removeEventListener("mousedown", onDocClick);
  }, []);

  const normalized = useMemo(
    () =>
      ports.map((p) => ({
        key: p.id,
        label: formatPort(p),
        p,
        hay: `${p.CITY} ${p.STATE ?? ""} ${p.COUNTRY}`.toLowerCase(),
      })),
    [ports]
  );

  const fromMatches = useMemo(() => {
    if (!fromQuery.trim()) return [] as typeof normalized;
    const q = fromQuery.toLowerCase();
    return normalized.filter((n) => n.hay.includes(q)).slice(0, 12);
  }, [normalized, fromQuery]);

  const toMatches = useMemo(() => {
    if (!toQuery.trim()) return [] as typeof normalized;
    const q = toQuery.toLowerCase();
    return normalized.filter((n) => n.hay.includes(q)).slice(0, 12);
  }, [normalized, toQuery]);

  return (
    <div
      ref={containerRef}
      className="absolute right-2 top-2 md:top-4 md:right-4 z-20 w-[min(520px,calc(100%-1rem))] bg-white/90 backdrop-blur rounded-md shadow p-2 border border-gray-200"
    >
      <div className="flex flex-col md:flex-row gap-2">
        <div className="relative flex-1">
          <label className="block text-xs text-gray-600 mb-1">From</label>
          <Input
            placeholder="Search port (city, state, country)"
            value={fromQuery}
            onChange={(e) => {
              setFromQuery(e.target.value);
              setActive("from");
            }}
            onFocus={() => setActive("from")}
          />
          {active === "from" && fromMatches.length > 0 && (
            <ul className="absolute mt-1 max-h-64 overflow-auto w-full bg-white border border-gray-200 rounded-md shadow">
              {fromMatches.map((m) => (
                <li
                  key={m.key}
                  className="px-3 py-2 hover:bg-gray-100 cursor-pointer text-sm"
                  onClick={() => {
                    setFromQuery(m.label);
                    setActive(null);
                    onSelectFrom(m.p);
                  }}
                >
                  {m.label}
                </li>
              ))}
            </ul>
          )}
        </div>

        <div className="relative flex-1">
          <label className="block text-xs text-gray-600 mb-1">To</label>
          <Input
            placeholder="Search port (city, state, country)"
            value={toQuery}
            onChange={(e) => {
              setToQuery(e.target.value);
              setActive("to");
            }}
            onFocus={() => setActive("to")}
          />
          {active === "to" && toMatches.length > 0 && (
            <ul className="absolute mt-1 max-h-64 overflow-auto w-full bg-white border border-gray-200 rounded-md shadow">
              {toMatches.map((m) => (
                <li
                  key={m.key}
                  className="px-3 py-2 hover:bg-gray-100 cursor-pointer text-sm"
                  onClick={() => {
                    setToQuery(m.label);
                    setActive(null);
                    onSelectTo(m.p);
                  }}
                >
                  {m.label}
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </div>
  );
}
