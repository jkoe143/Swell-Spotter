import type { Capabilities, TileJSON } from "@/types/waves.types";

export async function fetchCapabilities(
  baseUrl: string
): Promise<Capabilities> {
  const res = await fetch(`${baseUrl}/capabilities`, {
    mode: "cors",
    credentials: "omit",
  });
  if (!res.ok) {
    throw new Error(`Failed to fetch capabilities: ${res.status}`);
  }
  return res.json();
}

export async function fetchTileJSON(url: string): Promise<TileJSON> {
  const res = await fetch(url, { mode: "cors", credentials: "omit" });
  if (!res.ok) {
    throw new Error(`Failed to fetch TileJSON: ${res.status}`);
  }
  return res.json();
}
