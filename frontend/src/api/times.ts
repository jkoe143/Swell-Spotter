export async function fetchAvailableTimes(
  baseUrl: string,
  start: string,
  end: string,
  limit = 100
): Promise<string[]> {
  const url = `${baseUrl}/times/search?start=${encodeURIComponent(
    start
  )}&end=${encodeURIComponent(end)}&limit=${limit}`;
  const res = await fetch(url, { mode: "cors", credentials: "omit" });
  if (!res.ok) {
    throw new Error(`Failed to fetch times: ${res.status}`);
  }
  const data = await res.json();
  return data.times || [];
}

export async function findNearestTime(
  baseUrl: string,
  time: string
): Promise<string> {
  const url = `${baseUrl}/times/search?time=${encodeURIComponent(time)}`;
  const res = await fetch(url, { mode: "cors", credentials: "omit" });
  if (!res.ok) {
    throw new Error(`Failed to search time: ${res.status}`);
  }
  const data = await res.json();
  return data.nearest_available;
}
