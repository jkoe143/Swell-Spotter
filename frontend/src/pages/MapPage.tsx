import { Link } from "@tanstack/react-router";
import { WaveMap, type BuoyPoint } from "@/components/WaveMap";

const points: BuoyPoint[] = [
  { id: "46022", lat: 40.7, lon: -124.5, waveHeight: 3.2 },
  { id: "46026", lat: 37.8, lon: -122.8, waveHeight: 2.8 },
  { id: "46025", lat: 33.7, lon: -119.1, waveHeight: 2.1 },
  { id: "46086", lat: 32.5, lon: -118.0, waveHeight: 1.9 },
  { id: "46047", lat: 32.4, lon: -119.5, waveHeight: 2.3 },
  { id: "46054", lat: 34.3, lon: -120.5, waveHeight: 2.6 },
  { id: "46089", lat: 45.9, lon: -125.8, waveHeight: 4.1 },
  { id: "46050", lat: 44.7, lon: -124.5, waveHeight: 3.8 },
  { id: "46041", lat: 47.4, lon: -124.7, waveHeight: 3.5 },
  { id: "44025", lat: 40.3, lon: -73.2, waveHeight: 2.4 },
  { id: "44065", lat: 40.4, lon: -69.3, waveHeight: 3.1 },
  { id: "44008", lat: 40.5, lon: -69.2, waveHeight: 2.9 },
  { id: "44011", lat: 41.1, lon: -66.6, waveHeight: 3.3 },
  { id: "44014", lat: 36.6, lon: -74.8, waveHeight: 2.2 },
  { id: "41010", lat: 28.9, lon: -78.5, waveHeight: 2.0 },
  { id: "41009", lat: 28.5, lon: -80.2, waveHeight: 1.8 },
  { id: "41047", lat: 27.5, lon: -71.5, waveHeight: 2.5 },
  { id: "41048", lat: 31.8, lon: -69.6, waveHeight: 2.7 },
  { id: "44013", lat: 42.3, lon: -70.7, waveHeight: 2.6 },
  { id: "42001", lat: 25.9, lon: -89.7, waveHeight: 1.3 },
  { id: "42002", lat: 26.0, lon: -94.4, waveHeight: 1.5 },
  { id: "42003", lat: 26.0, lon: -85.6, waveHeight: 1.2 },
  { id: "42019", lat: 27.9, lon: -95.4, waveHeight: 1.6 },
  { id: "42020", lat: 26.9, lon: -96.7, waveHeight: 1.4 },
  { id: "42036", lat: 28.5, lon: -84.5, waveHeight: 0.9 },
  { id: "42039", lat: 28.8, lon: -86.0, waveHeight: 1.1 },
  { id: "42040", lat: 29.2, lon: -88.2, waveHeight: 1.3 },
  { id: "51001", lat: 23.4, lon: -162.3, waveHeight: 3.8 },
  { id: "51002", lat: 17.2, lon: -157.8, waveHeight: 3.2 },
  { id: "51003", lat: 19.2, lon: -160.9, waveHeight: 3.5 },
  { id: "51004", lat: 17.5, lon: -152.3, waveHeight: 3.9 },
  { id: "51101", lat: 24.4, lon: -162.1, waveHeight: 4.0 },
  { id: "46080", lat: 57.9, lon: -137.5, waveHeight: 4.5 },
  { id: "46082", lat: 59.7, lon: -143.8, waveHeight: 4.2 },
  { id: "46001", lat: 56.3, lon: -148.1, waveHeight: 5.1 },
  { id: "46083", lat: 58.3, lon: -137.9, waveHeight: 4.7 },
  { id: "41040", lat: 14.6, lon: -53.0, waveHeight: 2.1 },
  { id: "42058", lat: 14.8, lon: -75.1, waveHeight: 1.8 },
];

const MapPage = () => {
  return (
    <div className="min-h-screen bg-white flex flex-col items-center justify-center px-6">
      <h1 className="text-3xl font-bold mb-4 text-blue-800">Wave Map</h1>

      <div className="w-full h-[500px] mb-8 relative rounded-lg shadow-md overflow-hidden">
        <WaveMap points={points} />

        <div className="absolute bottom-4 left-4 bg-white/90 backdrop-blur-sm border border-gray-200 rounded-lg p-3 shadow-md w-64">
          <h2 className="text-sm font-semibold text-gray-800 mb-2">
            Wave Intensity
          </h2>

          <div className="w-full h-4 rounded-md bg-gradient-to-r from-[#440154] via-[#20A386] to-[#FDE725]" />

          <div className="flex justify-between text-xs text-gray-700 mt-2 font-medium">
            <span>Normal</span>
            <span>Calm</span>
            <span>Moderate</span>
            <span>Dangerous</span>
          </div>
        </div>
      </div>

      <Link
        to="/"
        className="inline-block bg-gray-600 hover:bg-gray-700 text-white font-semibold px-6 py-3 rounded shadow transition duration-200"
      >
        Home
      </Link>
    </div>
  );
};

export default MapPage;
