import { Link } from "@tanstack/react-router";
import { WaveMap } from "@/components/WaveMap";
import { Button } from "@/components/ui/button";

const MapPage = () => {
  return (
    <div className="min-h-screen bg-white flex flex-col items-center justify-center p-4 gap-4">
      <h1 className="text-3xl font-bold text-blue-800">Wave Map</h1>

      <div className="w-full max-h-lvh relative rounded-lg shadow-md overflow-hidden">
        <WaveMap />
      </div>

      <Link to="/">
        <Button size="lg" className="inline-block bg-gray-600 hover:bg-gray-700 text-white font-semibold">
          Home
        </Button>
      </Link>
    </div>
  );
};

export default MapPage;
