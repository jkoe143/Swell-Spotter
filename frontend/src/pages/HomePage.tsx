import { Link } from "@tanstack/react-router";

const HomePage = () => {
  return (
    <div className="min-h-screen bg-gradient-to-b from-blue-100 to-blue-200 flex flex-col items-center justify-center px-6">
      <header className="mb-6">
        <h1 className="text-4xl font-bold text-blue-700">Swell Spotter</h1>
      </header>

      <main className="max-w-2xl text-center">
        <img
          src="/ocean-sea.gif"
          alt="Waves splashing"
          className="mx-auto mb-6 w-3/4 max-w-sm rounded shadow-lg"
        />

        <p className="text-lg text-black leading-relaxed mb-9">
          Swell Spotter visualizes real‑time ocean wave conditions using
          open‑source buoy and Copernicus Marine data APIs. Wave height and
          intensity are displayed on a dynamic, color‑coded topographic map
          using a Viridis gradient: dark purple for the lowest waves, through
          blue and teal for moderate conditions, to green and yellow for the
          highest or most intense waves. To use the map, simply drag around
          using the mouse and zoom in and out using the scroll wheel.
        </p>

        <Link
          to="/map"
          className="inline-block bg-blue-700 hover:bg-blue-800 text-white font-semibold px-6 py-3 rounded shadow transition duration-200"
        >
          View Map
        </Link>
      </main>
    </div>
  );
};

export default HomePage;
