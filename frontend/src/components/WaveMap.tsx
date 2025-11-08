import { useEffect, useMemo, useState, useCallback } from "react";
import DeckGL from "@deck.gl/react";
import Map, { NavigationControl } from "react-map-gl/maplibre";
import { PathLayer } from "@deck.gl/layers";
import "maplibre-gl/dist/maplibre-gl.css";

import { TimeControls } from "./TimeControls";
import type { HoverInfo, ViewState } from "@/types/waves.types";
import { LoadingIndicator } from "./LoadingIndicator";
import { WaveLegend } from "./WaveLegend";
import { useTimePlayback } from "@/hooks/useTimePlayback";
import { VectorTooltip } from "./VectorTooltip";
import { MapTypeSelector } from "./MapTypeSelector";
import {
  useCmemsWmtsLayer,
  type WmtsLayer,
  type WmtsStyle,
} from "@/hooks/useWmtsLayer";
import { useAvailableTimes } from "@/hooks/useAvailableTimes";
import { ScatterplotLayer } from "@deck.gl/layers";
import { usePorts } from "@/hooks/usePorts";
import type { PortWithId } from "@/types/ports.types";
import { PortSearch } from "./PortSearch";
import { Card, CardContent } from "./ui/card";
import { Button } from "./ui/button";

type Props = {
  apiBaseUrl?: string;
  mapStyle?: string;
  initialViewState?: Partial<ViewState>;
};

const DEFAULT_VIEW_STATE: ViewState = {
  latitude: 20,
  longitude: 0,
  zoom: 2,
  bearing: 0,
  pitch: 0,
};

export type MapType = {
  layer: WmtsLayer;
  styleName: WmtsStyle;
};

const DEFAULT_MAP_STYLE =
  "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json";

export function WaveMap({
  apiBaseUrl = "http://localhost:5000",
  mapStyle = DEFAULT_MAP_STYLE,
  initialViewState,
}: Props) {
  const [mapType, setMapType] = useState<MapType>({
    layer: "VHM0",
    styleName: "cmap:amp",
  });

  const [viewState, setViewState] = useState<ViewState>({
    ...DEFAULT_VIEW_STATE,
    ...initialViewState,
  });
  const [hoverInfo, setHoverInfo] = useState<HoverInfo | null>(null);
  const { ports, isLoading: portsLoading } = usePorts();
  const [fromPort, setFromPort] = useState<PortWithId | null>(null);
  const [toPort, setToPort] = useState<PortWithId | null>(null);
  const [routeData, setRouteData] = useState<{
    route: [number, number][];
    distance_km: number;
    waypoints: number;
    from: PortWithId;
    to: PortWithId;
  } | null>(null);
  const [routeLoading, setRouteLoading] = useState(false);
  const [routeError, setRouteError] = useState<string | null>(null);

  // Time management
  const { availableTimes, selectedTime, setSelectedTime, isLoading } =
    useAvailableTimes(apiBaseUrl);

  // Playback controls
  const { isPlaying, togglePlayPause, stepForward, stepBackward } =
    useTimePlayback({
      availableTimes,
      selectedTime,
      onTimeChange: setSelectedTime,
    });

  // Handle hover
  const handleHover = useCallback((info: any) => {
    setHoverInfo(info.object ? info : null);
  }, []);

  useEffect(() => {
    if (routeError) {
      const timer = setTimeout(() => {
        setRouteError(null);
      }, 6000);

      return () => clearTimeout(timer);
    }
  }, [routeError]);

  const generateRoute = async () => {
    if (!fromPort || !toPort) {
      setRouteError("Please select both start and end ports");
      return;
    }

    setRouteLoading(true);
    setRouteError(null);

    try {
      const response = await fetch(`${apiBaseUrl}/api/route`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          from: fromPort.CITY,
          to: toPort.CITY,
          time: selectedTime
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Failed to generate route");
      }

      const data = await response.json();
      setRouteData(data);

      const lats = data.route.map((p: [number, number]) => p[0]);
      const lons = data.route.map((p: [number, number]) => p[1]);
      const minLat = Math.min(...lats);
      const maxLat = Math.max(...lats);
      const minLon = Math.min(...lons);
      const maxLon = Math.max(...lons);

      setViewState({
        latitude: (minLat + maxLat) / 2,
        longitude: (minLon + maxLon) / 2,
        zoom: 3,
        bearing: 0,
        pitch: 0,
      });
    } catch (err) {
      setRouteError(err instanceof Error ? err.message : "Other error");
    } finally {
      setRouteLoading(false);
    }
  };

  const clearRoute = () => {
    setRouteData(null);
    setFromPort(null);
    setToPort(null);
    setRouteError(null);
  };

  const analyticLayer = useCmemsWmtsLayer({
    timeISO: selectedTime,
    layer: mapType.layer,
    styleName: mapType.styleName,
    opacity: 0.8,
  });

  const portsLayer = useMemo(() => {
    if (!ports.length) return null;
    return new ScatterplotLayer<PortWithId>({
      id: "ports-scatter",
      data: ports,
      pickable: false,
      getPosition: (d) => [d.LONGITUDE, d.LATITUDE],
      getFillColor: [90, 90, 90, 160],
      radiusMinPixels: 1.5,
      radiusMaxPixels: 3,
    });
  }, [ports]);

  const fromLayer = useMemo(() => {
    if (!fromPort) return null;
    return new ScatterplotLayer<PortWithId>({
      id: "from-port",
      data: [fromPort],
      pickable: false,
      getPosition: (d) => [d.LONGITUDE, d.LATITUDE],
      getFillColor: [34, 197, 94, 255], // green
      radiusMinPixels: 6,
      radiusMaxPixels: 10,
    });
  }, [fromPort]);

  const toLayer = useMemo(() => {
    if (!toPort) return null;
    return new ScatterplotLayer<PortWithId>({
      id: "to-port",
      data: [toPort],
      pickable: false,
      getPosition: (d) => [d.LONGITUDE, d.LATITUDE],
      getFillColor: [239, 68, 68, 255], // red
      radiusMinPixels: 6,
      radiusMaxPixels: 10,
    });
  }, [toPort]);

  const routeLayer = useMemo(() => {
    if (!routeData) return null;
    return new PathLayer({
      id: "route-path",
      data: [
        {
          path: routeData.route.map((p) => [p[1], p[0]]),
          color: [255, 255, 0],
        },
      ],
      getPath: (d: any) => d.path,
      getColor: (d: any) => d.color,
      getWidth: 4,
      widthMinPixels: 3,
      capRounded: true,
      jointRounded: true,
    });
  }, [routeData]);

  const layers = useMemo(() => {
    return [analyticLayer, portsLayer, fromLayer, toLayer, routeLayer].filter(
      Boolean
    ) as any[];
  }, [analyticLayer, portsLayer, fromLayer, toLayer, routeLayer]);

  return (
    <div className="relative h-screen w-full">
      <DeckGL
        layers={layers}
        viewState={viewState}
        onViewStateChange={({ viewState }: any) => setViewState(viewState)}
        controller={{
          doubleClickZoom: false,
          touchRotate: true,
        }}
      >
        <Map reuseMaps mapStyle={mapStyle}>
          <div className="absolute left-2 bottom-2 md:left-4 md:bottom-4 z-50 pointer-events-auto">
            <NavigationControl visualizePitch showCompass={false} />
          </div>
        </Map>
      </DeckGL>

      {hoverInfo && <VectorTooltip hoverInfo={hoverInfo} />}

      <LoadingIndicator isLoading={isLoading} />

      <MapTypeSelector mapType={mapType} setMapType={setMapType} />

      {/* Port search and selection */}
      {!portsLoading && ports.length > 0 && (
        <PortSearch
          ports={ports}
          onSelectFrom={(p) => {
            setFromPort(p);
            setViewState((vs) => ({
              ...vs,
              latitude: p.LATITUDE,
              longitude: p.LONGITUDE,
              zoom: Math.max(vs.zoom, 6),
            }));
          }}
          onSelectTo={(p) => {
            setToPort(p);
            setViewState((vs) => ({
              ...vs,
              latitude: p.LATITUDE,
              longitude: p.LONGITUDE,
              zoom: Math.max(vs.zoom, 6),
            }));
          }}
        />
      )}
      {fromPort && toPort && (
        <div className="absolute top-25 right-4 z-20">
          <Button
            onClick={generateRoute}
            disabled={routeLoading}
            size="lg"
            className="bg-orange-600 hover:bg-orange-700"
          >
            {routeLoading ? "Generating..." : "Generate Route"}
          </Button>
        </div>
      )}

      {routeData && (
        <Card className="absolute bottom-27 right-4 z-20 w-58">
          <CardContent className="p-3 space-y-2">
            <div className="flex items-center justify-between">
              <h3 className="font-semibold text-sm">Route Info</h3>
              <Button
                onClick={clearRoute}
                variant="ghost"
                size="sm"
                className="h-6 px-2 text-xs"
              >
                Clear
              </Button>
            </div>
            <div className="text-xs space-y-1">
              <div>
                <span className="font-medium">Distance:</span>{" "}
                {routeData.distance_km.toLocaleString()} km
              </div>
              <div>
                <span className="font-medium">From:</span> {routeData.from.CITY}
              </div>
              <div>
                <span className="font-medium">To:</span> {routeData.to.CITY}
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {routeError && (
        <Card className="absolute top-20 right-4 z-20 bg-red-50 border-red-200">
          <CardContent className="p-4">
            <div className="flex items-center justify-between gap-2">
              <p className="text-red-800 text-xs flex-1">{routeError}</p>
              <Button
                onClick={() => setRouteError(null)}
                variant="ghost"
                size="sm"
                className="h-6 w-6 p-0 hover:bg-red-100"
              >
                ✕
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {!isLoading && availableTimes.length > 0 && selectedTime && (
        <>
          <TimeControls
            availableTimes={availableTimes}
            selectedTime={selectedTime}
            isPlaying={isPlaying}
            onTimeChange={setSelectedTime}
            onPlayPause={togglePlayPause}
            onStepForward={stepForward}
            onStepBackward={stepBackward}
          />
          <WaveLegend
            selectedTime={selectedTime}
            layer={mapType.layer}
            styleName={mapType.styleName}
          />
        </>
      )}
    </div>
  );
}
