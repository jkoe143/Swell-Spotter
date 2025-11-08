// components/WaveLegend.tsx
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { useCmemsLegend } from "@/hooks/useCmemsLegend";
import type { WmtsLayer, WmtsStyle } from "@/hooks/useWmtsLayer";

type Props = {
  selectedTime: string | null;
  layer: WmtsLayer;
  styleName: WmtsStyle;
};

export function WaveLegend({ selectedTime, layer, styleName }: Props) {
  // Fetch the exact legend JSON used by the WMTS tiles
  const { data, isLoading, isError, error } = useCmemsLegend({
    layer,
    styleName,
    maxStops: 128,
  });

  const title = data?.variableName ?? "Legend";
  const units = data?.units;
  const min = data?.min ?? 0;
  const max = data?.max ?? 1;
  const gradient = data?.gradientCss;

  return (
    <Card className="absolute bottom-4 left-4 w-72 z-20">
      <CardHeader>
        <CardTitle className="text-sm font-medium">
          {title}
          {units ? (
            <span className="text-muted-foreground"> ({units})</span>
          ) : null}
        </CardTitle>
      </CardHeader>
      <CardContent>
        {isLoading && (
          <div className="space-y-2">
            <Skeleton className="h-4 w-40" />
            <Skeleton className="h-4 w-full" />
            <div className="flex justify-between text-xs text-muted-foreground">
              <Skeleton className="h-3 w-10" />
              <Skeleton className="h-3 w-10" />
            </div>
          </div>
        )}

        {isError && (
          <Alert variant="destructive">
            <AlertDescription className="text-xs">
              Failed to load legend
              {error instanceof Error ? `: ${error.message}` : ""}
            </AlertDescription>
          </Alert>
        )}

        {!isLoading && !isError && gradient && (
          <>
            <div
              className="w-full h-4 rounded"
              style={{ backgroundImage: gradient }}
              aria-label="Legend color bar"
              role="img"
            />
            <div className="mt-1 flex items-center justify-between text-xs text-muted-foreground">
              <span>{Number.isFinite(min) ? min.toFixed(2) : "-"}</span>
              <span>{Number.isFinite(max) ? max.toFixed(2) : "-"}</span>
            </div>
          </>
        )}

        {selectedTime && (
          <div className="mt-3 border-t pt-2">
            <div className="text-xs text-muted-foreground">Current time</div>
            <div className="text-sm font-medium">
              {new Date(selectedTime).toLocaleString()}
            </div>
          </div>
        )}

        <div className="mt-2 text-xs text-muted-foreground capitalize">
          Mode: {layer + " / " + styleName}
        </div>
      </CardContent>
    </Card>
  );
}
