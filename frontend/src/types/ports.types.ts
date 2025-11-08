export type Port = {
  CITY: string;
  STATE?: string;
  COUNTRY: string;
  LATITUDE: number;
  LONGITUDE: number;
};

export type PortWithId = Port & { id: string };

