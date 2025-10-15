import pandas as pd
import numpy as np

def readData(filepath):
    df = pd.read_csv(filepath)

    latitudes = sorted(df['latitude'].unique())
    longitudes = sorted(df['longitude'].unique())

    arr = np.zeros((len(latitudes), len(longitudes)))

    for _, row in df.iterrows():
        latitudeIdx = latitudes.index(row['latitude'])
        longitudeIdx = longitudes.index(row['longitude'])
        arr[latitudeIdx, longitudeIdx] = row['land']
        
    return arr

def main():
    testMap = readData("testData.csv")
    print(testMap)
if __name__ == "__main__":
    main()