import numpy as np
from land_mask import generate_land_mask
import pickle

print("Generating land mask...")
mask = generate_land_mask()

print(f"Land mask generated: {mask.shape}")
print(f"Land cells: {(mask == 1).sum()}, Ocean cells: {(mask == 0).sum()}")

# saves numpy array to python pickle file
with open('data/land_mask.pkl', 'wb') as f:
    pickle.dump(mask, f)

print("Land mask saved to data/land_mask.pkl.")