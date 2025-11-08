import numpy as np
from heapq import heappush, heappop
from land_mask import get_land_mask, latlon_to_grid, grid_to_latlon

def distance(lat1, lon1, lat2, lon2):
    # calculate distance using haversine formula
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return np.degrees(c)

def grid_distance(point1, point2, width, height):
    # calculate distance between grid cells using their lat and lon
    lat1, lon1 = grid_to_latlon(point1[0], point1[1], width, height)
    lat2, lon2 = grid_to_latlon(point2[0], point2[1], width, height)
    return distance(lat1, lon1, lat2, lon2)

def get_adjacent_cells(row, col, grid_height, grid_width):
    neighbors = []
    for row_change in [-1, 0, 1]:
        for col_change in [-1, 0, 1]:
            if row_change == 0 and col_change == 0:
                continue 
            
            new_row = row + row_change
            new_col = col + col_change
            
            if 0 <= new_row < grid_height and 0 <= new_col < grid_width:
                neighbors.append((new_row, new_col))
    
    return neighbors

def find_nearest_ocean(start_cell, land_grid, width, height, max_search_radius=20):
    # find nearest ocean cell from port within 20 grids
    height_grid, width_grid = land_grid.shape
    
    # return if already in ocean
    if land_grid[start_cell] == 0:
        return start_cell
    
    best_ocean_cell = None
    best_distance = float('inf')
    
    search_radius = 1
    checked = set()
    
    while search_radius <= max_search_radius:
        # check cells at 20 radius
        for dr in range(-search_radius, search_radius + 1):
            for dc in range(-search_radius, search_radius + 1):
                row = start_cell[0] + dr
                col = start_cell[1] + dc
                
                if (row, col) in checked:
                    continue
                    
                checked.add((row, col))
                
                if 0 <= row < height_grid and 0 <= col < width_grid:
                    if land_grid[row, col] == 0:
                        dist = grid_distance(start_cell, (row, col), width, height)
                        if dist < best_distance:
                            best_distance = dist
                            best_ocean_cell = (row, col)
        
        if best_ocean_cell is not None:
            return best_ocean_cell
        search_radius += 1
    return best_ocean_cell

def calculate_wave_priority(grid_cell, wave_data, width, height):
    row, col = grid_cell
    
    try:
        # get wave height at a specific cell
        wave_height = wave_data[row, col]
        
        if np.isnan(wave_height):
            return 0.0
        
        # avoid waves > 6m (dangerous conditions)
        if wave_height > 6.0:
            return (wave_height - 6.0) * 1.0 # increase penalty for higher waves
        return 0.0
        
    except (IndexError, KeyError):
        return 0.0 # no penalty if out of bounds

def find_path(start_lat, start_lon, end_lat, end_lon, wave_data):
    land_grid = get_land_mask()
    height, width = land_grid.shape
    
    start = latlon_to_grid(start_lat, start_lon, width, height)
    goal = latlon_to_grid(end_lat, end_lon, width, height)
    
    original_start = start
    original_goal = goal
    
    # move to nearest ocean if on land
    if land_grid[start] == 1:
        start = find_nearest_ocean(start, land_grid, width, height)
        if start is None:
            return None
    
    if land_grid[goal] == 1:
        goal = find_nearest_ocean(goal, land_grid, width, height)
        if goal is None:
            return None

    to_check = []
    heappush(to_check, (0, start))
    
    came_from = {}  
    cost_so_far = {start: 0}
    
    while to_check:
        current = heappop(to_check)[1]
        
        if current == goal:
            return build_route(came_from, current, original_start, original_goal, 
                             width, height, land_grid)
        
        for next_cell in get_adjacent_cells(current[0], current[1], height, width):
            if land_grid[next_cell] == 1:
                continue
            
            move_cost = grid_distance(current, next_cell, width, height)
            wave_penalty = calculate_wave_priority(next_cell, wave_data, width, height)
            move_cost += wave_penalty
            new_cost = cost_so_far[current] + move_cost
            
            if next_cell not in cost_so_far or new_cost < cost_so_far[next_cell]:
                cost_so_far[next_cell] = new_cost
                lat_next, lon_next = grid_to_latlon(next_cell[0], next_cell[1], width, height)
                lat_goal, lon_goal = grid_to_latlon(goal[0], goal[1], width, height)
                heuristic = distance(lat_next, lon_next, lat_goal, lon_goal)
                priority = new_cost + heuristic
                heappush(to_check, (priority, next_cell))
                came_from[next_cell] = current
    return None

def build_route(came_from, end, original_start, original_goal, width, height, land_grid):
    path = [end]
    current = end
    
    while current in came_from:
        current = came_from[current]
        path.append(current)
    
    path.reverse()
    
    route = []
    for row, col in path:
        lat, lon = grid_to_latlon(row, col, width, height)
        route.append([lat, lon])
    
    start_lat, start_lon = grid_to_latlon(original_start[0], original_start[1], width, height)
    end_lat, end_lon = grid_to_latlon(original_goal[0], original_goal[1], width, height)
    
    if [start_lat, start_lon] != route[0]:
        route.insert(0, [start_lat, start_lon])
    
    if [end_lat, end_lon] != route[-1]:
        route.append([end_lat, end_lon])
    
    return route 


