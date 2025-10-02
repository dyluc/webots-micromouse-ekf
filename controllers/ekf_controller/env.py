# define individual grid cell
class GridCell:
    def __init__(self):
        self.tw = False # Top Wall
        self.bw = False # Bottom Wall
        self.rw = False # Right Wall
        self.lw = False # Left Wall

# define environment as grid
class WalledGrid:
    # map size is 2.8m x 2.8m, tileSize is 0.18 (16x16 map)
    def __init__(self, r=16, c=16):
        self.grid = [[GridCell() for _ in range(c)] for _ in range(r)]
        self.tile_size = 0.18 # metres

    def _get_grid_indices(self, x, y):
        i, j = int(y / self.tile_size), int(x / self.tile_size)
        return i, j
    
    # function to update grid cell walls
    def update_grid(self, x, y, ds):
        distance_threshold = 0.06 # readings found from stepping through sim
        i, j = self._get_grid_indices(x, y)
        cell = self.grid[i][j]

        # front, right, bottom, left
        cell.tw = ds[0] < distance_threshold
        cell.rw = ds[1] < distance_threshold
        cell.bw = ds[2] < distance_threshold
        cell.lw = ds[3] < distance_threshold

    def get_grid(self):
        return self.grid
    