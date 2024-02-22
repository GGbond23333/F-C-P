import numpy as np
import matplotlib.pyplot as plt
import copy

images = []
Size = 100

grid = np.random.choice([0, 1], size=(Size, Size), p=[0.5, 0.5])
R_grid = grid.tolist()


def refresh_grid(old_grid, size):
    new_grid = []
    for y in range(size):
        new_grid_row = []
        for x in range(size):
            count = 0
            new_value = 0
            for k in range(-1, 2):
                for m in range(-1, 2):
                    count += old_grid[(y + k) % size][(x + m) % size]
            count -= old_grid[y][x]
            if count == 3:
                new_value = 1
            elif old_grid[y][x] == 1 and count == 2:
                new_value = 1
            new_grid_row.append(new_value)
        new_grid.append(new_grid_row)
    return new_grid


def test_refresh_grid():
    assert refresh_grid([[0, 0, 0], [0, 0, 0], [0, 0, 0]], 3)[1][1] == 0, "test 1"
    assert refresh_grid([[1, 1, 1], [1, 0, 1], [1, 0, 1]], 3)[1][1] == 0, "test 2"
    assert refresh_grid([[1, 0, 1], [1, 0, 0], [0, 0, 0]], 3)[1][1] == 1, "test 3"
    assert refresh_grid([[0, 0, 1], [0, 0, 0], [0, 1, 1]], 3)[1][1] == 1, "test 4"
    assert refresh_grid([[0, 0, 0], [0, 1, 0], [0, 0, 0]], 3)[1][1] == 0, "test 5"
    assert refresh_grid([[0, 1, 1], [0, 1, 0], [0, 0, 0]], 3)[1][1] == 1, "test 6"


test_refresh_grid()


for picture in range(50):
    image = np.zeros((Size, Size), dtype=np.int8)
    up_dated = refresh_grid(R_grid, Size)
    for step1 in range(Size):
        grid2 = R_grid[step1]
        image[step1, :] = np.array(grid2)
        R_grid = copy.deepcopy(up_dated)
    images.append(image)

for i, data_update in enumerate(images):
    ax = plt.axes()
    ax.set_axis_off()
    ax.imshow(data_update, cmap='plasma')
    ax.set_title(f"step{i+1}")
    plt.pause(0.1)
    plt.clf()
