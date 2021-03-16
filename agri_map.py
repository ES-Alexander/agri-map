#!/usr/bin/env python3

from dataclasses import dataclass
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np

@dataclass
class Plant:
    name         : str
    grid_spacing : float
    grid_size    : float
    color        : Tuple[float] # RGB/RGBA, 0-1


def plot_spacing(width, length, grid_size, types):
    ''' Display a 'width'x'length' plot with the given 'grid_size' and 'types'.
    '''
    X,Y = np.meshgrid(np.arange(0, width, grid_size),
                      np.arange(0, length, grid_size))
    Z = np.empty(X.shape)
    for index, type_ in enumerate(types):
        spacing = type_.grid_spacing
        Z[(X % spacing == 0) & (Y % spacing == 0)] = index

    fig, ax = plt.subplots()
    ax.set_title(f'{width}x{length}m plot')
    scale = grid_size * 20
    for index, type_ in enumerate(types):
        locations = (Z == index)
        count = locations.sum()
        label = f'{type_.name} ({count})'
        ax.scatter(X[locations], Y[locations], label=label,
                   color=type_.color, s=type_.grid_size * scale)
    ax.legend()
    ax.axis('equal')
    plt.show()


if __name__ == '__main__':
    description = '''
    Example for cocoa growing with temporary and permanent shade trees.
    '''
    from argparse import ArgumentParser
    parser = ArgumentParser(description=description)
    parser.add_argument('-w', '--width', type=float, default=20,
                        help='width of the plot [m]')
    parser.add_argument('-l', '--length', type=float, default=40,
                        help='length of the plot [m]')
    parser.add_argument('-g', '--grid_size', type=float, default=3,
                        help='grid spacing between tree centers [m]')
    parser.add_argument('-c', '--cocoa', type=float, default=1,
                        help='cocoa grid spacing')
    parser.add_argument('-t', '--temp_shade', type=float, default=2,
                        help='temp shade grid spacing')
    parser.add_argument('-p', '--perm_shade', type=float, default=4,
                        help='permanent shade grid spacing')

    args = parser.parse_args()

    trees = (
        Plant('cocoa', args.cocoa, 1, (0.5,0.25,0)),    # brown
        Plant('temps', args.temp_shade, 1, (0,0,1)),    # blue
        Plant('perms', args.perm_shade, 5, (0,1,0,0.5)) # translucent green
    )

    plot_spacing(args.width, args.length, args.grid_size, trees)
