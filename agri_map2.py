#!/usr/bin/env python3

from dataclasses import dataclass
from typing import Tuple
import plotly.graph_objects as go
import numpy as np
import math

@dataclass
class Tree:
    name     : str
    d_trunk  : float # trunk diameter [m]
    d_canopy : float # canopy diameter [m]
    min_dist : float # center-center dist [m] to other relevant Trees
    color    : Tuple[float] # RGB/RGBA, 0-1


@dataclass
class ShadeTree(Tree):
    shade_factor : float # multiplier (sun is mostly not directly overhead)


class Cocoa(Tree):
    def __init__(self, d_trunk=0.5, d_canopy=3, min_dist=None,
                 color=(0.5, 0.25, 0)):
        ''' A cocoa tree.

        'd_trunk' is the trunk diameter [m]. Default 0.5m.
        'd_canopy' is the canopy diameter [m]. Default 3m.
        'min_dist' is the minimum center-center distance [m] to the
            surrounding cocoa trees. If left as None, defaults to
            (d_trunk + d_canopy).
        'color' is the RGB[A] colour used when plotting. Defaults to brown
            (0.5, 0.25, 0) -> 50% red, 25% green, 0% blue.

        '''
        # keep if set, or replace with default
        min_dist = min_dist or (d_trunk + d_canopy)
        super().__init__('Cocoa', d_trunk, d_canopy, min_dist, color)


class TempShade(ShadeTree):
    def __init__(self, d_trunk=0.5, d_canopy=3, min_dist=None,
                 shade_factor=1.3, color=(0, 0, 1)):
        ''' A tree for providing temporary shading of cocoa tree trunks.

        'd_trunk' is the trunk diameter [m]. Default 0.5m.
        'd_canopy' is the canopy diameter [m]. Default 3m.
        'min_dist' is the minimum center-center distance to any surrounding
            trees. If left as None, defaults to (2 * d_trunk).
        'shade_factor' is a multiplier of d_canopy determining where is
            considered to be adequately shaded. Helps to reduce shade
            redundancy for the majority of the day when the sun isn't directly
            overhead. Defaults to 1.3.
        'color' is the RGB[A] colour used when plotting. Defaults to blue
            (0, 0, 1) -> 0% red, 0% green, 100% blue.

        '''
        # keep if set, or replace with default
        min_dist = min_dist or (2 * d_trunk)
        super().__init__('Temp Shade', d_trunk, d_canopy, min_dist, color,
                         shade_factor)


class PermShade(ShadeTree):
    def __init__(self, d_trunk=1, d_canopy=15, min_dist=None,
                 shade_factor=1.3, color=(0, 1, 0)):
        ''' A tree for providing permanent shading of cocoa tree canopies.
        
        'd_trunk' is the trunk diameter [m]. Default 1m.
        'd_canopy' is the canopy diameter [m]. Default 15m.
        'min_dist' is the minimum center-center distance to any surrounding
            trees. If left as None, defaults to (2 * d_trunk).
        'shade_factor' is a multiplier of d_canopy determining where is
            considered to be adequately shaded. Helps to reduce shade
            redundancy for the majority of the day when the sun isn't directly
            overhead. Defaults to 1.3.
        'color' is the RGB[A] colour used when plotting. Defaults to green
            (0, 1, 0) -> 0% red, 100% green, 0% blue.

        '''
        # keep if set, or replace with default
        min_dist = min_dist or (2 * d_trunk)
        super().__init__('Perm Shade', d_trunk, d_canopy, min_dist, color,
                         shade_factor)


@dataclass
class CocoaFarm:
    ''' Create a plot with the given dimensions and tree parameters.

    'dims' are the dimensions of the plot in metres (width, height).
    'cocoa' is a Tree to maximally pack into the plot for optimal yield.
    'temp_shade' is a ShadeTree to minimally pack into the plot such that all
        cocoa trunks are shaded (for during initial growth).
    'perm_shade' is a ShadeTree to minimally pack into the plot such that all
        cocoa canopies are shaded (for long term production).

    '''
    dims: tuple = (100, 100)
    cocoa: Tree = Cocoa()
    temp_shade: ShadeTree = TempShade()
    perm_shade: ShadeTree = PermShade()

    def calculate_and_display(self, verbose=True):
        self.optimise_spacings()
        self.display_results(verbose)

    def optimise_spacings(self):
        dims = self.dims
        cocoa = self.cocoa
        # adjust dimensions to ensure cocoa canopies stay inside the plot
        adjusted_dims = [side - cocoa.d_canopy for side in dims]
        # optimal cocoa spacing comes from flat gaps along the shorter axis
        #  and staggered gaps along the longer one.
        # 1/True if width has more room, else 0/False
        expand_ind = dims[0] >= dims[1]
        fill_ind   = not expand_ind
        # use expansion index to determine even spacing of the main grid
        primary_adjusted = adjusted_dims[expand_ind]
        primary_spacing  = (primary_adjusted /
                           (primary_adjusted // cocoa.min_dist))
        # calculate the offset and spacing for the indented grid
        secondary_spacing  = 2 * math.sqrt(cocoa.min_dist**2 -
                                        (primary_spacing / 2)**2)
        secondary_dist     = dims[fill_ind]
        secondary_adjusted = adjusted_dims[fill_ind]
        secondary_offset   = (secondary_dist - secondary_spacing *
                            (secondary_adjusted // secondary_spacing)) / 2
        # set up spacing and offset variables for both directions
        spacing = (secondary_spacing, primary_spacing)
        cocoa_r = cocoa.d_canopy / 2
        offsets = [cocoa_r] * 2
        half_step = (offsets[fill_ind] + spacing[fill_ind] / 2,
                     offsets[expand_ind] + spacing[expand_ind] / 2)
        half_max = (dims[fill_ind] - cocoa_r, dims[expand_ind] - cocoa_r)

        # create the main and secondary grids
        cocoa_r /= 1.0001 # handle float inaccuracy

        X0,Y0 = np.meshgrid(np.arange(offsets[fill_ind], dims[0] - cocoa_r,
                                        spacing[fill_ind]),
                            np.arange(offsets[expand_ind], dims[1] - cocoa_r,
                                        spacing[expand_ind]))
        X1,Y1 = np.meshgrid(np.arange(half_step[0], half_max[fill_ind],
                                        spacing[fill_ind]),
                            np.arange(half_step[1], half_max[expand_ind],
                                        spacing[expand_ind]))

        self.CX = np.hstack((X0.reshape(-1), X1.reshape(-1)))
        self.CY = np.hstack((Y0.reshape(-1), Y1.reshape(-1)))

    def display_results(self, verbose=True):
        """
        X,Y = np.meshgrid(np.arange(0, width, grid_size),
                        np.arange(0, length, grid_size))
        Z = np.empty(X.shape)
        for index, type_ in enumerate(types):
            spacing = type_.grid_spacing
            Z[(X % spacing == 0) & (Y % spacing == 0)] = index
        """
        X = self.CX
        Y = self.CY
        dims = self.dims
        cocoa = self.cocoa
        cocoa_color = ','.join(str(c) for c in cocoa.color)

        def circle(x, y, d, **kwargs):
            ''' Create a circle at (x,y) with diameter 'd'. '''
            r = d / 2 # convert to radius
            return go.layout.Shape(x0=x-r, y0=y-r, x1=x+r, y1=y+r, **kwargs)

        fig = go.Figure()
        fig.update_layout(title=f'{dims[0]}x{dims[1]}m Cocoa Farm')

        canopy_kwargs = dict(type='circle', xref='x', yref='y',
                             fillcolor=f'rgba({cocoa_color},0.7)',
                             line_color='rgba(0,0,0,0)')
        trunk_kwargs = {**canopy_kwargs, 'fillcolor': f'rgb({cocoa_color})'}
        bound_kwargs = {**canopy_kwargs, 'fillcolor': 'rgba(0,0,0,0)',
                        'line_dash': 'dashdot', 'line_color': 'red'}

        shapes = []#[None] # enable first shape to be displayed
        for x,y in zip(X,Y):
            shapes.append(circle(x, y, cocoa.d_canopy, **canopy_kwargs))
            shapes.append(circle(x, y, cocoa.d_trunk, **trunk_kwargs))
            shapes.append(circle(x, y, cocoa.min_dist, **bound_kwargs))
        fig.add_trace(go.Scatter(name=f'{cocoa.name} ({X.size})', x=X, y=Y,
                                 mode='markers',
                                 marker=dict(color=f'rgb({cocoa_color})')))

        fig.update_layout(shapes=shapes, showlegend=True)
        fig.add_shape(type='rect', x0=0, y0=0, x1=dims[0], y1=dims[1])
        fig.update_yaxes(scaleanchor='x', scaleratio=1)# # axis equal

        if verbose:
            print(f'Plotting {dims[0]}x{dims[1]}m plot -> {X.size} cocoa trees')
        fig.show()


if __name__ == '__main__':
    description = '''
    Example for cocoa growing with temporary and permanent shade trees.
    '''
    from argparse import ArgumentParser
    parser = ArgumentParser(description=description)
    parser.add_argument('-d', '--dims', type=float, nargs=2, default=(100,100),
                        help='dimensions (width, height) of the plot [m]')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='turn off verbose status prints')
    """
    parser.add_argument('-g', '--grid_size', type=float, default=3,
                        help='grid spacing between tree centers [m]')
    parser.add_argument('-c', '--cocoa', type=float, default=1,
                        help='cocoa grid spacing')
    parser.add_argument('-t', '--temp_shade', type=float, default=2,
                        help='temp shade grid spacing')
    parser.add_argument('-p', '--perm_shade', type=float, default=4,
                        help='permanent shade grid spacing')
    """

    args = parser.parse_args()

    """
    trees = (
        Plant('cocoa', args.cocoa, 1, (0.5,0.25,0)),    # brown
        Plant('temps', args.temp_shade, 1, (0,0,1)),    # blue
        Plant('perms', args.perm_shade, 5, (0,1,0,0.5)) # translucent green
    )

    plot_spacing(args.width, args.length, args.grid_size, trees)
    """

    CocoaFarm(args.dims).calculate_and_display()
    
