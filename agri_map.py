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
                 shade_factor=1.3, color=(0, 0, 1.0)):
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
                 shade_factor=1.3, color=(0, 1.0, 0)):
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
    view_shade: bool = True

    def calculate_and_display(self, verbose=True):
        self.optimise_spacings()
        self.display_results(verbose)

    def optimise_spacings(self):
        self.optimise_cocoa_spacings()
        self.optimise_temp_spacings()
        self.optimise_perm_spacings()

    def optimise_cocoa_spacings(self):
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

        # save the relevant results internally for later use
        self.primary, self.secondary = primary_spacing, secondary_spacing
        self.CX0 = X0
        self.CX1 = X1
        self.CY0 = Y0
        self.CY1 = Y1
        self.CX = np.hstack((self.CX0.reshape(-1), self.CX1.reshape(-1)))
        self.CY = np.hstack((self.CY0.reshape(-1), self.CY1.reshape(-1)))

    def optimise_temp_spacings(self):
        temp = self.temp_shade
        cocoa = self.cocoa
        d_shade = temp.d_canopy * temp.shade_factor
        if d_shade < cocoa.min_dist + cocoa.d_trunk:
            # requires one temp shade tree per cocoa tree
            #  put as close as possible, at 45 degrees, to try to ensure
            #  all temp shade trees end up in the grid.
            offset = (temp.min_dist + cocoa.d_trunk/2) / math.sqrt(2)
            self.TX = self.CX.reshape(-1) + offset
            self.TY = self.CY.reshape(-1) + offset
        elif d_shade < self.primary + cocoa.d_trunk:
            # each temp tree can cover max two trees on the diagonal
            self.TX = self.CX0.reshape(-1) + self.primary / 4
            self.TY = self.CY0.reshape(-1) + self.secondary / 4
            # TODO handle trees outside on the right side (unlikely to occur)
            """
            right_limit = self.dims[0] - temp.d_trunk / 2
            if TX.max() > right_limit:
                out_on_right = TX > right_limit
                TX[out_on_right] 

            # , and possibly top
            """
        else:
            base = self.primary + cocoa.d_trunk
            height = (self.secondary + cocoa.d_trunk) / 2
            d_circum = (height**2 + (base**2) / 4) / height
            h_offset = self.primary / 2
            if d_shade < d_circum:
                # each temp tree can cover max two trees on the horizontal
                # so get every second tree position horizontally
                TX0 = self.CX0[:,::2].reshape(-1)
                TY0 = self.CY0[:,::2].reshape(-1)
                TX1 = self.CX1[:,::2].reshape(-1)
                TY1 = self.CY1[:,::2].reshape(-1)
                # offset horizontally to half-way between each pair
                TX = np.hstack((TX0, TX1)) + h_offset
                # handle trees outside on the right
                TX[TX > self.dims[0]] -= self.primary
                self.TX = TX
                self.TY = np.hstack((TY0, TY1))
            elif d_shade < self.secondary + cocoa.d_trunk:
                # each temp tree can cover three trees in a triangle
                # determine circumcircle center vertical offset
                v = math.sqrt(d_circum**2 - (self.primary + cocoa.d_trunk)**2) / 2
                # get every third tree position horizontally
                TX0 = self.CX0[:,::3].reshape(-1)
                TY0 = self.CY0[:,::3].reshape(-1) + v
                # offset secondary rows by one for flipped side
                TX1 = self.CX1[:,1::3].reshape(-1)
                TY1 = self.CY1[:,1::3].reshape(-1) - v
                self.TX = np.hstack((TX0, TX1)) + h_offset
                self.TY = np.hstack((TY0, TY1))
                # TODO check full coverage and ensure no shade trees can be
                #  outside boundary
            else: # assume tree is only large enough to cover a diamond of 4
                TX0 = self.CX0[::2,::2].reshape(-1)
                TY0 = self.CY0[::2,::2].reshape(-1)
                TX1 = self.CX0[1::2,1::2].reshape(-1)
                TY1 = self.CY0[1::2,1::2].reshape(-1)
                self.TX = np.hstack((TX0, TX1)) + h_offset
                self.TY = np.hstack((TY0, TY1))
                # TODO handle missing coverage and ones outside boundary

    def optimise_perm_spacings(self):
        ... # TODO

    def display_results(self, verbose=True):
        """
        X,Y = np.meshgrid(np.arange(0, width, grid_size),
                        np.arange(0, length, grid_size))
        Z = np.empty(X.shape)
        for index, type_ in enumerate(types):
            spacing = type_.grid_spacing
            Z[(X % spacing == 0) & (Y % spacing == 0)] = index
        """
        dims = self.dims

        def circle(x, y, d, **kwargs):
            ''' Create a circle at (x,y) with diameter 'd'. '''
            r = d / 2 # convert to radius
            return go.layout.Shape(x0=x-r, y0=y-r, x1=x+r, y1=y+r, **kwargs)

        fig = go.Figure()
        fig.update_layout(title=f'{dims[0]}x{dims[1]}m Cocoa Farm - {self.temp_shade.d_canopy}m temp canopy')

        shapes = []
        clear = 'rgba(0,0,0,0)'
        for tree, (X, Y) in ((self.cocoa, (self.CX, self.CY)),
                             (self.temp_shade, (self.TX, self.TY))):
            color = ','.join(str(c) for c in tree.color)

            canopy_kwargs = dict(type='circle', xref='x', yref='y',
                                fillcolor=f'rgba({color},0.5)',
                                line_color=clear)
            trunk_kwargs = {**canopy_kwargs, 'fillcolor': f'rgb({color})'}
            bound_kwargs = {**canopy_kwargs, 'fillcolor': clear,
                            'line_dash': 'dashdot', 'line_color': 'red'}

            plot_components = [(tree.d_canopy, canopy_kwargs),
                               (tree.d_trunk, trunk_kwargs),
                               (tree.min_dist, bound_kwargs)]

            if isinstance(tree, ShadeTree) and self.view_shade:
                shade_kwargs = {**canopy_kwargs,
                                'fillcolor': f'rgba({color},0.2)'}
                plot_components.append((tree.d_canopy * tree.shade_factor,
                                        shade_kwargs))

            for x,y in zip(X,Y):
                for d, kwargs in plot_components:
                    shapes.append(circle(x, y, d, **kwargs))

            fig.add_trace(go.Scatter(name=f'{tree.name} ({X.size})', x=X, y=Y,
                                    mode='markers',
                                    marker=dict(color=f'rgb({color})')))

        fig.update_layout(shapes=shapes, showlegend=True)
        fig.add_shape(type='rect', x0=0, y0=0, x1=dims[0], y1=dims[1])
        fig.update_yaxes(scaleanchor='x', scaleratio=1)# axis equal

        if verbose:
            print(f'Plotting {dims[0]}x{dims[1]}m plot')
            print(f' -> {self.CX.size} cocoa trees')
            print(f' -> {self.TX.size} temp shade trees')
            #print(f' -> {self.PX.size} perm shade trees')
        fig.show()


if __name__ == '__main__':
    description = '''
    Example for cocoa growing with temporary and permanent shade trees.
    '''
    from argparse import ArgumentParser
    parser = ArgumentParser(description=description)
    parser.add_argument('-d', '--dims', type=float, nargs=2, default=(100,100),
                        help='dimensions (width, height) of the plot [m]')
    parser.add_argument('-n', '--no_shade', action='store_true',
                        help='flag to turn off viewing factored shade regions')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='turn off verbose status prints')


    import inspect
    empty = inspect.Parameter.empty

    for tree in (Cocoa, TempShade, PermShade):
        prefix = tree.__name__.replace('Shade', '').lower()
        p = prefix[0]
        Prefix = prefix.title()
        # get the class default initialisation values programatically
        signature = inspect.signature(tree.__init__)
        defaults = {k: v.default
                    for k, v in signature.parameters.items()
                    if v.default is not empty}

        parser.add_argument(f'-{p}c', f'--{prefix}_canopy',
                            type=float, default=defaults['d_canopy'],
                            help=f'{Prefix} tree canopy diameter [m]')
        parser.add_argument(f'-{p}t', f'--{prefix}_trunk',
                            type=float, default=defaults['d_trunk'],
                            help=f'{Prefix} tree trunk diameter [m]')
        parser.add_argument(f'-{p}m', f'--{prefix}_min_dist',
                            type=float, default=defaults['min_dist'],
                            help=f'{Prefix} tree minimum spacing [m]')
        if issubclass(tree, ShadeTree):
            parser.add_argument(f'-{p}s', f'--{prefix}_shade_factor',
                                type=float, default=defaults['shade_factor'],
                                help=f'{Prefix} tree shade factor')

    # parse the input arguments
    args = parser.parse_args()

    # create the trees with the requested parameters
    cocoa = Cocoa(args.cocoa_trunk, args.cocoa_canopy, args.cocoa_min_dist)
    temp = TempShade(args.temp_trunk, args.temp_canopy, args.temp_min_dist,
                     args.temp_shade_factor)
    perm = PermShade(args.perm_trunk, args.perm_canopy, args.perm_min_dist,
                     args.perm_shade_factor)

    # create the cocoa farm and display the resulting tree configuration
    farm = CocoaFarm(args.dims, cocoa, temp, perm, not args.no_shade)
    farm.calculate_and_display()
