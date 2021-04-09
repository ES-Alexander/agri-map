#!/usr/bin/env python3

from dataclasses import dataclass
from typing import Tuple
import plotly.graph_objects as go # plotting
from tqdm import tqdm # progress bar
import numpy as np

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
    'view_shade' is a boolean indicating whether to display factored shade
        regions in the visualisation.
    'min_coverage' is the minimum allowed permanent shade coverage of any
        single cocoa tree.
    'avg_coverage' is the minimum allowed average permanent shade coverage of
        all the cocoao trees.
    'iterations' is the number of runs for perm-tree reduction - used to
        ensure that the result is relatively optimal (low number of perm shade
        trees).
    'samples' is the number of points to sample in each cocoa tree, when
        estimating coverage percentage. More samples is more accurate but
        uses more memory and takes longer.

    '''
    dims: tuple = (100, 100)
    cocoa: Tree = Cocoa()
    temp_shade: ShadeTree = TempShade()
    perm_shade: ShadeTree = PermShade()
    view_shade: bool = True
    min_coverage: float = 0.3
    avg_coverage: float = 0.5
    iterations: int = 5
    samples: int = 20
    extra: bool = False
    debug: bool = False

    def calculate_and_display(self, verbose=True):
        self.optimise_spacings()
        self.display_results(verbose)

    def optimise_spacings(self):
        pbar = tqdm(total=3)
        for name, func in (('cocoa', self.optimise_cocoa_spacings),
                           ('temp shade', self.optimise_temp_spacings),
                           ('perm shade', self.optimise_perm_spacings)):
            pbar.set_description(f'Optimising {name} tree spacings')
            func()
            pbar.update()
        pbar.close()

    def optimise_cocoa_spacings(self):
        # Optimal cocoa spacing comes from flat gaps along the shorter axis
        #  and staggered gaps along the longer one.
        # Ensure first index is the smallest.
        dims = min(self.dims), max(self.dims)
        cocoa = self.cocoa
        # adjust dimensions to ensure cocoa canopies stay inside the plot
        adjusted_dims = [side - cocoa.d_canopy for side in dims]
        # use expansion index to determine even spacing of the main grid
        primary_adjusted = adjusted_dims[0]
        primary_spacing  = (primary_adjusted /
                            (primary_adjusted // cocoa.min_dist))
        # calculate the offset and spacing for the indented grid
        secondary_spacing  = 2 * np.sqrt(cocoa.min_dist**2 -
                                         (primary_spacing / 2)**2)
        secondary_dist     = dims[1]
        secondary_adjusted = adjusted_dims[1]
        secondary_offset   = (secondary_dist - secondary_spacing *
                              (secondary_adjusted // secondary_spacing)) / 2
        # set up spacing and offset variables for both directions
        cocoa_r = cocoa.d_canopy / 2
        half_step = (cocoa_r + primary_spacing / 2,
                     cocoa_r + secondary_spacing / 2)
        half_max = (dims[0] - cocoa_r, dims[1] - cocoa_r)

        # create the main and secondary grids
        cocoa_r /= 1.0001 # handle float inaccuracy

        X0,Y0 = np.meshgrid(np.arange(cocoa_r, dims[0] - cocoa_r,
                                      primary_spacing),
                            np.arange(cocoa_r, dims[1] - cocoa_r,
                                      secondary_spacing))
        X1,Y1 = np.meshgrid(np.arange(half_step[0], half_max[0],
                                      primary_spacing),
                            np.arange(half_step[1], half_max[1],
                                      secondary_spacing))

        # save the relevant results internally for later use
        self.primary, self.secondary = primary_spacing, secondary_spacing
        self.CX0, self.CX1 = X0, X1
        self.CY0, self.CY1 = Y0, Y1
        self.CX = np.hstack((self.CX0.reshape(-1), self.CX1.reshape(-1)))
        self.CY = np.hstack((self.CY0.reshape(-1), self.CY1.reshape(-1)))

    def optimise_temp_spacings(self):
        '''

        NOTE: assumes x is the narrower axis, so vertical (y) is secondary.

        '''
        # !!TODO!! check and make sure this still works for primary on y-axis
        temp = self.temp_shade
        cocoa = self.cocoa
        d_shade = temp.d_canopy * temp.shade_factor
        h_offset = self.primary / 2
        if d_shade < cocoa.min_dist + cocoa.d_trunk:
            # requires one temp shade tree per cocoa tree
            #  put as close as possible, at 45 degrees, to try to ensure
            #  all temp shade trees end up in the grid.
            offset = (temp.min_dist + cocoa.d_trunk/2) / np.sqrt(2)
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
            h_offset = self.primary / 2
            # calculate circumcircle diameter (around triangle of trunks)
            base = self.primary + cocoa.d_trunk
            height = (self.secondary + cocoa.d_trunk) / 2
            d_circum = (height**2 + (base**2) / 4) / height
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
                v = np.sqrt(d_circum**2
                            - (self.primary + cocoa.d_trunk)**2) / 2
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
        # generate possible perm-shade tree positions
        perm_poss = self._perm_poss()

        # generate 'samples' sample points for one cocoa tree
        samples = self.circ_sample()
        N = len(samples)

        # -> sample points for all cocoa trees (add offsets)
        samples = (samples.reshape(1,-1)
                   .repeat(self.CX.size, axis=0)
                   .reshape(-1,2)
                   + np.array([self.CX, self.CY]).T
                   .repeat(len(samples), axis=0))
        self.sample_result = samples

        # calculate boolean coverage dataframe with multi-index of cocoa-tree,
        #  point_index vs covered-by-perm-tree-location
        d2 = (self.perm_shade.d_canopy / 2 * self.perm_shade.shade_factor) ** 2
        coverage = np.empty((len(samples), len(perm_poss)), dtype=bool)
        for index, p in enumerate(perm_poss): # TODO vectorise
            coverage[:, index] = ((samples - p)**2).sum(axis=1) <= d2

        # try to remove each tree, but add it back in if removing it means a
        #  tree loses its required coverage, or the average coverage becomes
        #  too low
        backup = coverage.copy()
        rng = np.random.default_rng()
        min_count = len(perm_poss)
        best = None

        def indices_options():
            ''' Create a generator of removal orderings.

            Tries some logical orderings to start with (point creation order,
            outside first to the middle, middle first to the outside), then
            randomly shuffles the indices for any remaining iterations.

            '''
            indices = creation_order = np.arange(len(perm_poss))
            # intelligent options
            yield 'creation', creation_order
            yield ('out first',
                   (out_to_center := np.argsort(((perm_poss-perm_poss
                                                  .mean(axis=0))**2)
                                                .sum(axis=1))))
            yield ('center first',
                   (center_to_out := (len(perm_poss) - out_to_center - 1)))

            # random for any remaining options
            for _ in range(self.iterations - 3):
                rng.shuffle(indices) # in-place array shuffle
                yield 'random', indices

        pbar = tqdm(total=self.iterations * len(perm_poss))
        for i, (order, indices) in enumerate(indices_options(), start=1):
            pbar.set_description(f'{i}/{self.iterations} - Trying '
                                 f'{order.replace("_"," ")} indices')
            keep = np.ones(len(perm_poss), dtype=bool)

            for index in indices:
                p = perm_poss[index]
                stored = coverage[:, index].copy()
                coverage[:, index] = 0
                # use bitwise-or to check that each point is covered by at
                #  least one tree
                cov_prop = (np.bitwise_or.reduce(coverage, axis=1)
                            .reshape(-1, N).sum(axis=1) / N)
                min_cov = cov_prop.min()
                avg_cov = cov_prop.mean()
                if min_cov < self.min_coverage or avg_cov < self.avg_coverage:
                    coverage[:, index] = stored
                    best_min_cov = min_cov
                    best_avg_cov = avg_cov
                else:
                    keep[index] = 0
                pbar.update()

            if (count := keep.sum()) < min_count:
                self.min_cov_result = best_min_cov
                self.avg_cov_result = best_avg_cov
                best = keep.copy()
                min_count = count
            if self.debug:
                print(f'iter {i}: {order}) {best_min_cov=:.3f}, '
                      f'{best_avg_cov=:.3f}, {count=}')

            # reset for next round
            coverage = backup.copy()

        pbar.close()

        perm_poss = perm_poss[best]
        self.PX, self.PY = perm_poss.T

    def _perm_poss(self):
        ''' Calculate possible perm-tree positions '''
        # green poss, +/- v for blue poss
        PX0, PY0 = np.meshgrid(self.CX1[0], self.CY0[:,0])
        PX1, PY1 = np.meshgrid(self.CX0[0, 1:-1], self.CY1[:,0])

        PX = np.hstack((PX0.reshape(-1), PX1.reshape(-1)))
        PY = np.hstack((PY0.reshape(-1), PY1.reshape(-1)))

        if self.extra:
            # calculate extra points to consider (in biggest gaps
            #  -> worse coverage, more likely to fit)
            cocoa = self.cocoa
            h_offset = self.primary / 2
            # calculate circumcircle diameter (around triangle of trunks)
            base = self.primary + cocoa.d_trunk
            height = (self.secondary + cocoa.d_trunk) / 2
            d_circum = (height**2 + (base**2) / 4) / height
            # each temp tree can cover three trees in a triangle
            # determine circumcircle center vertical offset
            v = np.sqrt(d_circum**2 - (self.primary + cocoa.d_trunk)**2) / 2
            above = PY + v
            below = PY - v

            PX = np.hstack([PX]*3)
            PY = np.hstack((above, PY, below))

        pts = np.array((PX, PY)).T

        # remove any too close to temp trees
        min_sep2 = self.perm_shade.min_dist ** 2
        for temp in zip(self.TX, self.TY):
            pts = pts[((pts - temp)**2).sum(axis=1) > min_sep2]

        self.insufficient = ('Failed to create sufficient perm-tree '
                             'locations.\n Consider trying extra with `-e`,'
                             'or reducing `--perm_min_dist` (`-pm`) or '
                             '`--perm_trunk` (`-pt`)')
        r_perm = self.perm_shade.d_canopy * self.perm_shade.shade_factor / 2
        assert len(pts) > (self.dims[0] * self.dims[1]
                           / r_perm**2), self.insufficient
        return pts

    def circ_sample(self):
        ''' Uniform sample of cocoa canopy disk using sunflower arrangement.

        Samples 'samples' points in a circle of radius cocoa.d_canopy / 2.

        wolframcloud.com/objects/demonstrations/SunflowerSeedArrangements-source.nb

        '''
        phi = (np.sqrt(5) + 1) / 2
        n = np.arange(1, self.samples+1)
        r = np.sqrt(n)
        r /= r[-1] # normalise to 0-1 range
        r *= self.cocoa.d_canopy / 2
        theta = 2 * n * np.pi / phi**2
        return np.column_stack((r * np.cos(theta), r * np.sin(theta)))

    def display_results(self, verbose=True):
        dims = self.dims

        def circle(x, y, d, **kwargs):
            ''' Create a circle at (x,y) with diameter 'd'. '''
            r = d / 2 # convert to radius
            return go.layout.Shape(x0=x-r, y0=y-r, x1=x+r, y1=y+r, **kwargs)

        pbar = tqdm(total=14+self.debug)
        pbar.set_description('Initialising plot')
        fig = go.Figure()
        fig.update_layout(title=(f'{dims[0]}x{dims[1]}m Cocoa Farm - '
                                 f'min-coverage={self.min_cov_result:.3f}, '
                                 f'avg-coverage={self.avg_cov_result:.3f}'))
        pbar.update()

        shapes = []
        clear = 'rgba(0,0,0,0)'
        for tree, (X, Y) in ((self.cocoa, (self.CX, self.CY)),
                             (self.temp_shade, (self.TX, self.TY)),
                             (self.perm_shade, (self.PX, self.PY))):
            pbar.set_description(f'Plotting {tree.name} Trees')
            color = ','.join(str(c) for c in tree.color)

            canopy_kwargs = dict(type='circle', xref='x', yref='y',
                                fillcolor=f'rgba({color},0.5)',
                                line_color=clear)
            trunk_kwargs = {**canopy_kwargs, 'fillcolor': f'rgb({color})'}
            bound_kwargs = {**canopy_kwargs, 'fillcolor': clear,
                            'line_dash': 'dashdot', 'line_color': 'red'}

            plot_components = [('canopies', tree.d_canopy, canopy_kwargs),
                               ('trunks', tree.d_trunk, trunk_kwargs),
                               ('bound-lines', tree.min_dist, bound_kwargs)]

            if isinstance(tree, ShadeTree) and self.view_shade:
                shade_kwargs = {**canopy_kwargs,
                                'fillcolor': f'rgba({color},0.2)'}
                plot_components.append(('factored-shade',
                                        tree.d_canopy * tree.shade_factor,
                                        shade_kwargs))

            for stage, d, kwargs in plot_components:
                pbar.set_description(f'Plotting {tree.name} {stage}')
                for x,y in zip(X,Y):
                    shapes.append(circle(x, y, d, **kwargs))
                pbar.update()

            fig.add_trace(go.Scatter(name=f'{tree.name} ({X.size})', x=X, y=Y,
                                    mode='markers',
                                    marker=dict(color=f'rgb({color})')))

        if self.debug:
            pbar.set_description('Plotting sampled points')
            fig.add_trace(
                go.Scatter(name='samples', x=self.sample_result[:,0],
                           y=self.sample_result[:,1], mode='markers'))
            pbar.update()

        pbar.set_description('Registering plot elements')
        fig.update_layout(shapes=shapes, showlegend=True)
        fig.add_shape(type='rect', x0=0, y0=0, x1=dims[0], y1=dims[1])
        fig.update_yaxes(scaleanchor='x', scaleratio=1)# axis equal

        pbar.update()

        pbar.set_description('Displaying plot')
        fig.show()
        pbar.update()
        pbar.close()

        if verbose:
            print(f'Plotting {dims[0]}x{dims[1]}m plot')
            print(f' -> {self.CX.size} cocoa trees')
            print(f' -> {self.TX.size} temp shade trees')
            print(f' -> {self.PX.size} perm shade trees')


if __name__ == '__main__':
    description = '''
    Example for cocoa growing with temporary and permanent shade trees.
    '''
    import inspect
    from argparse import ArgumentParser

    def get_defaults(cls, method='__init__'):
        signature = inspect.signature(getattr(cls, method))
        return {k: v.default
                for k, v in signature.parameters.items()
                if v.default is not inspect.Parameter.empty}

    farm = get_defaults(CocoaFarm)

    parser = ArgumentParser(description=description)
    parser.add_argument('-d', '--dims', type=float, nargs=2, default=(100,100),
                        help='dimensions (width <= height) of the plot [m]')
    parser.add_argument('-n', '--no_shade', action='store_true',
                        help='flag to turn off viewing factored shade regions')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='turn off verbose status prints')
    parser.add_argument('-i', '--iterations', default=farm['iterations'],
                        type=int, help='number of perm-tree ')
    parser.add_argument('--samples', default=farm['samples'], type=int,
                        help='number of cocoa samples for perm coverage')
    parser.add_argument('--min_coverage', default=farm['min_coverage'],
                        type=float, help=('minimum cocoa tree perm coverage '
                                          'proportion [0,1)'))
    parser.add_argument('--avg_coverage', default=farm['avg_coverage'],
                        type=float, help=('minimum average cocoa tree perm '
                                          'coverage proportion [0,1)'))
    parser.add_argument('-e', '--extra', action='store_true',
                        help='flag to try additional perm-tree locations')
    parser.add_argument('--debug', action='store_true',
                        help='extra display when output')

    for tree in (Cocoa, TempShade, PermShade):
        prefix = tree.__name__.replace('Shade', '').lower()
        p = prefix[0]
        Prefix = prefix.title()
        # get the class default initialisation values programatically
        defaults = get_defaults(tree)

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
    farm = CocoaFarm(args.dims, cocoa, temp, perm, not args.no_shade,
                     args.min_coverage, args.avg_coverage, args.iterations,
                     args.samples, args.extra, args.debug)
    farm.calculate_and_display()
