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
            trees. If left as None, defaults to (d_trunk + 0.25).
        'shade_factor' is a multiplier of d_canopy determining where is
            considered to be adequately shaded. Helps to reduce shade
            redundancy for the majority of the day when the sun isn't directly
            overhead. Defaults to 1.3.
        'color' is the RGB[A] colour used when plotting. Defaults to green
            (0, 1, 0) -> 0% red, 100% green, 0% blue.

        '''
        # keep if set, or replace with default
        min_dist = min_dist or (d_trunk * 0.25)
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
        all the cocoa trees.
    'iterations' is the number of runs for perm-tree reduction - used to
        ensure that the result is relatively optimal (low number of perm shade
        trees). First 5 are logical/intelligent, then random shuffles.
    'samples' is the number of points to sample in each cocoa tree, when
        estimating coverage percentage. More samples is more accurate but
        uses more memory and takes longer.
    'boundary_dist' is the minimum distance between the center of a cocoa
        trunk and the boundary.

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
    cov_focus: bool = False
    boundary_dist: float = cocoa.d_trunk

    def calculate_and_display(self, verbose=True):
        self.optimise_spacings()
        self.display_results(verbose)

    def optimise_spacings(self):
        with tqdm(total=3) as pbar:
            for name, func in (('cocoa', self.optimise_cocoa_spacings),
                               ('temp shade', self.optimise_temp_spacings),
                               ('perm shade', self.optimise_perm_spacings)):
                pbar.set_description(f'Optimising {name} tree spacings')
                func()
                pbar.update()

    def optimise_cocoa_spacings(self):
        # Optimal cocoa spacing comes from flat gaps along the shorter axis
        #  and staggered gaps along the longer one.
        # Ensure first index is the smallest.
        dims = self.dims = min(self.dims), max(self.dims)
        cocoa = self.cocoa
        # adjust dimensions to ensure cocoa canopies stay inside the plot
        adjusted_dims = [side - 2*self.boundary_dist for side in dims]
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
        major_offset = self.boundary_dist
        half_step = (self.boundary_dist + primary_spacing / 2,
                     self.boundary_dist + secondary_spacing / 2)
        half_max = (dims[0] - self.boundary_dist, dims[1] - self.boundary_dist)

        # create the main and secondary grids
        fuzzy_boundary = self.boundary_dist / 1.0001 # handle float inaccuracy

        X0,Y0 = np.meshgrid(np.arange(major_offset, dims[0] - fuzzy_boundary,
                                      primary_spacing),
                            np.arange(major_offset, dims[1] - fuzzy_boundary,
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
        temp = self.temp_shade
        cocoa = self.cocoa
        d_shade = temp.d_canopy * temp.shade_factor
        h_offset = self.primary / 2
        if d_shade < cocoa.min_dist + cocoa.d_trunk:
            # requires one temp shade tree per cocoa tree
            #  put as close as possible, at 45 degrees, to try to ensure
            #  all temp shade trees end up in the grid.
            r = (temp.min_dist + cocoa.d_trunk/2)
            offset = r / np.sqrt(2)
            TX = self.CX + offset
            TY = self.CY + offset
            # handle trees above the top boundary (rotate to RHS)
            top = TY > self.dims[1] - self.boundary_dist
            TX[top] += r - offset
            TY[top] -= offset
            # handle trees outside the right boundary (rotate to above)
            right = TX > self.dims[0] - self.boundary_dist
            TX[right] -= offset
            TY[right] += r - offset
            # if odd number of rows, remove top right temp tree
            remove = top & right
            self.TX, self.TY = TX[~remove], TY[~remove]
        elif d_shade < self.primary + cocoa.d_trunk:
            # each temp tree can cover max two trees on the diagonal
            TX = self.CX0.reshape(-1) + self.primary / 4
            TY = self.CY0.reshape(-1) + self.secondary / 4
            # handle trees outside on the right side and top
            r = np.sqrt(self.primary**2 + self.secondary**2) / 4
            theta = np.arctan(self.secondary / self.primary)
            # handle trees above the top boundary (rotate to RHS)
            top = TY > self.dims[1] - self.boundary_dist
            TX[top] += r - self.primary / 4
            TY[top] -= self.secondary / 4
            # handle trees outside the right boundary (rotate to above)
            right = TX > self.dims[0] - self.boundary_dist
            TX[right] -= self.primary / 4
            TY[right] += r - self.secondary / 4
            # if odd number of rows, remove top right temp tree
            remove = top & right
            self.TX, self.TY = TX[~remove], TY[~remove]
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
                right = TX > self.dims[0] - self.boundary_dist
                TX[right] -= self.primary
                self.TX = TX
                self.TY = np.hstack((TY0, TY1))
            elif d_shade < self.secondary + cocoa.d_trunk:
                # each temp tree can cover three trees in a triangle
                # determine circumcircle center vertical offset
                v = np.sqrt(d_circum**2
                            - (self.primary + cocoa.d_trunk)**2) / 2
                # get every third tree position horizontally
                rows = len(self.CX0)
                odd = rows % 2
                top = rows if not odd else rows - 1
                TX0 = self.CX0[:top,::3].reshape(-1)
                TY0 = self.CY0[:top,::3].reshape(-1) + v
                right0 = TX0 > self.dims[0] - self.boundary_dist - h_offset
                TX0[right0] -= h_offset
                TY0[right0] += self.secondary / 2 - v
                # offset secondary rows by one for flipped side
                TX1 = self.CX1[:,1::3].reshape(-1)
                TY1 = self.CY1[:,1::3].reshape(-1) - v
                right1 = TX1 > self.dims[0] - self.boundary_dist - h_offset
                TX1[right1] -= h_offset
                TY1[right1] -= self.secondary / 2 - v
                X_stack = [TX0, TX1]
                Y_stack = [TY0, TY1]
                if odd: # handle top row
                    top_X = self.CX0[-1,::2]
                    if top_X.size % 2:
                        top_X = top_X[:-1]
                    top_Y = np.repeat(self.CY0[-1,0], top_X.size)
                    X_stack.append(top_X)
                    Y_stack.append(top_Y)
                self.TX = np.hstack(X_stack) + h_offset
                self.TY = np.hstack(Y_stack)
            else: # assume tree is only large enough to cover a diamond of 4
                
                TX0 = self.CX0[::2,::2]
                TY0 = self.CY0[::2,::2]
                TX1 = self.CX0[1::2,1::2]
                TY1 = self.CY0[1::2,1::2]
                # move in trees outside right side, and add extras on left side
                odd_cols = self.CX0.shape[1] % 2
                if odd_cols:
                    TX0[:,-1] -= self.primary
                else:
                    TX1[:,-1] -= self.primary
                left_Y = TY1[:,0]
                left_X = np.repeat(TX0[0,0], left_Y.size)

                # create lists of the arrays for joining later
                #  (arrays reallocate memory when joining, so best to only
                #   join once all the items are available)
                X_stack = [X.reshape(-1) for X in (TX0, TX1)]
                X_stack.append(left_X)
                Y_stack = [Y.reshape(-1) for Y in (TY0, TY1)]
                Y_stack.append(left_Y)

                # add extra trees in the top row if required
                #  (same number of secondary and primary grid rows)
                if self.CX0.shape[0] == self.CX1.shape[0]:
                    # same number of secondary and primary temp grid rows
                    if TX0.shape[0] == TX1.shape[0]:
                        top_X = TX0[0,1:-1]
                        top_Y = np.repeat(TY1[-1,0], top_X.size)
                    else:
                        top_X = TX1[0,:-1]
                        top_Y = np.repeat(TY0[-1,0], top_X.size)
                    X_stack.append(top_X)
                    Y_stack.append(top_Y)

                self.TX = np.hstack(X_stack) + h_offset
                self.TY = np.hstack(Y_stack)

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

        def indices_options():
            ''' Create a generator of removal orderings.

            Tries some logical orderings to start with (point creation order,
            outside first to the middle, middle first to the outside), then
            randomly shuffles the indices for any remaining iterations.

            '''
            indices = creation_order = np.arange(len(perm_poss))
            # intelligent options
            yield 'creation', creation_order
            diff = (perm_poss-perm_poss.mean(axis=0))
            yield ('out first',
                   (out_to_center := np.argsort((diff**2).sum(axis=1))))
            yield ('manhattan out', # out first but by manhattan distance (L1)
                   (out_man := np.argsort(diff.sum(axis=1))))
            yield ('center first', (len(perm_poss) - out_to_center - 1))
            yield ('manhattan center', (len(perm_poss) - out_man - 1))

            # random for any remaining options
            rng = np.random.default_rng()
            for _ in range(self.iterations - 5):
                rng.shuffle(indices) # in-place array shuffle
                yield 'random', indices

        with tqdm(total=self.iterations * len(perm_poss)) as pbar:
            self._selective_perm_remove(indices_options(), perm_poss, coverage,
                                        N, pbar)

    def _selective_perm_remove(self, indices_options, perm_poss, coverage, N,
                               pbar):
        '''

        Tries to remove each perm tree, but restores it if removal causes a
          cocoa tree to lose its required coverage, or the average coverage
          becomes too low.

        '''
        backup = coverage.copy()
        min_count = len(perm_poss)
        best = None

        for i in range(self.iterations):
            order, indices = next(indices_options)
            i += 1
            pbar.set_description(f'{i}/{self.iterations} - Trying '
                                 f'{order.replace("_"," ")} indices')
            keep = np.ones(len(perm_poss), dtype=bool)

            for index in indices:
                p = perm_poss[index] # get a tree
                stored = coverage[:, index].copy() # save its coverage for later
                coverage[:, index] = 0 # try removing the tree
                # use bitwise-or to check that each point is covered by at
                #  least one tree -> array of cocoa tree coverage percentages
                cov_prop = (np.bitwise_or.reduce(coverage, axis=1)
                            .reshape(-1, N).sum(axis=1) / N)
                min_cov = cov_prop.min()
                avg_cov = cov_prop.mean()
                if min_cov < self.min_coverage or avg_cov < self.avg_coverage:
                    # failed -> put the tree back
                    coverage[:, index] = stored
                else:
                    # worked -> don't need that tree
                    keep[index] = 0
                    best_min_cov = min_cov
                    best_avg_cov = avg_cov

                pbar.update()

            if ((count := keep.sum()) < min_count or
                (self.cov_focus and count == min_count and
                 (best_min_cov < self.min_cov_result or
                  best_avg_cov < self.avg_cov_result))):
                self.min_cov_result = best_min_cov
                self.avg_cov_result = best_avg_cov
                best = keep.copy()
                min_count = count
            if self.debug:
                print(f'\niter {i}: {order}) {best_min_cov=:.3f}, '
                      f'{best_avg_cov=:.3f}, {count=}\n', flush=True)

            # reset for next round
            coverage = backup.copy()

        perm_poss = perm_poss[best]
        self.PX, self.PY = perm_poss.T

    def _perm_poss(self):
        ''' Calculate possible perm-tree positions '''
        # start with between every neighbouring horizontal pair of cocoa trees
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
        with tqdm(total=14+self.debug) as pbar:
            self._display_results(verbose, pbar)

        if verbose:
            print(f'Plotting {self.dims[0]}x{self.dims[1]}m plot\n'
                  f' -> {self.CX.size} cocoa trees\n'
                  f' -> {self.TX.size} temp shade trees\n'
                  f' -> {self.PX.size} perm shade trees')


    def _display_results(self, verbose, pbar):
        dims = self.dims

        def circle(x, y, d, **kwargs):
            ''' Create a circle at (x,y) with diameter 'd'. '''
            r = d / 2 # convert to radius
            return go.layout.Shape(x0=x-r, y0=y-r, x1=x+r, y1=y+r, **kwargs)

        pbar.set_description('Initialising plot')
        fig = go.Figure()
        fig.update_layout(title=f'{dims[0]}x{dims[1]}m Cocoa Farm')

        text = self.gen_plot_text()
        fig.add_annotation(text=text, xref='paper', yref='paper',
                           x=0.95, y=0.5, showarrow=False, align='left')

        pbar.update()

        shapes = []
        clear = 'rgba(0,0,0,0)'
        for tree, (X, Y) in ((self.cocoa, (self.CX, self.CY)),
                             (self.temp_shade, (self.TX, self.TY)),
                             (self.perm_shade, (self.PX, self.PY))):
            pbar.set_description(f'Plotting {tree.name} Trees')
            color = ','.join(str(c) for c in tree.color)

            canopy_kwargs = dict(type='circle',xref='x',yref='y',
                                 fillcolor=f'rgba({color},0.5)',
                                 line_color=clear)
            trunk_kwargs = {**canopy_kwargs, 'fillcolor': f'rgb({color})'}
            bound_kwargs = {**canopy_kwargs, 'fillcolor': clear,
                            'line_color': 'red','line_dash': 'dashdot'}

            plot_components = [('canopies', tree.d_canopy, canopy_kwargs),
                               ('trunks', tree.d_trunk, trunk_kwargs),
                               ('bound-lines', tree.min_dist, bound_kwargs)]

            if isinstance(tree, ShadeTree) and self.view_shade:
                shade_kwargs = {**canopy_kwargs,
                                'fillcolor': f'rgba({color},0.3)'}
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
                                     marker=dict(color=f'rgb({color})'),
                                     legendgroup=f'{tree.name}'))

        if self.debug:
            pbar.set_description('Plotting sampled points')
            fig.add_trace(
                go.Scatter(name='samples', x=self.sample_result[:,0],
                           y=self.sample_result[:,1], mode='markers'))
            pbar.update()

        pbar.set_description('Registering plot elements')
        fig.update_layout(shapes=shapes)
        fig.update_layout(legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ))
        fig.add_shape(type='rect', x0=0, y0=0, x1=dims[0], y1=dims[1])
        fig.update_yaxes(scaleanchor='x', scaleratio=1)# axis equal

        """ # TODO link tree component cicles to legend clicks
        def legend_click(*a, **kw):
            ... # do something - NOTE: in JS land (`print` doesn't work because
                #  plot is no longer connected to the Python program that
                #  created it)

        fig.layout.on_change(legend_click, 'legend')
        # """

        pbar.update()

        pbar.set_description('Displaying plot')
        fig.show()
        pbar.update()
 
    def gen_plot_text(self):
        sep = '<br> ' # text is parsed as HTML
        min_cov = self.min_cov_result * 100
        avg_cov = self.avg_cov_result * 100
        text = (f'<b>Coverage:</b>{sep}'
                f'{min_cov=:.1f}% >= {self.min_coverage * 100}%{sep}'
                f'{avg_cov=:.1f}% >= {self.avg_coverage * 100}%{sep*2}')

        for index, tree in enumerate((self.cocoa, self.temp_shade,
                                      self.perm_shade)):
            name = tree.name
            d_trunk = tree.d_trunk
            d_canopy = tree.d_canopy
            min_dist = tree.min_dist
            text += sep.join((f'<b>{name}:</b>', f'{d_trunk=:.2f}m',
                              f'{d_canopy=:.2f}m', f'{min_dist=:.2f}m'))
            if isinstance(tree, ShadeTree):
                shade_factor = tree.shade_factor
                text += f'{sep}{shade_factor=:.2f}'

            text += '<br>'*2
        return text


if __name__ == '__main__':
    description = '''
    Example for cocoa growing with temporary and permanent shade trees.
    '''
    import inspect
    from argparse import ArgumentParser

    def get_defaults(cls, method='__init__'):
        ''' Automatically extract the default values of a method. '''
        signature = inspect.signature(getattr(cls, method))
        return {k: v.default
                for k, v in signature.parameters.items()
                if v.default is not inspect.Parameter.empty}

    farm = get_defaults(CocoaFarm)

    parser = ArgumentParser(description=description)
    parser.add_argument('-d', '--dims', type=float, nargs=2,
                        default=farm['dims'],
                        help='dimensions (width <= height) of the plot [m]')
    parser.add_argument('-n', '--no_shade', action='store_true',
                        help='flag to turn off viewing factored shade regions')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='turn off verbose status prints')
    parser.add_argument('-b', '--boundary_dist', type=float, default=None,
                        help='distance from cocoa trunk center to boundaries'
                             ' (default is one cocoa trunk diameter)')
    parser.add_argument('-i', '--iterations', default=farm['iterations'],
                        type=int, help='number of perm-tree runs (first 5 are'
                                       ' logical/intelligent, rest random)')
    parser.add_argument('--samples', default=farm['samples'], type=int,
                        help='number of cocoa samples for perm coverage')
    parser.add_argument('--min_coverage', default=farm['min_coverage'],
                        type=float, help=('minimum cocoa tree perm coverage '
                                          'proportion [0,1)'))
    parser.add_argument('--avg_coverage', default=farm['avg_coverage'],
                        type=float, help=('minimum average cocoa tree perm '
                                          'coverage proportion [0,1)'))
    parser.add_argument('--cov_focus', action='store_true',
                        help='flag for preferring less coverage over order'
                             ' (given the same number of perm trees)')
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

    boundary_dist = args.boundary_dist or cocoa.d_trunk

    # create the cocoa farm and display the resulting tree configuration
    farm = CocoaFarm(args.dims, cocoa, temp, perm, not args.no_shade,
                     args.min_coverage, args.avg_coverage, args.iterations,
                     args.samples, args.extra, args.debug, args.cov_focus,
                     boundary_dist)
    farm.calculate_and_display()
