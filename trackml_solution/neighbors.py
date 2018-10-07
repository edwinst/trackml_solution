"""Neighborhood search for finding hits to start and extend track
candidates with.

For mostly historical reasons, the classes in this file also handle
the calculation of per-layer interpolated functions.

---

Copyright 2018 Edwin Steiner

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from collections import OrderedDict

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

from sklearn.neighbors import NearestNeighbors

def createLayerFunctionInterpolators(is_cylinder, layer_id, layer_functions=None):
    if layer_functions is not None:
        interpolators = {}
        fns = layer_functions.functions
        fdata = layer_functions.getGridValues(is_cylinder=is_cylinder, layer_id=layer_id, functions=fns)
        for name, (points, values) in zip(fns, fdata):
            if points is None:
                interpolators[name] = (0, 0, None)
            else:
                ndropped_dims = 0
                while (len(points) > 1) and (len(points[0]) == 1):
                    ndropped_dims += 1
                    points = points[1:]
                    values = values.reshape(values.shape[1:])
                #if ndropped_dims > 0:
                #    print("    dropped %d dimensions for function %s on (%s %d)"
                #        % (ndropped_dims, name, "cyl_id" if is_cylinder else "cap_id", layer_id))
                for coord in points:
                    assert len(coord) > 1
                interpolators[name] = (len(points), ndropped_dims, RegularGridInterpolator(
                    points=points, values=values,
                    method='linear', bounds_error=False, fill_value=None))
    else:
        interpolators = None
    return interpolators

class CylinderNeighbors:
    class Cylinder:
        def __init__(self, cyl_id, cyl_group, params, layer_functions=None):
            xyz = cyl_group.as_matrix(columns=['x', 'y', 'z'])
            self.cyl_hit_id_array = cyl_group['hit_id'].values
            # choose radius of neighborhood cylinder
            # and project onto a cylinder of this radius
            r2 = np.linalg.norm(xyz[:,0:2], axis=1)
            self.zscale = params['nb__cyl_scale'] # XXX refine choice of radius in order to tune dz vs. d-azimuthal neighborhood sizes
            self.cyl_mean_r2 = np.mean(r2)
            self.cyl_nbr = self.zscale * self.cyl_mean_r2
            self.nb_coords = self.transformToNeighborCoords(xyz)
            self.neighbors = NearestNeighbors()
            self.neighbors.fit(self.nb_coords)
            self.cyl_hits_df = cyl_group
            self.layer_functions = createLayerFunctionInterpolators(is_cylinder=True, layer_id=cyl_id, layer_functions=layer_functions)

        def transformToNeighborCoords(self, v):
            """Transform the given Euclidean coordinates into the
            coordinate system used to define the neighborhoods in
            this cylinder.
            Args:
                v (array (n_points,3)): Euclidean coordinates of the n_points points.
            Returns:
                (array (n_points,3)): transformed coordinates
            """
            vr2 = np.linalg.norm(v[:,0:2], axis=1)
            vcyl = v * self.cyl_mean_r2 / vr2[:,np.newaxis]
            vnorm = vcyl.copy()
            vnorm[:,0:2] = vnorm[:,0:2] * self.zscale
            return vnorm

        def centralNeighborhood(self, absz):
            return self.cyl_hits_df.loc[self.cyl_hits_df['z'].abs() < absz]

        def evaluateLayerFunctions(self, x, y, z, functions, points=None):
            """Evaluate per-layer interpolated functions at the given coordinates.
            Args:
                x, y, z (float array(N,)): coordinates at which to evaluate the functions.
                    Note: Ignored if `points` is given.
                functions (list of str): names of the functions to evaluate.
                points (None or array(N,M)): If given, evaluate layer functions at the
                    given point coordinates and ignore (x, y, z) arguments.
            Returns:
                vals (list of float array(N,)): for each name passed in `functions`,
                    this list containes the interpolated values of the corresponding
                    layer function for the given coordinates.
            """
            vals = []
            phi_z = None
            if self.layer_functions is not None:
                for fn in functions:
                    val = None
                    nargs, ndrop, interpolator = self.layer_functions[fn]
                    if interpolator is not None:
                        if points is not None:
                            arg = points[:,ndrop:]
                        elif nargs == 1:
                            assert ndrop == 0
                            arg = z
                        else:
                            assert ndrop == 0
                            if phi_z is None:
                                phi_z = np.stack([z, np.arctan2(y, x)], axis=1)
                            arg = phi_z
                        val = interpolator(arg)
                    vals.append(val)
            return vals

    def __init__(self, cylspec, params, layer_functions=None):
        self.cylspec = cylspec
        self.cylinders = []
        self.params = params
        self.layer_functions = layer_functions

    def fit(self, hits, hit_in_cyl=None, hit_layer_id=None):
        cyl_hits = hits.join(self.cylspec.cylinders_df, on=['volume_id', 'layer_id'], how='inner', sort=False)
        if hit_in_cyl is not None:
            hit_in_cyl[cyl_hits['hit_id'].values] = True
        cyl_groups = cyl_hits.groupby('cyl_id')
        for cyl_id, cyl_group in cyl_groups:
            self.cylinders.append(self.Cylinder(cyl_id, cyl_group, self.params, layer_functions=self.layer_functions))
            if hit_layer_id is not None:
                hit_layer_id[cyl_group['hit_id'].values] = cyl_id

    def findNeighborhood(self, cyl_id, v, dz):
        cyl = self.cylinders[cyl_id]
        # project vertices to neighborhood cylinder
        vnorm = cyl.transformToNeighborCoords(v)
        # find radius neighbors
        dist_aa, ind_aa = cyl.neighbors.radius_neighbors(vnorm, radius=dz)
        # build DataFrame
        nb_inds = np.concatenate(ind_aa)
        dists = np.concatenate(dist_aa)
        v_inds = np.repeat(np.arange(0, len(v), 1, dtype=np.int32), repeats=[len(ind_a) for ind_a in ind_aa])
        df = pd.DataFrame(data={'vind': v_inds, 'nb_hit_id': cyl.cyl_hit_id_array[nb_inds], 'nb_dist': dists})
        return df

    def findNeighborhoodK(self, cyl_id, v, k):
        cyl = self.cylinders[cyl_id]
        # make sure there are enough hits in the cylinder neighborhood
        assert len(cyl.cyl_hit_id_array) >= k
        # project vertices to neighborhood cylinder
        vnorm = cyl.transformToNeighborCoords(v)
        # find k neighbors
        dist_a, ind_a = cyl.neighbors.kneighbors(vnorm, n_neighbors=k)
        # build DataFrame
        nb_inds = ind_a.flatten()
        dists = dist_a.flatten()
        v_inds = np.repeat(np.arange(0, len(v), 1, dtype=np.int32), repeats=k)
        df = pd.DataFrame(data={'vind': v_inds, 'nb_hit_id': cyl.cyl_hit_id_array[nb_inds], 'nb_dist': dists})
        return df

class CapNeighbors:
    class Cap:
        def __init__(self, cap_id, cap_group, cap_z, layer_functions=None):
            self.cap_z = cap_z
            self.nb_coords = self.transformToNeighborCoords(cap_group.as_matrix(columns=['x', 'y', 'z']))
            self.cap_hit_id_array = cap_group['hit_id'].values
            self.neighbors = NearestNeighbors()
            self.neighbors.fit(self.nb_coords)
            self.cap_hits_df = cap_group
            self.layer_functions = createLayerFunctionInterpolators(is_cylinder=False, layer_id=cap_id, layer_functions=layer_functions)

        def transformToNeighborCoords(self, v):
            """Transform the given Euclidean coordinates into the
            coordinate system used to define the neighborhoods in
            this cap.
            Args:
                v (array (n_points,3)): Euclidean coordinates of the n_points points.
            Returns:
                (array (n_points,2)): transformed coordinates
            """
            return v[:,0:2] * (self.cap_z / v[:,2])[:,np.newaxis]

        def centralNeighborhood(self, dr2):
            return self.cap_hits_df.loc[np.square(self.cap_hits_df['x']) + np.square(self.cap_hits_df['y']) < np.square(dr2)]

        def evaluateLayerFunctions(self, x, y, z, functions, points=None):
            """Evaluate per-layer interpolated functions at the given coordinates.
            Args:
                x, y, z (float array(N,)): coordinates at which to evaluate the functions.
                    Note: Ignored if `points` is given.
                functions (list of str): names of the functions to evaluate.
                points (None or array(N,M)): If given, evaluate layer functions at the
                    given point coordinates and ignore (x, y, z) arguments.
            Returns:
                vals (list of float array(N,)): for each name passed in `functions`,
                    this list containes the interpolated values of the corresponding
                    layer function for the given coordinates.
            """
            vals = []
            r2 = None
            phi_r2 = None
            if self.layer_functions is not None:
                for fn in functions:
                    val = None
                    nargs, ndrop, interpolator = self.layer_functions[fn]
                    if interpolator is not None:
                        if points is not None:
                            arg = points[:,ndrop:]
                        elif nargs == 1:
                            assert ndrop == 0
                            if r2 is None:
                                r2 = np.sqrt(np.square(x) + np.square(y))
                            arg = r2
                        else:
                            assert ndrop == 0
                            if phi_r2 is None:
                                r2 = np.sqrt(np.square(x) + np.square(y))
                                phi_r2 = np.stack([np.arctan2(y, x), r2], axis=1)
                            arg = phi_r2
                        val = interpolator(arg)
                    vals.append(val)
            return vals

    def __init__(self, capspec, layer_functions=None):
        self.capspec = capspec
        self.caps = []
        self.layer_functions = layer_functions

    def fit(self, hits, hit_layer_id=None):
        cap_hits = hits.join(self.capspec.cap_layers_df, on=['volume_id', 'layer_id'])
        cap_groups = cap_hits.groupby('cap_id')
        for cap_id, cap_group in cap_groups:
            cap_id = int(cap_id) # XXX find out why that is needed
            self.caps.append(self.Cap(cap_id, cap_group, self.capspec.cap_z[cap_id], layer_functions=self.layer_functions))
            if hit_layer_id is not None:
                hit_layer_id[cap_group['hit_id'].values] = cap_id

    def findNeighborhood(self, cap_id, v, radius):
        cap = self.caps[cap_id]
        # project vertices to neighborhood coordinate system
        vnorm = cap.transformToNeighborCoords(v)
        # find radius neighbors
        dist_aa, ind_aa = cap.neighbors.radius_neighbors(vnorm, radius=radius)
        # build DataFrame
        nb_inds = np.concatenate(ind_aa)
        dists = np.concatenate(dist_aa)
        v_inds = np.repeat(np.arange(0, len(v), 1, dtype=np.int32), repeats=[len(ind_a) for ind_a in ind_aa])
        df = pd.DataFrame(data={'vind': v_inds, 'nb_hit_id': cap.cap_hit_id_array[nb_inds], 'nb_dist': dists})
        return df

    def findNeighborhoodK(self, cap_id, v, k):
        cap = self.caps[cap_id]
        # make sure there are enough hits in the cap neighborhood
        assert len(cap.cap_hit_id_array) >= k
        # project vertices to neighborhood coordinate system
        vnorm = cap.transformToNeighborCoords(v)
        # find k neighbors
        dist_a, ind_a = cap.neighbors.kneighbors(vnorm, n_neighbors=k)
        # build DataFrame
        nb_inds = ind_a.flatten()
        dists = dist_a.flatten()
        v_inds = np.repeat(np.arange(0, len(v), 1, dtype=np.int32), repeats=k)
        df = pd.DataFrame(data={'vind': v_inds, 'nb_hit_id': cap.cap_hit_id_array[nb_inds], 'nb_dist': dists})
        return df

class Neighbors:
    default_params = OrderedDict([
        ('nb__cyl_first_dz_0', 1000.0),
        ('nb__cyl_first_dz_1', 200.0),
        ('nb__cyl_first_dz_4', 400.0),
        ('nb__cap_first_dr2_0', 100.0),
        ('nb__cap_first_dr2_1', 200.0),
        ('nb__cap_first_inc'  , 5.0),
        ('nb__min_radius'     , 1.0),
        ('nb__nbins_radius'   , 100),
        ('nb__cyl_scale', 5.0),
    ])

    def __init__(self, spec, params={}, layer_functions=None):
        self.params = self.default_params
        self.params.update({k: v for k, v in params.items() if k in self.default_params})
        self.capn = CapNeighbors(spec.caps, layer_functions=layer_functions)
        self.cyln = CylinderNeighbors(spec.cylinders, params=self.params, layer_functions=layer_functions)
        self.layer_functions = layer_functions

    def fit(self, hits, hit_in_cyl=None, hit_layer_id=None):
        # XXX maybe we should be more careful with the name "hit_layer_id" and use our own to generalize cap_id/cyl_id
        self.capn.fit(hits, hit_layer_id=hit_layer_id)
        self.cyln.fit(hits, hit_in_cyl=hit_in_cyl, hit_layer_id=hit_layer_id)

    def _findCylinderNeighborhood(self, cyl_id, xi, yi, zi, sel_id_and_radius, where_id_and_radius, max_radius):
        ints = np.stack((xi, yi, zi), axis=1).compress(sel_id_and_radius, axis=0)
        nb_df = self.cyln.findNeighborhood(cyl_id, ints, dz=max_radius)
        return nb_df

    def _findCylinderNeighborhoodK(self, cyl_id, xi, yi, zi, sel_id_and_radius, where_id_and_radius, k):
        ints = np.stack((xi, yi, zi), axis=1).compress(sel_id_and_radius, axis=0)
        nb_df = self.cyln.findNeighborhoodK(cyl_id, ints, k=k)
        return nb_df

    def _findCapNeighborhood(self, cap_id, xi, yi, zi, sel_id_and_radius, where_id_and_radius, max_radius):
        ints = np.stack((xi, yi, zi), axis=1).compress(sel_id_and_radius, axis=0)
        nb_df = self.capn.findNeighborhood(cap_id, ints, radius=max_radius)
        return nb_df

    def _findCapNeighborhoodK(self, cap_id, xi, yi, zi, sel_id_and_radius, where_id_and_radius, k):
        ints = np.stack((xi, yi, zi), axis=1).compress(sel_id_and_radius, axis=0)
        nb_df = self.capn.findNeighborhoodK(cap_id, ints, k=k)
        return nb_df

    def _findNeighborhoodsGroupedByIdAndRadius(self, xi, yi, zi, selection, next_id, radius, finder):
        nb_dfs = []
        ids = np.unique(next_id[selection])
        for id in ids:
            selection_id = selection & (next_id == id)
            where_id = np.where(selection_id)[0]
            # Problem: The neighborhood search only supports a single scalar radius.
            # Solution: We cluster the logarithms of the radii and bin them together.
            radius_id = radius[selection_id]
            max_radius = max(self.params['nb__min_radius'], np.amax(radius_id))
            min_radius = max(self.params['nb__min_radius'], np.amin(radius_id))
            log_steps = np.linspace(np.log(min_radius / max_radius), 0.0, int(self.params['nb__nbins_radius']))
            radius_bins = np.exp(log_steps) * max_radius
            # make sure the last step is exactly max_radius, not rounded down, etc.
            radius_bins[-1] = max_radius
            # bin radii
            radius_bin_indices = np.digitize(radius_id, radius_bins, right=True)
            assert np.all(0 <= radius_bin_indices) and np.all(radius_bin_indices < len(radius_bins))
            assert np.all(radius_id <= radius_bins[radius_bin_indices])
            for i_radius, bin_radius in enumerate(radius_bins):
                sel_id_and_radius = selection_id.copy()
                sel_id_and_radius[where_id[radius_bin_indices != i_radius]] = False
                if np.any(sel_id_and_radius):
                    where_id_and_radius = np.where(sel_id_and_radius)[0]
                    nb_df = finder(id, xi, yi, zi, sel_id_and_radius, where_id_and_radius, bin_radius)
                    nb_df['vind'] = where_id_and_radius[nb_df['vind'].values]
                    #print("    cluster ", i_radius, " has max radius ", '%.2f' % bin_radius, " and ", len(where_id_and_radius), " instances and ", len(nb_df), " neighbors")
                    nb_dfs.append(nb_df)
        return nb_dfs

    def _findNeighborhoodsGroupedById(self, xi, yi, zi, selection, next_id, finder, **finder_args):
        nb_dfs = []
        ids = np.unique(next_id[selection])
        for id in ids:
            selection_id = selection & (next_id == id)
            where_id = np.where(selection_id)[0]
            nb_df = finder(id, xi, yi, zi, selection_id, where_id, **finder_args)
            nb_df['vind'] = where_id[nb_df['vind'].values]
            nb_dfs.append(nb_df)
        return nb_dfs

    def evaluateLayerFunctions(self, xi, yi, zi, cyl_closer, next_id, functions=None, points=None):
        """Evaluate per-layer interpolated functions at the given coordinates.
        Args:
            xi, yi, zi (float array(N,)): coordinates at which to evaluate the functions.
                Note: Ignored if `points` is given.
            cyl_closer (bool or bool array(N,)): True if the respective point is on a
                cylinder layer, False if it is on a cap.
            next_id (int array(N,)): id of the layer the respective point is on
                (cyl_id for cyl_closer==True, cap_id for cyl_closer==False).
            functions (list of str): names of the functions to evaluate.
                If `None`, evaluate all available layer functions.
            points (None or array(N,M)): If given, evaluate layer functions at the
                given point coordinates and ignore (xi, yi, zi) arguments.
        Returns:
            vals (list of float array(N,)): for each name passed in `functions`,
                this list containes the interpolated values of the corresponding
                layer function for the given coordinates.
        """
        assert self.layer_functions is not None
        if functions is None:
            functions = self.layer_functions.functions
        if points is not None:
            n = points.shape[0]
            int_finite = True
        else:
            n = len(xi)
            int_finite = np.isfinite(xi) & np.isfinite(yi) & np.isfinite(zi)
        values = [np.full(n, np.nan, dtype=np.float64) for fn in functions] # XXX use float32?
        # evaluate functions on cylinder layers
        cyl_finite = int_finite & cyl_closer
        selection = cyl_finite
        ids = np.unique(next_id[selection])
        for id in ids:
            selection_id = selection & (next_id == id)
            lay = self.cyln.cylinders[id]
            if points is not None:
                val = lay.evaluateLayerFunctions(None, None, None, functions, points=points[selection_id,:])
            else:
                val = lay.evaluateLayerFunctions(xi[selection_id], yi[selection_id], zi[selection_id], functions)
            for dst, src in zip(values, val):
                if src is not None:
                    dst[selection_id] = src
        # evaluate functions on cap layers
        # XXX unify with cylinder code path?
        cap_finite = int_finite & np.logical_not(cyl_closer)
        selection = cap_finite
        ids = np.unique(next_id[selection])
        for id in ids:
            selection_id = selection & (next_id == id)
            lay = self.capn.caps[id]
            if points is not None:
                val = lay.evaluateLayerFunctions(None, None, None, functions, points=points[selection_id,:])
            else:
                val = lay.evaluateLayerFunctions(xi[selection_id], yi[selection_id], zi[selection_id], functions)
            for dst, src in zip(values, val):
                if src is not None:
                    dst[selection_id] = src
        return values

    def findIntersectionNeighborhood(self, xi, yi, zi, cyl_closer, next_id, radius):
        nb_dfs = []
        int_finite = np.isfinite(xi) & np.isfinite(yi) & np.isfinite(zi)
        # find cylinder neighborhoods
        cyl_finite = int_finite & cyl_closer
        nb_dfs += self._findNeighborhoodsGroupedByIdAndRadius(xi, yi, zi, cyl_finite, next_id, radius,
            self._findCylinderNeighborhood)
        # find cap neighborhoods
        cap_finite = int_finite & ~cyl_closer
        nb_dfs += self._findNeighborhoodsGroupedByIdAndRadius(xi, yi, zi, cap_finite, next_id, radius,
            self._findCapNeighborhood)
        # combine results
        nb_df = pd.concat(nb_dfs, ignore_index=True) if nb_dfs else None
        return nb_df

    def findIntersectionNeighborhoodK(self, xi, yi, zi, cyl_closer, next_id, k):
        nb_dfs = []
        int_finite = np.isfinite(xi) & np.isfinite(yi) & np.isfinite(zi)
        # find cylinder neighborhoods
        cyl_finite = int_finite & cyl_closer
        nb_dfs += self._findNeighborhoodsGroupedById(xi, yi, zi, cyl_finite, next_id,
            self._findCylinderNeighborhoodK, k=k)
        # find cap neighborhoods
        cap_finite = int_finite & ~cyl_closer
        nb_dfs += self._findNeighborhoodsGroupedById(xi, yi, zi, cap_finite, next_id,
            self._findCapNeighborhoodK, k=k)
        # combine results
        nb_df = pd.concat(nb_dfs, ignore_index=True) if nb_dfs else None
        return nb_df

    def findFirstHitNeighborhood(self):
        # XXX would be cleaner to return an np.array of hit_ids from this function
        dfs = []
        for cyl_id, cyl in enumerate(self.cyln.cylinders):
            parname = 'nb__cyl_first_dz_%d' % cyl_id
            if parname in self.params:
                dfs.append(cyl.centralNeighborhood(absz=self.params[parname])) # XXX refine

        # find potential first hits on caps
        r2_min = self.capn.capspec.caps_df['r2_min'].values
        r2_first = self.capn.capspec.caps_df['r2_first'].values
        center = len(self.capn.caps) // 2
        for i in range(center):
            parname = 'nb__cap_first_dr2_%d' % i
            for cap_id in (center + i, center - i - 1):
                # by default, look at the part of the surface which is exposed to rays from the center
                # (plus a configurable margin)
                r2 = r2_first[cap_id] + self.params['nb__cap_first_inc'] # XXX refine
                # optionally, override the neighborhood radius by parameter
                if parname in self.params:
                    r2 = self.params[parname] # XXX refine
                if r2 > 0:
                    dfs.append(self.capn.caps[cap_id].centralNeighborhood(dr2=r2))

        return pd.concat(dfs)
