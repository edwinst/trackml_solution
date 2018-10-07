"""Euclidean geometry of detector layers and idealized helices.

The functions in this file calculate intersections between an idealized
detector geometry made out of perfect cylinders and caps (planar disks)
with idealized helical particle trajectories.

There are basically no heuristics in this file, except in some cases
where some values are overridden heuristically to avoid numerical
instabilities.

The intersection finding methods allow specifying a HelixCorrector for
applying perturbative corrections to the idealized helix predictions,
but the corrections themselves are not calculated in here (see
corrections.py for that).

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

import sys

import pandas as pd
import numpy as np
from sklearn.cluster import MeanShift, KMeans

def df_vec2_length(df, prefix):
    """Return the length of a vector in the x,y-plane defined by coordinates in
    the given dataframe."""
    return np.sqrt(np.sum([np.square(df[prefix + comp]) for comp in ('x', 'y')],axis=0))

class GeometrySpec:
    """Holds geometrical data about detector module positions and orientations.
    """
    def from_detectors_csv(filename):
        detectors_df = pd.read_csv(filename, header=0, index_col=False,
            dtype={
                'volume_id': 'i1',
                'layer_id': 'i1',
                'module_id': 'i2',
                'cx': 'f4',
                'cy': 'f4',
                'cz': 'f4',
                'rot_xu': 'f4',
                'rot_xv': 'f4',
                'rot_xw': 'f4',
                'rot_yu': 'f4',
                'rot_yv': 'f4',
                'rot_yw': 'f4',
                'rot_zu': 'f4',
                'rot_zv': 'f4',
                'rot_zw': 'f4',
                'module_t': 'f4',
                'module_minhu': 'f4',
                'module_maxhu': 'f4',
                'module_hv': 'f4',
                'pitch_u': 'f4',
                'pitch_v': 'f4'})
        geospec = GeometrySpec()
        geospec.detectors_df = detectors_df
        return geospec

class CylindersSpec:
    """Holds geometrical information about the cylinder layers of the detector.
    """
    cylinder_volume_ids = [8, 13, 17]

    def __init__(self, geospec):
        cyl_modules_df = geospec.detectors_df.loc[geospec.detectors_df['volume_id'].isin(self.cylinder_volume_ids)].copy()
        cyl_modules_df['cr2'] = df_vec2_length(cyl_modules_df, 'c')
        cyl_modules_df['abscz'] = cyl_modules_df['cz'].abs()
        cylinders = cyl_modules_df.groupby(['volume_id', 'layer_id']).agg({'cr2': ['mean', 'max'], 'abscz': 'max', 'module_hv': 'max'})
        cylinders.columns = ['_'.join(col) for col in cylinders.columns.values]
        cylinders = cylinders.sort_values('cr2_mean')
        cylinders['absz_max'] = cylinders['abscz_max'] + cylinders['module_hv_max']
        cylinders['cyl_id'] = range(len(cylinders))
        cylinders.rename(columns={'cr2_mean': 'cyl_r2'}, inplace=True)
        # We assume that cylinders cover monotonically increasing ranges of z from inside out
        assert np.all(np.diff(cylinders['absz_max']) >= 0)
        # absz_cummax gives the largest z-range covered by the respective cylinder or any
        # cylinder inside it.
        # Note: Under current assumptions this should be equal to absz_max, actually.
        self.cyl_absz_max = cylinders['absz_max'].values
        cylinders['absz_cummax'] = cylinders['absz_max'].cummax(axis=0)
        self.cyl_absz_cummax = cylinders['absz_cummax'].values
        self.cyl_absz_cuts = np.unique(self.cyl_absz_cummax)
        absz_max_binned = np.digitize(self.cyl_absz_cummax, self.cyl_absz_cuts, right=True)
        assert np.all(np.diff(absz_max_binned) >= 0)
        binned_unique, binned_indices = np.unique(absz_max_binned, return_index=True)
        # every cut (as the right edge of its bin) must have been hit by at least one absz_cummax
        assert len(binned_unique) == len(self.cyl_absz_cuts)
        # cyl_absz_min_cyl_id[np.digitize(Z, cyl_absz_cuts)] is the minimum cyl_id
        # that can have an intersection at abs(z) > Z
        # Note: We append a dummy value for abs(z) > the range of the longest cylinder
        #       to avoid indexing problems. This case must be handled specially by
        #       user code.
        self.cyl_absz_min_cyl_id = np.append(binned_indices, [0], axis=0).astype(np.int8)
        self.cylinders_df = cylinders
        # squares of the cylinder radii:
        self.cyl_rsqr = np.square(cylinders['cyl_r2'].values)

    def __len__(self):
        """Return the number of cylinder layers."""
        return len(self.cylinders_df)

class CapsSpec:
    """Holds geometrical information about the cap layers of the detector.
    """
    cap_volume_ids = [7, 9, 12, 14, 16, 18]

    def __init__(self, geospec):
        # select all modules which are part of cap volumes
        cap_modules_df = geospec.detectors_df.loc[geospec.detectors_df['volume_id'].isin(self.cap_volume_ids)].copy()
        # (cylinder) radius of module centers, and approximate extreme values in cylinder-radial direction
        cap_modules_df['cr2'] = df_vec2_length(cap_modules_df, 'c')
        cap_modules_df['r2_max'] = cap_modules_df['cr2'] + cap_modules_df['module_hv']
        cap_modules_df['r2_min'] = cap_modules_df['cr2'] - cap_modules_df['module_hv']

        # cluster radial extends to find the geometry of the gaps between rings of modules in the caps
        # Note: We prescribe the number of rings so we can use the much faster KMeans clusterer.
        nrings = 7
        kmeans = KMeans(n_clusters=nrings, n_init=1, random_state=1)
        kmeans.fit(cap_modules_df['r2_max'].values.reshape(-1, 1))
        ring_r2_max = np.sort(kmeans.cluster_centers_.flatten())
        kmeans = KMeans(n_clusters=nrings, n_init=1, random_state=1)
        kmeans.fit(cap_modules_df['r2_min'].values.reshape(-1, 1))
        ring_r2_min = np.sort(kmeans.cluster_centers_.flatten())
        ring_overlap = ring_r2_max[:-1] - ring_r2_min[1:]
        # we expect to find 3 gaps left by the rings (the first one is a small gap between the first two rings)
        is_ring_gap = ring_overlap < 0
        assert np.sum(is_ring_gap) == 3
        self.ring_gap_r2_min = ring_r2_max[ :-1][is_ring_gap]
        self.ring_gap_r2_max = ring_r2_min[1:  ][is_ring_gap]
        self.ring_gap_r2_min_sqr = np.square(self.ring_gap_r2_min)
        self.ring_gap_r2_max_sqr = np.square(self.ring_gap_r2_max)
        # store regions where the cap module rings overlap radially
        self.ring_overlap_r2_min = ring_r2_min[1:  ][~is_ring_gap]
        self.ring_overlap_r2_max = ring_r2_max[ :-1][~is_ring_gap]
        self.ring_overlap_r2_min_sqr = np.square(self.ring_overlap_r2_min)
        self.ring_overlap_r2_max_sqr = np.square(self.ring_overlap_r2_max)

        # aggregate data by cap (volume_id, layer_id), sort by z-coordinate
        caps = cap_modules_df.groupby(['volume_id', 'layer_id']).agg({'cz': 'mean', 'r2_min': 'min', 'r2_max': 'max' })
        caps = caps.sort_values('cz')

        # cluster cap layers into z-layers
        meanshift = MeanShift(bandwidth = 2.0)
        meanshift.fit(caps['cz'].values.reshape(-1, 1))
        caps['cap_cz'] = meanshift.cluster_centers_[meanshift.labels_]
        caps['cap_dz'] = caps['cz'] - caps['cap_cz']

        # assign unique cap_ids to the clustered z-layers
        _, indices = np.unique(meanshift.labels_, return_index=True)
        cap_id_codes = np.zeros(len(indices), dtype=np.int8)
        cap_id_codes[np.argsort(indices)] = list(range(len(indices)))
        caps['cap_id'] = cap_id_codes[meanshift.labels_]

        # aggregate data per cap_id (z-layer), also sorts by cap_id
        caps_df = caps.groupby('cap_id').agg({'r2_max': 'max', 'r2_min': 'min', 'cap_id': 'first', 'cap_cz': 'first'})

        # find central areas of caps which are the first detector surfaces exposed
        # to straight rays from the origin
        ncaps = len(caps_df)
        icenter = ncaps // 2 # [icenter] is the first cap towards positive z
        cz = caps_df['cap_cz'].values
        r2_min = caps_df['r2_min'].values
        r2_first = np.zeros_like(r2_min)
        for direction, i_first in ((-1, icenter-1), (1, icenter)):
            r2_min_before = r2_min[i_first]
            cz_before = cz[i_first]
            for i in range(1, icenter):
                i_this = i_first + direction * i
                r2_first[i_this] = r2_min_before * cz[i_this] / cz_before
                r2_min_before = min(r2_first[i_this], r2_min[i_this])
                cz_before = cz[i_this]
                if r2_first[i_this] < r2_min[i_this]:
                    r2_first[i_this] = 0.0
        caps_df['r2_first'] = r2_first

        # cap_layers_df: pd.DataFrame, indexed by (volume_id, layer_id), maps to cap_id
        self.cap_layers_df = caps

        # caps_df: pd.DataFrame, holds summary data per cap_id
        self.caps_df = caps_df[['cap_id', 'r2_min', 'r2_max', 'r2_first', 'cap_cz']]

        # prepare attributes used directly by the CapIntersector
        self.caps_r2_max = self.caps_df['r2_max'].values
        self.caps_r2_max_sqr = np.square(self.caps_r2_max)
        self.caps_r2_min = self.caps_df['r2_min'].values
        self.caps_r2_min_sqr = np.square(self.caps_r2_min)
        self.cap_z = np.unique(caps['cap_cz'].values)

    def __len__(self):
        """Return the number of cap layers."""
        return len(self.caps_df)

class DetectorSpec:
    """Holds geometrical information about cylinder and cap layers of the detector
    and about module positions and orientations.
    """
    def __init__(self, geospec):
        self.geospec = geospec
        self.cylinders = CylindersSpec(geospec)
        self.caps = CapsSpec(geospec)

    def from_detectors_csv(filename):
        geospec = GeometrySpec.from_detectors_csv(filename)
        return DetectorSpec(geospec)

class CylinderIntersector:
    """Calculates intersections between helices and cylinder layers.
    """
    def __init__(self, cylspec):
        self.cylspec = cylspec

    def intersectFromInside(self, cyl_id, v, dir):
        """Intersect straight rays from inside of the cylinder with the cylinder.
        Note: This function works for both 2-dimensional and 3-dimensional coordinates.
        XXX: Split arguments into coordinates?
        """
        # cylinder radius
        r2 = self.cylspec.cylinders_df.iloc[cyl_id]['cyl_r2']
        # vertex position in the x,y plane
        v2 = v[:,0:2]
        # square of vertex radius in the x,y plane minus square of cylinder radius
        r2sqrd = np.einsum('ij,ij->i', v2, v2) - r2**2
        # assert all vertices are inside the cylinder
        assert np.all(r2sqrd < 0)
        # normalize direction in the x,y plane
        dir2 = dir[:,0:2]
        dirnorm = dir / np.linalg.norm(dir2, axis=1)[:,np.newaxis]
        dir2norm = dirnorm[:,0:2]
        # x,y vertex position projected onto the direction in the x,y plane
        v2in = np.einsum('ij,ij->i', v2, dir2norm)
        # discriminant of quadratic equation
        disc = np.square(v2in) - r2sqrd
        assert np.all(disc > 0)
        # solve for line parameter
        alpha = -v2in + np.sqrt(disc)
        # intersections
        ints = v + alpha[:,np.newaxis] * dirnorm
        return ints

    def intersectHelices(self, x0, y0, z0, uz0, hel_xm, hel_ym, hel_r, hel_pitch,
                         iterations=10, pre_move=10.0, missable=True,
                         corrector=None):
        """Intersect helical trajectories with the next cylinder hit by the respective trajectory.
        Args:
            x0, y0, z0 (array (n_points,)): coordinates of initial vertices of the trajectories
            uz0 (array (n_points,)): estimated tangent vector z-component of the trajectories
                Note: Only the sign of uz0 is used in the calculation.
            hel_xm, hel_ym (array (n_points,)): center coordinates of helices in the x,y-plane
            hel_r (array (n_points,)): radii of the helices in the x,y-plane
            hel_pitch (array (n_points,)): pitches of the helices along the z-axis
            iterations (positive int): maximum number of iterations of intersection finding to do
                Note: Multiple iterations are needed when helices intersect cylinders in the
                      x,y-plane but miss the cylinder in z-coordinate.
            pre_move (float or array (n_points,)): minimum distance to move (when projected
                to the x,y-plane) along the helix before considering the next intersection
                May be negative if the intention is to hit the intersection point again.
            missable...if True, detect cases where a layer is laterally missed, and in these cases
                try to find the following layer intersection
            corrector (None or HelixCorrector): if given, apply this helix corrector to correct
                the predicted intersection points
        Returns:
            xi (array (n_points,)): x-coordinate of estimated next hit, or np.nan if none
            yi (array (n_points,)): y-coordinate of estimated next hit, or np.nan if none
            zi (array (n_points,)): z-coordinate of estimated next hit, or np.nan if none
            cyl_id (integer array (n_points,)): cyl_id of estimated next hit, or -1 if none
            hel_dphi (array (n_points,)): helix phase difference to next hit, or np.nan if none
            hel_s (array (n_points,)): positive arc length until next hit, or np.inf if none
            mult (integer array (n_points,)): estimated maximum multiplicity of the intersection
        """
        # In the variables r2sqr = (x^2 + y^2) and z, a helical trajectory has harmonic motion
        # according to:
        #     r2sqr = (hel_rm^2 + hel_r^2) + 2*hel_rm*hel_r * cos(2*np.pi*(z - z0)/hel_pitch + phim0)
        # where
        #     hel_rm = np.sqrt(hel_xm^2 + hel_ym^2)
        #     phim0 is the azimutal phase relative to the point of largest (x^2 + y^2):
        #     phim0 = np.arctan2(y0 - hel_ym, x0 - hel_xm) - np.arctan2(hel_ym, hel_xm)
        #
        # max r2sqr = (hel_rm + hel_r)^2
        # min r2sqr = (hel_rm - hel_r)^2
        #
        # We bin by r2sqr such that points inside the innermost cylinder get
        # index 0, points between the innermost and next-to-innermost cylinder
        # get index 1, and so on.
        #
        # A helix can only intersect a cylinder if the indices for max(r2sqr) and min(r2sqr)
        # differ and, more specifically, min_bin <= cyl_id < max_bin.
        #
        # x,y-coordinates relative to the helix center, and the current phase angle
        dx = x0 - hel_xm
        dy = y0 - hel_ym
        phi0 = np.arctan2(dy, dx)
        # distance from origin to helix center in the x,y-plane
        hel_rm = np.sqrt(np.square(hel_xm) + np.square(hel_ym))
        # phase relative to the phase of maximum distance to the origin in the x,y-plane
        phim0 = phi0 - np.arctan2(hel_ym, hel_xm)
        # normalize phim0 to the range [0, 2*np.pi)
        phim0[phim0 < 0] += 2 * np.pi
        phim0[phim0 >= 2 * np.pi] -= 2 * np.pi
        max_r2sqr = np.square(hel_rm + hel_r)
        min_r2sqr = np.square(hel_rm - hel_r)
        max_bin = np.digitize(max_r2sqr, self.cylspec.cyl_rsqr)
        min_bin = np.digitize(min_r2sqr, self.cylspec.cyl_rsqr)
        # calculate the current x,y-radius squared for binning
        # Note: For this calculation we shift the phase angle a bit in order
        # not to get stuck in trajectories which currently are exactly at
        # an intersection point.
        dphi_tolerance = pre_move / hel_r
        hel_rm_sqr_plus_hel_r_sqr = np.square(hel_rm) + np.square(hel_r)
        two_hel_rm_hel_r = 2 * hel_rm * hel_r
        r2sqr0 = hel_rm_sqr_plus_hel_r_sqr + two_hel_rm_hel_r * np.cos(phim0 + np.sign(uz0 * hel_pitch) * dphi_tolerance)
        current_bin = np.digitize(r2sqr0, self.cylspec.cyl_rsqr)
        can_intersect = max_bin > min_bin
        # check which helices can no longer intersect any cylinder since they are
        # beyond the z-range covered by the cylinders they can reach
        moving_outward = (z0 * uz0) > 0
        if missable:
            can_intersect[moving_outward & (np.abs(z0) > np.take(self.cylspec.cyl_absz_cummax, max_bin - 1, mode='clip'))] = False
        # Note: can_intersect is automatically False for cases with (hel_rm == 0) or (hel_r == 0)
        # calculate the sign of (d/dt r2sqr) at z=z0:
        sign_r2sqr_dot = np.sign( -hel_pitch * uz0 * np.sin(phim0) )
        # at points where the derivative is zero, we use the second derivative to predict the sign
        # XXX check how this works for hel_pitch == 0 or uz0 == 0
        # XXX for sin(...)==0 this is not really needed due to the handling of the extreme bins below
        where_zero = np.where(sign_r2sqr_dot == 0)[0]
        sign_r2sqr_dot[where_zero] = np.sign( -np.cos(phim0[where_zero]) )
        # If the trajectory is already in the maximum r2sqr bin, it will
        # need to fall in r2sqr again to hit a cylinder, and vice versa.
        sign_r2sqr_dot[current_bin == max_bin] = -1
        sign_r2sqr_dot[current_bin == min_bin] =  1
        # we can now calculate the cyl_id of the next cylinder being hit
        # Note: This does not yet take into acount the finite lengths of the cylinders.
        next_cyl_id = (current_bin + (sign_r2sqr_dot - 1) / 2).astype(np.int8)
        # Now bump the next_cyl_id upward for outward moving helices which can
        # only ever more hit higher layer cylinders:
        # Note: This cannot set next_cyl_id to a cylinder that is too large to be
        #       reached by the helix, because we checked that case above by comparing
        #       with cyl_absz_cummax. HOWEVER, this holds only for missable == True.
        #       Therefore we must only do this if missable, which is fine, since this
        #       is about laterally missing cylinders.
        if missable:
            min_cyl_id_bins = np.digitize(np.abs(z0[moving_outward]), self.cylspec.cyl_absz_cuts).astype(np.int8)
            next_cyl_id[moving_outward] = np.maximum(next_cyl_id[moving_outward],
                                                     self.cylspec.cyl_absz_min_cyl_id[min_cyl_id_bins])
        next_cyl_id[~can_intersect] = 0 # dummy value to avoid indexing errors
        next_r2sqr = self.cylspec.cyl_rsqr[next_cyl_id]
        can_intersect &= (next_r2sqr <= max_r2sqr)
        can_intersect &= (next_r2sqr >= min_r2sqr)
        next_cos = (next_r2sqr - hel_rm_sqr_plus_hel_r_sqr) / two_hel_rm_hel_r
        next_cos[~can_intersect] = 0.0 # dummy value to avoid warnings about impossible cos values
        next_arccos = np.arccos(next_cos) # in [0; pi]
        # expected sign of dphi
        sign_dphi = np.sign(uz0 * hel_pitch)
        # candidates for abs(dphi)
        # Note: In general, a particular cylinder is crossed at two differnt phase
        #       angles of the helix. Application of the arccos function provides us
        #       with two forward and two backward intersection points. We choose the one
        #       which is closest to the current phase angle in the right direction.
        cand_dphi = np.zeros((x0.shape[0], 4))
        cand_dphi[:,0] =  next_arccos - phim0     # in [  0; pi] + (-2pi, 0] = (-2 pi;   pi]
        cand_dphi[:,1] = -next_arccos - phim0     # in [-pi;  0] + (-2pi, 0] = (-3 pi;    0]
        cand_dphi[:,2] = cand_dphi[:,0] + 2*np.pi # in                         (    0; 3 pi]
        cand_dphi[:,3] = cand_dphi[:,1] + 2*np.pi # in                         (-  pi; 2 pi]
        cand_dphi *= sign_dphi[:,np.newaxis] # takes direction into account
        # reject dphi values with the wrong sign or too close to the current phase
        cand_dphi[cand_dphi < dphi_tolerance[:,np.newaxis]] = np.inf
        # selected smallest absolute value and apply sign
        dphi = sign_dphi * np.amin(cand_dphi, axis=1)
        # discard the trajectories for which no finite dphi could be found
        can_intersect[np.isinf(dphi)] = False
        dphi[~can_intersect] = 0.0 # dummy value to avoid warnings about invalid arguments to cos/sin
        # calculate intersections
        hel_p = hel_pitch / (2 * np.pi)
        cos_dphi = np.cos(dphi)
        sin_dphi = np.sin(dphi)
        xi = hel_xm + cos_dphi * dx - sin_dphi * dy
        yi = hel_ym + sin_dphi * dx + cos_dphi * dy
        zi = z0 + dphi * hel_p
        # apply corrections, if any
        if corrector is not None:
            xi, yi, zi = corrector.correctCylinderIntersections(xi, yi, zi, next_cyl_id, mask=can_intersect)
        # calculate arc length until next hit
        hel_s = np.sqrt(np.square(hel_r) + np.square(hel_p)) * np.abs(dphi)
        # estimate multiplicity of the intersection
        # XXX for cylinders, we currently always assume the maximum of 4
        mult = np.full(x0.shape[0], 4, dtype=np.int8)
        if missable:
            # check if the intersection is beyond the length of the cylinder
            is_beyond = moving_outward & can_intersect & (np.abs(zi) > self.cylspec.cyl_absz_max[next_cyl_id])
            if np.any(is_beyond) and iterations > 1:
                # recurse to find the following intersections for the helices that
                # missed their cylinder in z-coordinate
                x0_next = xi[is_beyond]
                y0_next = yi[is_beyond]
                z0_next = zi[is_beyond]
                if corrector is not None:
                    corrector_next = corrector.subset(is_beyond)
                    hel_params_next = corrector_next.helixParams()
                else:
                    corrector_next = None
                    hel_params_next = (par[is_beyond] for par in (uz0, hel_xm, hel_ym, hel_r, hel_pitch))
                (xi_next, yi_next, zi_next, cyl_id_next, dphi_next, hel_s_next, mult_next) = self.intersectHelices(
                    x0_next, y0_next, z0_next, *hel_params_next, corrector=corrector_next,
                    iterations = iterations - 1)
                np.place(xi, is_beyond, xi_next)
                np.place(yi, is_beyond, yi_next)
                np.place(zi, is_beyond, zi_next)
                np.place(next_cyl_id, is_beyond, cyl_id_next)
                dphi[is_beyond] += dphi_next   # phase difference is accumulated!
                hel_s[is_beyond] += hel_s_next # acr length is accumulated!
                np.place(mult, is_beyond, mult_next)
            else:
                can_intersect &= ~is_beyond
        # set values to np.nan/-1 where we have no intersections
        xi[~can_intersect] = np.nan
        yi[~can_intersect] = np.nan
        zi[~can_intersect] = np.nan
        next_cyl_id[~can_intersect] = -1
        dphi[~can_intersect] = np.nan
        hel_s[~can_intersect] = np.inf
        mult[~can_intersect] = 0
        return (xi, yi, zi, next_cyl_id, dphi, hel_s, mult)

class CapIntersector:
    """Calculates intersections between helices and cap layers.
    """
    def __init__(self, capspec):
        self.capspec = capspec

    def intersectHelices(self, x0, y0, z0, uz0, hel_xm, hel_ym, hel_r, hel_pitch,
                         iterations=10, pre_move=50.0, missable=True,
                         corrector=None):
        """Intersect helical trajectories with the next cap hit by the respective trajectory.
        Args:
            x0, y0, z0 (array (n_points,)): coordinates of initial vertices of the trajectories
            uz0 (array (n_points,)): estimated tangent vector z-component of the trajectories
                Note: Only the sign of uz0 is used in the calculation.
            hel_xm, hel_ym (array (n_points,)): center coordinates of helices in the x,y-plane
            hel_r (array (n_points,)): radii of the helices in the x,y-plane
            hel_pitch (array (n_points,)): pitches of the helices along the z-axis
            iterations (positive int): maximum number of iterations of intersection finding to do
                Note: Multiple iterations are needed when helices pass z-coordinates of caps
                      but miss the cap radially.
            pre_move (float or array (n_points,)): minimum distance in z-coordinate to move
                along the helix before considering an intersection. May be negative if the
                intention is to hit the previous intersection point again.
            missable...if True, detect cases where a layer is laterally missed, and in these cases
                try to find the following layer intersection
        Returns:
            xi (array (n_points,)): x-coordinate of estimated next hit, or np.nan if none
            yi (array (n_points,)): y-coordinate of estimated next hit, or np.nan if none
            zi (array (n_points,)): z-coordinate of estimated next hit, or np.nan if none
            cap_id (integer array (n_points,)): cap_id of estimated next hit, or -1 if none
            hel_dphi (array (n_points,)): helix phase difference to next hit, or np.nan if none
            hel_s (array (n_points,)): positive arc length until next hit, or np.inf if none
            mult (integer array (n_points,)): estimated maximum multiplicity of the intersection
            corrector (None or HelixCorrector): if given, apply this helix corrector to correct
                the predicted intersection points
        """
        # First we find which cap is the next to be possibly hit by each trajectory.
        # We do this by binning the z-coordinate. In order not to hit the cap of the
        # last hit twice, we firts shift the z-coordinates (only for binning) in the
        # direction the trajectory is moving.
        sign_uz0 = np.sign(uz0)
        binning_z = z0 + sign_uz0 * pre_move
        bins = np.digitize(binning_z, self.capspec.cap_z)
        next_cap_id = (bins + (sign_uz0 - 1) / 2).astype(np.int8)
        # Note: Trajectories leaving the detector towards -z have next_cap_id == -1
        #       Trajectories leaving the detector towards +z have next_cap_id == len(self.capspec.cap_z)
        can_hit = (sign_uz0 != 0) & (next_cap_id >= 0) & (next_cap_id < len(self.capspec.cap_z))
        where_can_hit = np.where(can_hit)[0]
        next_cap_id_ch = next_cap_id.compress(can_hit) # which is faster, compress(can_hit) of [where_can_hit]?
        # calculate z-coordinate of next cap crossing and its difference to the current position
        next_z = self.capspec.cap_z[next_cap_id_ch]
        next_dz = next_z - z0.compress(can_hit)
        # using the helix pitch, calculate change of helix-azimuth until the next hit
        hel_p_ch = hel_pitch.compress(can_hit) / (2 * np.pi)
        dphi = next_dz / hel_p_ch
        # rotate in x,y-plane to find the predicted intersection point
        hel_xm_ch = hel_xm.compress(can_hit)
        hel_ym_ch = hel_ym.compress(can_hit)
        x_ch = x0.compress(can_hit) - hel_xm_ch
        y_ch = y0.compress(can_hit) - hel_ym_ch
        cos_dphi = np.cos(dphi)
        sin_dphi = np.sin(dphi)
        next_x = hel_xm_ch + cos_dphi * x_ch - sin_dphi * y_ch
        next_y = hel_ym_ch + sin_dphi * x_ch + cos_dphi * y_ch
        # apply corrections, if any
        if corrector is not None:
            corrector_ch = corrector.subset(can_hit)
            next_x, next_y = corrector_ch.correctCapIntersections(next_x, next_y, next_z, next_cap_id_ch)
        # assemble data about predicted intersections
        xi = np.full(x0.shape, np.nan)
        yi = np.full(y0.shape, np.nan)
        zi = np.full(y0.shape, np.nan)
        cap_id = np.full(x0.shape, -1, dtype=np.int8)
        xi[where_can_hit] = next_x
        yi[where_can_hit] = next_y
        zi[where_can_hit] = next_z
        cap_id[where_can_hit] = next_cap_id_ch
        # store phase difference until next hit
        hel_dphi = np.full(x0.shape, np.nan)
        hel_dphi[where_can_hit] = dphi
        # calculate arc length until next hit
        hel_s_ch = np.sqrt(np.square(hel_r.compress(can_hit)) + np.square(hel_p_ch)) * np.abs(dphi)
        hel_s = np.full(x0.shape, np.inf)
        hel_s[where_can_hit] = hel_s_ch
        # estimate multiplicity of the intersection
        # XXX for some reason, restricting to 2 vs. 4 intersections does not work well. Need to investiate sometime.
        #     For now, let's restrict to 3 or 4 intersections, respectively.
        ri2_sqr_ch = np.square(next_x) + np.square(next_y)
        mult = np.zeros(x0.shape, dtype=np.int8)
        mult[where_can_hit] = 3
        for r2_min, r2_max in zip(self.capspec.ring_overlap_r2_min_sqr, self.capspec.ring_overlap_r2_max_sqr):
            mult[where_can_hit] += 1 * ((ri2_sqr_ch > r2_min) & (ri2_sqr_ch < r2_max))
        if missable:
            # check if the interaction is beyond the radius of the cap
            cap_r2_max_sqr_ch = self.capspec.caps_r2_max_sqr[next_cap_id_ch]
            cap_r2_min_sqr_ch = self.capspec.caps_r2_min_sqr[next_cap_id_ch]
            # XXX refine, maybe introduce marginal region in which candidates are extended/(kept with nskipped++)
            is_beyond = (ri2_sqr_ch < cap_r2_min_sqr_ch) | (ri2_sqr_ch > cap_r2_max_sqr_ch)
            # detect false intersections in the gaps between cap module rings, XXX refine
            for r2_min, r2_max in zip(self.capspec.ring_gap_r2_min_sqr, self.capspec.ring_gap_r2_max_sqr):
                is_beyond |= ((ri2_sqr_ch > r2_min) & (ri2_sqr_ch < r2_max))
            where_is_beyond = where_can_hit[is_beyond]
            if where_is_beyond.size > 0 and iterations > 1:
                # recurse to find the following intersections for the helices that
                # missed their cap
                x0_next = next_x[is_beyond]
                y0_next = next_y[is_beyond]
                z0_next = next_z[is_beyond]
                if corrector is not None:
                    corrector_next = corrector_ch.subset(is_beyond)
                    hel_params_next = corrector_next.helixParams()
                else:
                    corrector_next = None
                    hel_params_next = (par[where_is_beyond] for par in (uz0, hel_xm, hel_ym, hel_r, hel_pitch))
                (xi_next, yi_next, zi_next, cap_id_next, dphi_next, hel_s_next, mult_next) = self.intersectHelices(
                    x0_next, y0_next, z0_next, *hel_params_next, corrector=corrector_next,
                    iterations = iterations - 1)
                xi[where_is_beyond] = xi_next
                yi[where_is_beyond] = yi_next
                zi[where_is_beyond] = zi_next
                cap_id[where_is_beyond] = cap_id_next
                hel_dphi[where_is_beyond] += dphi_next # phase difference is accumulated!
                hel_s[where_is_beyond] += hel_s_next # acr length is accumulated!
                mult[where_is_beyond] = mult_next
            else:
                xi[where_is_beyond] = np.nan
                yi[where_is_beyond] = np.nan
                zi[where_is_beyond] = np.nan
                cap_id[where_is_beyond] = -1
                hel_dphi[where_is_beyond] = np.nan
                hel_s[where_is_beyond] = np.inf
                mult[where_is_beyond] = 0
        return (xi, yi, zi, cap_id, hel_dphi, hel_s, mult)

class Intersector:
    """Calculates intersections between helices and detector layers.
    """
    def __init__(self, spec):
        """Args:
            spec (DetectorSpec): specifies the detector geometry.
        """
        self.spec = spec
        self.cylis = CylinderIntersector(self.spec.cylinders)
        self.capis = CapIntersector(self.spec.caps)

    def findNextHelixIntersection(self, x0, y0, z0, uz0, hel_xm, hel_ym, hel_r, hel_pitch,
                                  cyl_pre_move=None, cap_pre_move=None, missable=True, force_cyl_closer=None,
                                  corrector=None):
        """Find the next intersection of each helix with the detector structure.
        Args:
            x0, y0, z0 (array (n_points,)): coordinates of initial vertices of the trajectories
            uz0 (array (n_points,)): estimated tangent vector z-component of the trajectories
                Note: Only the sign of uz0 is used in the calculation.
            hel_xm, hel_ym (array (n_points,)): center coordinates of helices in the x,y-plane
            hel_r (array (n_points,)): radii of the helices in the x,y-plane
            hel_pitch (array (n_points,)): pitches of the helices along the z-axis
            cyl_pre_move (None or 'back' or float or array (n_points,)): minimum distance to move
                (when projected to the x,y-plane) along the helix before considering the next
                cylinder intersection.
                May be negative if the intention is to hit the previous intersection point again.
                None.....chose value heuristically to avoid repeating previous intersections
                'back'...chose value heuristically to ensure repeating previous intersections
            cap_pre_move (None or 'back' or float or array (n_points,)): minimum distance in
                z-coordinate to move along the helix before considering the next cap intersection.
                May be negative if the intention is to hit the previous intersection point again.
                None.....chose value heuristically to avoid repeating previous intersections
                'back'...chose value heuristically to ensure repeating previous intersections
            missable...if True, detect cases where a layer is laterally missed, and in these cases
                try to find the following layer intersection
            force_cyl_closer (None or bool array (n_points,)): if given, force the next
                intersection to be in a cylinder if True, otherwise force it to be
                in a cap.
            corrector (None or HelixCorrector): if given, apply this helix corrector to correct
                the predicted intersection points
        Returns:
            xi (array (n_points,)): x-coordinate of estimated next hit, or np.nan if none
            yi (array (n_points,)): y-coordinate of estimated next hit, or np.nan if none
            zi (array (n_points,)): z-coordinate of estimated next hit, or np.nan if none
            cyl_closer (boolean array (n_points,)): True if the next intersection is with a cylinder
            next_id (integer array (n_points,)): cyl_id or cap_id of estimated next hit, or -1 if none
            hel_dphi (array (n_points,)): helix phase difference to next hit, or np.nan if none
            hel_s (array (n_points,)): positive arc length until next hit, or np.inf if none
            mult (integer array (n_points,)): estimated maximum multiplicity of the intersection
        """
        if cyl_pre_move is None or cyl_pre_move == 'back':
            d = 10.0 + 0.05 * np.maximum(0, np.sqrt(np.square(x0) + np.square(y0)) - 32.0) # XXX refine
            cyl_pre_move = -d if cyl_pre_move == 'back' else d
        if cap_pre_move is None or cap_pre_move == 'back':
            d = 50.0 # XXX refine
            cap_pre_move = -d if cap_pre_move == 'back' else d
        cyl_corrector = None
        cap_corrector = None
        if corrector is not None:
            cyl_corrector = corrector.clone()
            cap_corrector = corrector
        (cyl_xi, cyl_yi, cyl_zi, cyl_next_id, cyl_dphi, cyl_hel_s, cyl_mult) = self.cylis.intersectHelices(
            x0, y0, z0, uz0, hel_xm, hel_ym, hel_r, hel_pitch,
            pre_move=cyl_pre_move, missable=missable, corrector=cyl_corrector)
        (cap_xi, cap_yi, cap_zi, cap_next_id, cap_dphi, cap_hel_s, cap_mult) = self.capis.intersectHelices(
            x0, y0, z0, uz0, hel_xm, hel_ym, hel_r, hel_pitch,
            pre_move=cap_pre_move, missable=missable, corrector=cap_corrector)
        if force_cyl_closer is None:
            cyl_closer = (cyl_hel_s < cap_hel_s)
        else:
            cyl_closer = force_cyl_closer
        xi = np.where(cyl_closer, cyl_xi, cap_xi)
        yi = np.where(cyl_closer, cyl_yi, cap_yi)
        zi = np.where(cyl_closer, cyl_zi, cap_zi)
        next_id = np.where(cyl_closer, cyl_next_id, cap_next_id)
        dphi = np.where(cyl_closer, cyl_dphi, cap_dphi)
        hel_s = np.where(cyl_closer, cyl_hel_s, cap_hel_s)
        mult = np.where(cyl_closer, cyl_mult, cap_mult)
        if corrector is not None:
            corrector.merge(other=cyl_corrector, use_other=cyl_closer)
        return (xi, yi, zi, cyl_closer, next_id, dphi, hel_s, mult)

def circleFromThreePoints(x1, y1, x2, y2, x3, y3, large_radius=np.float64(1e6)):
    """Calculate the circle going through three points in the x,y-plane.
    Args:
        x1, y1, x2, y2, x3, y3 (array (N,)): x,y-coordinates of the three points.
        large_radius (np.float64): large radius to use instead of the theoretically
            "infinite radius" if the points are exactly collinear.
    Returns:
        xm, ym (array (N,)): x,y-coordinates of the circle centers
        r (array (N,)): circle radii
    """
    x21 = x2 - x1
    y21 = y2 - y1
    x31 = x3 - x1
    y31 = y3 - y1
    x32 = x3 - x2
    y32 = y3 - y2
    # Check whether we have pair-wise distinct points:
    same21 = ~np.logical_or(x21, y21)
    same31 = ~np.logical_or(x31, y31)
    same32 = ~np.logical_or(x32, y32) # XXX used only for diagnosis
    # Note: If all three points are the same, we construct a large circle
    #       which goes approximately through the origin and x1, y1.
    #       If only two points are the same, we construct a large circle
    #       which approximately touches the spanned line segment at x1, y1.
    allsame = same21 & same31
    # coordinate difference used to construct a large tangential circle below
    xd = np.where(allsame, x1, np.where(same21, x31, x21))
    yd = np.where(allsame, y1, np.where(same21, y31, y21))
    if np.any(same21 ): print("PROBLEM12", file=sys.stderr) # XXX solve this
    if np.any(same31 ): print("PROBLEM13", file=sys.stderr) # XXX solve this
    if np.any(same32 ): print("PROBLEM23", file=sys.stderr) # XXX solve this
    if np.any(allsame): print("PROBLEMALL", file=sys.stderr) # XXX solve this
    rsqr1 = np.square(x1) + np.square(y1)
    rsqr2 = np.square(x2) + np.square(y2)
    rsqr3 = np.square(x3) + np.square(y3)
    denom = 2*(y1*x32 - x1*y32 + x2*y3 - x3*y2)
    r_too_large = (denom == 0)
    denom[r_too_large] = 1.0 # dummy value to avoid errors in division. we fix these cases below
    xm = -(rsqr1*y32 - rsqr2*y31 + rsqr3*y21) / denom
    ym =  (rsqr1*x32 - rsqr2*x31 + rsqr3*x21) / denom
    r = np.sqrt(np.square(x1-xm) + np.square(y1-ym))
    # override results for too-large circles
    r_too_large |= (r > large_radius)
    r[r_too_large] = large_radius
    r_over_d = large_radius / np.sqrt(np.square(xd[r_too_large]) + np.square(yd[r_too_large]))
    xm[r_too_large] = x1[r_too_large] - r_over_d * yd[r_too_large]
    ym[r_too_large] = y1[r_too_large] + r_over_d * xd[r_too_large]
    return (xm, ym, r)

def helixPitchFromTwoPoints(x1, y1, z1, x2, y2, z2, hel_xm, hel_ym, zero_pitch=0.001):
    """Calculate the pitch of a helix from two points on the helix,
    when the helix center is known. The points are assumed to be
    separated by less than half a turn of the helix.
    Args:
        x1, y1, z1, x2, y2, z2 (array (N,)): coordinates of the two points
            Note: The order of the points on the helix does not matter for
            the pitch calculation, it only affects the signs of the returned
            phid and dz values.
        hel_xm, hel_ym (array (n_points,)): center coordinates of helices in the x,y-plane
        zero_pitch (float): pitch value to replace zero pitch with to avoid numerical problems
    Returns:
        hel_pitch (array (N,)): helix pitch
        phid (array (N,)): helix polar angle difference between the points
        dz (array (N,)): z-difference of the points. Normally this is (z2 - z1),
            but if the helix pitch would be zero, this is replaced with a
            matching value.
    """
    # calculate phase difference on the fitted helix
    phi1 = np.arctan2(y1 - hel_ym, x1 - hel_xm)
    phi2 = np.arctan2(y2 - hel_ym, x2 - hel_xm)
    phid = phi2 - phi1
    # normalize phid to (-pi, pi]
    phid[phid >   np.pi] -= 2 * np.pi
    phid[phid <= -np.pi] += 2 * np.pi
    # calculate pitch
    dz = z2 - z1
    # avoid dividing by zero; it's rare enough that we accept a potentially nonsensical outcome
    # XXX is there a better solution taking additional points into account?
    phid[phid == 0] = 0.001
    hel_pitch = dz / phid * 2 * np.pi
    # avoid the case of exactly zero pitch
    has_zero_pitch = (hel_pitch == 0.0)
    # XXX activate the following assertion as soon as the NaN helix values do no longer occur
    ###assert np.all(has_zero_pitch == (dz == 0.0))
    hel_pitch[has_zero_pitch] = zero_pitch
    dz[has_zero_pitch] = zero_pitch * phid[has_zero_pitch] / (2 * np.pi)
    return hel_pitch, phid, dz

def helixPitchLeastSquares(x1, y1, z1, x2, y2, z2, x3, y3, z3, hel_xm, hel_ym, zero_pitch=0.001):
    """Calculate the pitch of a helix from three points on the helix,
    when the helix center is known. The points are assumed to be
    in helix path order and separated by less than half a turn of the
    helix between consecutive points. The result is the
    helix pitch that minimizes the squared error in the z-coordinate.
    Args:
        x1, y1, z1, x2, y2, z2, x3, y3, z3 (array (N,)): coordinates of the points
        hel_xm, hel_ym (array (n_points,)): center coordinates of helices in the x,y-plane
        zero_pitch (float): pitch value to replace zero pitch with to avoid numerical problems
    Returns:
        hel_pitch (array (N,)): helix pitch
        phid (array (N,)): helix polar angle difference from point 2 to point 3
        dz (array (N,)): z-difference of the points. Normally this is (z3 - z2),
            but if the helix pitch would be zero, this is replaced with a
            matching value.
        loss (array (N,)): sum of squared residuals (assuming the optimum
            intercept is chosen).
    """
    # calculate phase differences on the fitted helix
    phi1 = np.arctan2(y1 - hel_ym, x1 - hel_xm)
    phi2 = np.arctan2(y2 - hel_ym, x2 - hel_xm)
    phi3 = np.arctan2(y3 - hel_ym, x3 - hel_xm)
    phid21 = phi2 - phi1
    phid31 = phi3 - phi1
    # normalize phid21 to [-pi, pi)
    phid21[phid21 >= np.pi] -= 2 * np.pi
    phid21[phid21 < -np.pi] += 2 * np.pi
    # normalize phid31 to [      phid21, phid21 + pi) for positive phid21
    #              and to (-pi + phid21, phid21     ] for negative phid21
    phid21neg = (phid21 < 0)
    phid31[~phid21neg & (phid31 < phid21)] += 2 * np.pi
    phid31[ phid21neg & (phid31 > phid21)] -= 2 * np.pi
    # calculate pitch
    sum_phi = phid21 + phid31
    sum_phisqr = np.square(phid21) + np.square(phid31)
    sum_z = z1 + z2 + z3
    sum_zphi = z2 * phid21 + z3 * phid31
    n = 3.0
    hel_p = (n * sum_zphi - sum_z * sum_phi) / (n * sum_phisqr - np.square(sum_phi))
    hel_pitch = 2 * np.pi * hel_p
    dz = z3 - z2
    # avoid the case of exactly zero pitch
    has_zero_pitch = (hel_pitch == 0.0)
    # XXX activate the following assertion as soon as the NaN helix values do no longer occur
    ###assert np.all(has_zero_pitch == (dz == 0.0)) XXX adapt assertion
    phid = phid31 - phid21
    hel_pitch[has_zero_pitch] = zero_pitch
    dz[has_zero_pitch] = zero_pitch * phid[has_zero_pitch] / (2 * np.pi)
    # calculate loss
    zs = np.stack([z1, z2, z3], axis=1)
    phis = np.stack([np.zeros_like(phid21), phid21, phid31], axis=1)
    zs   -= (sum_z   / n)[:,np.newaxis]
    phis -= (sum_phi / n)[:,np.newaxis]
    loss = np.sum(np.square(zs - hel_p[:,np.newaxis] * phis), axis=1)
    return hel_pitch, phid, dz, loss

def helixDirectionFromTwoPoints(hel_xm, hel_ym, hel_pitch, x1, y1, z1, x2, y2, z2):
    """Get a value indicating the direction of helix motion in the z-coordinate,
    in the sense of moving from a given first point towards a given second point.
    The sign of this value also determines the sense of rotation in the x,y-plane,
    so it is needed (and must be non-zero) even if there is no real motion
    along the z-axis.
    Args:
        hel_xm, hel_ym, hel_pitch: helix parameters
        x1, y1, z1: first point on the helix
        x2, y2, z2: second point on the helix
    Returns:
        hel_dz: A value, the sign of which specifies the sign of the motion in
            the z-coordinate. Usually, hel_dz is simply the difference z2 - z1,
            but the function needs to take care of some special cases when the
            two points have the same z-coordinate.
    """
    hel_dz    = z2 - z1
    # if dz is zero, choose a hel_dz that gives the correct sense of rotation
    # of the track in the x,y-plane
    nodz = (hel_dz == 0)
    hel_dz[nodz] = hel_pitch[nodz] * np.sign(
        (y1[nodz] - y2[nodz])*(hel_xm[nodz] - x1[nodz])
      + (x2[nodz] - x1[nodz])*(hel_ym[nodz] - y1[nodz]))
    return hel_dz

def helixWithTangentVector(x0, y0, z0, ux0, uy0, uz0, hel_pitch, min_uz0=1e-3, max_mg_over_qB=1e6):
    """Construct helices through the given initial points and
    with the given tangent vectors and pitches.
    Args:
        x0, y0, z0 (array (n_points,)): coordinates of initial vertices on the helices
        ux0, uy0, uz0 (array (n_points,)): tangent vector components
            The magnitude of the tangent vector is not used.
            uz0 must be non-zero.
        hel_pitch (array (n_points,) or scalar): pitch(es) of the helices along the z-axis
            Note: For a physical relativistic particle in a constant magnetic
                  field (0, 0, Bz), hel_pitch = -2 * np.pi * pz / (q * Bz),
                  where
                      pz is the z-component of the momentum
                      q is the particle charge
                      Bz is the z-component of the magnetic field
        min_uz0 (float): minimum absolute value to use for uz0 in order to regulate
            numerical instabilities.
        max_mg_over_qB (float): maximum absolute value to allow for (m gamma / (q * Bz)).
    Returns:
        hel_xm, hel_ym (array (n_points,)): center coordinates of helices in the x,y-plane
        hel_r (array (n_points,)): radii of the helices in the x,y-plane
    """
    uz0_regulated = np.where(np.abs(uz0) >= min_uz0, uz0, min_uz0 * (1 - 2*(uz0 < 0)))
    mg_over_qB = -hel_pitch / (2 * np.pi * uz0_regulated)
    # Note: If u*0 is the velocity, then mg_over_qB = m gamma / (q * Bz),
    # where
    #     m is the rest mass of the particle
    #     gamma is the relativistic gamma factor
    #     q is the particle charge
    #     Bz is the z-component of the magnetic field
    # Clip too-large values of mg_over_qB in order to avoid numerical instabilities:
    too_large = (np.abs(mg_over_qB) > max_mg_over_qB)
    mg_over_qB[too_large] = np.sign(mg_over_qB[too_large]) * max_mg_over_qB
    # calculate helix center and radius in the x,y-plane
    hel_xm = x0 + uy0 * mg_over_qB
    hel_ym = y0 - ux0 * mg_over_qB
    hel_r = np.sqrt(np.square(x0 - hel_xm) + np.square(y0 - hel_ym))
    return (hel_xm, hel_ym, hel_r)

def helixMove(x0, y0, z0, uz0, hel_xm, hel_ym, hel_r, hel_pitch, hel_s):
    """Move the given point along the helix by the given arc length.
    Args:
        x0, y0, z0 (array (n_points,)): coordinates of initial vertices of the trajectories
        uz0 (array (n_points,)): estimated tangent vector z-component of the trajectories
            Note: Only the sign of uz0 is used in the calculation.
        hel_xm, hel_ym (array (n_points,)): center coordinates of helices in the x,y-plane
        hel_r (array (n_points,)): radii of the helices in the x,y-plane
        hel_pitch (array (n_points,)): pitches of the helices along the z-axis
        hel_s (array (n_points,)): arc length to move in direction given by uz0
            (negative values move in negated uz0 direction)
    Returns:
        xf (array (n_points,)): x-coordinate of end point
        yf (array (n_points,)): y-coordinate of end point
        zf (array (n_points,)): z-coordinate of end point
        dphi (array (n_points,)): helix phase difference moved
    """
    # expected sign of dphi for positive arc length
    sign_dphi = np.sign(uz0 * hel_pitch)
    # phase difference to move
    hel_p = hel_pitch / (2 * np.pi)
    dphi = sign_dphi * hel_s / np.sqrt(np.square(hel_r) + np.square(hel_p))
    # calculate target points
    dx = x0 - hel_xm
    dy = y0 - hel_ym
    cos_dphi = np.cos(dphi)
    sin_dphi = np.sin(dphi)
    xf = hel_xm + cos_dphi * dx - sin_dphi * dy
    yf = hel_ym + sin_dphi * dx + cos_dphi * dy
    zf = z0 + dphi * hel_p
    return (xf, yf, zf, dphi)

def helixNearestPointDistance(x0, y0, z0, hel_xm, hel_ym, hel_r, hel_pitch, x, y, z, iterations=3,
                              return_hel_s=False):
        """On each helix, find the point nearest to a respectively given refernce point
        and calculate the Euclidean distance between those points.
        Args:
            x0, y0, z0 (array (n_points,) [*]): coordinates of initial vertices on the helices
            hel_xm, hel_ym (array (n_points,) [*]): center coordinates of helices in the x,y-plane
            hel_r (array (n_points,) [*]): radii of the helices in the x,y-plane
            hel_pitch (array (n_points,) [*]): pitches of the helices along the z-axis
            x, y, z (array (n_points,)): reference point coordinates
            iterations (positive int): number of iteration of Newton's method to perform
            return_hel_s (bool): If True, return dphi and hel_s
        Notes:
           [*]...The helix parameters x0, y0, z0, hel_xm, hel_ym, hel_r, hel_pitch may also be scalars.
                 In that case the same helix is used for each reference point.
        Returns:
            x1 (array (n_points,)): x-coordinate of nearest point
            y1 (array (n_points,)): y-coordinate of nearest point
            z1 (array (n_points,)): z-coordinate of nearest point
            dist (array (n_points,)): Euclidean distance from the reference point to
                the nearest point on the corresponding helix
        Returns optionally:
            dphi (aray (n_points,)): helix phase difference from (x0, y0, z0) to the
                nearest point
            hel_s (aray (n_points,)): positive helix arc length from (x0, y0, z0) to the
                nearest point
        """
        # make x,y-coordinates relative to the helix center
        dx = x - hel_xm
        dy = y - hel_ym
        dx0 = x0 - hel_xm
        dy0 = y0 - hel_ym
        # distance from reference point to helix center in the x,y-plane
        dr2 = np.sqrt(np.square(dx) + np.square(dy))
        # The nearest point (x1, y1, z1) satisfies the equivalent equations:
        #        sin(w + dph_diff) + (1/e) w                  = 0
        #        sin(u)            + (1/e) u - (1/e) dph_diff = 0
        #     -e sin(E)            +       E -       M        = 0
        # where
        #     E = np.pi + u        - 2*k*np.pi
        #     M = np.pi + dph_diff - 2*k*np.pi
        #     e is the "excentricity" coefficient defined below,
        #     u = w + dph_diff
        #         modulo 2*np.pi, u is the difference in polar angle between
        #         the nearest point and the reference point
        #     w = 2*np.pi * (z1 - z) / hel_pitch
        #         is the helix polar angle difference between the nearest point
        #         and the point with the same height (z-coordinate) as the reference point
        #     dph_z = 2*np.pi * (z - z0) / hel_pitch
        #         is the height of the reference point over z0 converted to
        #         a polar angle difference (i.e. fractional measure of turns)
        #         i.e. by advancing (x0, y0, z0) by dph_z along the helix, one
        #         gets to the *height* z of the reference point.
        #     dph_xy = arctan2(dy, dx) - arctan2(dy0, dx0)
        #         is the polar angle of the reference point relative to (x0, y0)
        #         i.e. by advancing (x0, y0, z0) by dph_xy along the helix, one
        #         gets to a point with the same *polar angle* as the reference
        #         point.
        #     dph_diff = dph_z - dph_xy
        #         Modulo 2*np.pi, dph_diff indicates how much the reference point
        #         is "out-of-sync" with the helix, i.e. the difference in polar
        #         angle between a point on the helix at the *same height* as
        #         the reference point and a point on the helix at the *same polar angle*
        #         as the refernce point.
        #
        # phase angles needed for the equation below
        hel_p = hel_pitch / (2 * np.pi)
        dph_z = (z - z0) / hel_p
        dph_xy = np.arctan2(dy, dx) - np.arctan2(dy0, dx0)
        dph_diff = dph_z - dph_xy
        # coefficient of sin(E) in the equation for z1 we are to set up
        e = hel_r * dr2 / np.square(hel_p)
        # We use Newton's method with:
        #     f (E) = E - e sin(E) - M
        #     f'(E) = 1 - e cos(u)
        # and
        #     E_(n+1) = E_n - f(E)/f'(E)
        E = np.full(x.shape[0], np.pi)
        k = np.ceil(dph_diff / (2*np.pi) - 0.5)
        M = np.pi + dph_diff - 2*k*np.pi
        for i in range(iterations):
            f = E - e * np.sin(E) - M
            fprime = 1.0 - e * np.cos(E)
            E = E - f/fprime
        # translate the solution back into the coordinates of the nearest point
        u = E - np.pi + 2*k*np.pi
        w = u - dph_diff
        z1 = z + hel_p * w
        # phase difference between the nearest point and (x0, y0, z0)
        dphi = (z - z0) / hel_p + w
        dx0 = x0 - hel_xm
        dy0 = y0 - hel_ym
        cos1 = np.cos(dphi)
        sin1 = np.sin(dphi)
        x1 = hel_xm + cos1 * dx0 - sin1 * dy0
        y1 = hel_ym + sin1 * dx0 + cos1 * dy0
        dist = np.sqrt(np.square(x - x1) + np.square(y - y1) + np.square(z - z1))
        if return_hel_s:
            hel_s = np.sqrt(np.square(hel_r) + np.square(hel_p)) * np.abs(dphi)
            return (x1, y1, z1, dist, dphi, hel_s)
        else:
            return (x1, y1, z1, dist)

def helixUnitTangentVector(x0, y0, z0, uz0, hel_xm, hel_ym, hel_r, hel_pitch):
    """"Return the unit tangent vectors to the given helices at the given points.
    Args:
        x0, y0, z0 (array (n_points,)): coordinates of the points on the helices
        uz0 (array (n_points,)): estimated tangent vector z-component of the trajectories
            Note: Only the sign of uz0 is used in the calculation.
        hel_xm, hel_ym (array (n_points,)): center coordinates of helices in the x,y-plane
        hel_r (array (n_points,)): radii of the helices in the x,y-plane
        hel_pitch (array (n_points,)): pitches of the helices along the z-axis
    Returns:
        udir (array (n_points,3)): unit tangent vectors
    """
    # sign of phase difference for forward motion along the helix trajectory
    sign_dphi = np.sign(uz0 * hel_pitch)
    # calculate unit tangent vectors
    hel_p = hel_pitch / (2 * np.pi)
    signed_norm = sign_dphi * np.sqrt(np.square(hel_r) + np.square(hel_p))
    dx = x0 - hel_xm
    dy = y0 - hel_ym
    udir = np.stack((-dy, dx, hel_p), axis=1) / signed_norm[:,np.newaxis]
    return udir

