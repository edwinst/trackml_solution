"""Correction of helix predictions due to perturbations of the idealized geometry.

The `HelixCorrector` class uses numerical maps learned from training events
(see `supervised.py`) in order to correct intersection predictions (based on
perfect helix trajectories) to take systematic deviations of the trajectories
from perfect helices into account.

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

import numpy as np

from trackml_solution.geometry import circleFromThreePoints, helixPitchFromTwoPoints

def submask(super_mask, mask):
    if super_mask is not None:
        if mask is not None:
            sub_mask = super_mask.copy()
            sub_mask[super_mask] = mask
        else:
            sub_mask = super_mask
    else:
        sub_mask = mask
    return sub_mask

class HelixCorrector:
    """This class handles perturbative corrections of helix intersection points and
    the corresponding update of the estimated helix parameters.
    Attributes:
        ._algo...Algorithm object
        ._run....the current algorithm run
        ._mask...None or bool array(run.candidates.n,)): selects which helices the corrector
            is working on. This array is always aligned with run.candidates.
            If `None`, the corrector works on the full candidates list.
        ._hel_params...tuple (hel_dz, hel_xm, hel_ym, hel_r, hel_pitch) of helix parameter
            arrays with the following alignment:
            if ._hel_params_mask is None: .hel_params[i] is aligned with candidates[mask]
            otherwise: .hel_params[i][.hel_params_mask] is aligned with candidates[mask]
                (i.e. np.sum(.hel_params_mask) == np.sum(mask),
                      .hel_params_mask.shape == .hel_params[i].shape)
        ._hel_points...list of length 2: stores coordinates of the latest 2 points
            used to update the helix fits.
            Each entry is a tuple (x, y, z) of arrays of coordinates, each array being
            aligned with candidates[mask].
            Note: If the corrector was called to update helix parameters with a non-trivial
                  mask, the coordinates not selected by the mask may be invalid after the
                  update. (This applies only to ._hel_points, not to ._hel_params.)
                  If derived subset correctors made parameter updates, ._hel_points may
                  not really reflect the latest points used to calculate ._hel_params.
    """
    def __init__(self, algo, run, mask=None,
                 crossing=-1,
                 hel_params=None, hel_params_mask=None, hel_points=None):
        """Create a HelixCorrector working on the (potentially masked) candidates list
        of the given algorithm run.
        Args:
            algo (Algorithm): the algorithm object to be used for calculating helix
                parameters, etc.
            run (Algorithm.Run): the current algorithm run containing the candidates
            mask (None or bool array(run.candidates.n,)): if given, the helix corrector
                will work only on the candidates for which mask is True.
            crossing (int): up to which crossing to get the helix parameters from.
                Default is -1, meaning to get the helix parameters up to the
                latest layer crossing of each candidate.
            hel_params: Users should pass `None`. See 'Attributes' for details.
            hel_params_mask: Users should pass `None`. See 'Attributes' for details.
            hel_points: Users should pass `None`. See 'Attributes' for details.
        """
        self._algo = algo
        self._run = run
        self._mask = mask
        self._crossing = crossing
        if hel_params is None:
            assert hel_params_mask is None
            hel_params = self._algo.getHelixParams(run, crossing-2, crossing-1, crossing, mask=self._mask)
            hel_params = tuple(par.copy() for par in hel_params)
        if hel_points is None:
            hel_points = [self._run.candidates.hitCoordinates(i, mask=mask) for i in (crossing-1, crossing)]
        self._hel_params = hel_params
        self._hel_params_mask = hel_params_mask
        self._hel_points = hel_points

    def isAlignedWith(self, mask):
        """Check whether this corrector works on the same helices as those selected
        by the given mask (or on all if no mask is given).
        Args:
            mask (None or bool array(run.candidates.n,)): the mask to which to
                compare.
        """
        if mask is None:
            return self._mask is None or np.all(self._mask)
        else:
            return self._mask is not None and np.all(self._mask == mask)

    def clone(self):
        """Create a HelixCorrector working on the same helices as this one, but which
        can keep track of a separate set of helix parameters.
        Note: The clone will start working from the current helix parameters of this
              corrector, but updates done by the clone will not automatically
              propagate back to this corrector. You need to use the `merge` method
              to incorporate the parameter changes back into this corrector.
        Returns:
            corrector (HelixCorrector): the new corrector
        """
        hel_params = tuple(par.copy() for par in self._hel_params)
        the_clone = HelixCorrector(self._algo, self._run, self._mask,
                                   self._crossing,
                                   hel_params, self._hel_params_mask,
                                   self._hel_points.copy())
        return the_clone

    def merge(self, other, use_other):
        """Merge the helix parameters obtained by another HelixCorrector into this one.
        Precondition:
            Both this corrector and the other one must work on the same list of helices.
        Args:
            other (HelixCorrector): the other corrector
            use_other (bool array (N,)): take helix parameters from the other corrector
                for entries with use_other[i] == True, otherwise use this corrector's
                parameters.
        Note: The .hel_points attribute is not updated by this function.
        """
        assert self.isAlignedWith(other._mask)
        if np.any(use_other):
            parmask = submask(self._hel_params_mask, use_other)
            for i, par in enumerate(self._hel_params):
                assert par.shape == other._hel_params[i].shape
                par[parmask] = other._hel_params[i][parmask]

    def subset(self, mask):
        """Get a HelixCorrector working on the given subset of this corrector's helices.
        Note: Updates of helix parameters by the subset corrector will be automatically
              propagated back to this corrector.
        Args:
            mask (bool array(N,)): specifies which helices to include in the subset.
        where
            N...is the number of helices this corrector is working on.
        """
        subset_mask = submask(self._mask, mask)
        hel_params_mask = submask(self._hel_params_mask, mask)
        hel_points = [tuple(coord[mask] for coord in coords) for coords in self._hel_points]
        the_subset = HelixCorrector(self._algo, self._run, subset_mask,
            self._crossing,
            self._hel_params, hel_params_mask, hel_points)
        return the_subset

    def helixParams(self):
        """Get current helix parameters.
        Returns:
            hel_dz, hel_xm, hel_ym, hel_r, hel_pitch: helix parameters, aligned with
                the list of helices this corrector is working on.
        """
        if self._hel_params_mask is None:
            return self._hel_params
        else:
            return tuple(par[self._hel_params_mask] for par in self._hel_params)

    def correctCapIntersections(self, xi, yi, zi, cap_id):
        """Correct intersections points with caps (and re-calculate
        helix parameters if needed).
        Args:
            xi, yi, zi (np.float64 array (N,)): coordinates of the intersections
                before corrections
            cap_id (int array (N,)): identifies the cap containing the respective intersection
        where
            N...is the number of helices this corrector is working on.
        Returns:
            xi, yi (np.float64 array (N,)): corrected intersection coordinates,
                aligned like the inputs
        """
        if self._run.layer_functions is not None and 'dp0' in self._run.layer_functions.functions:
            # polar coordinates of the uncorrected intersection point in the x,y-plane
            r2 = np.sqrt(np.square(xi) + np.square(yi))
            phi = np.arctan2(yi, xi)

            # unit vectors in radial and azimuthal direction at the uncorrected
            # intersection point
            ur2_x = xi / r2
            ur2_y = yi / r2
            uphi_x = -ur2_y
            uphi_y =  ur2_x

            # get helix params and infer particle charge, get absolute pitch
            hel_dz, hel_xm, hel_ym, hel_r, hel_pitch = self.helixParams()
            q = -np.sign(hel_dz * hel_pitch)
            hel_p_abs = np.abs(hel_pitch / (2*np.pi))

            # interpolate intersection displacement functions
            last_z = self._hel_points[-1][2]
            points = np.stack([last_z, phi, r2], axis=1)
            qdur2_p, qduphi_p = self._run.neighbors.evaluateLayerFunctions(
                None, None, None, False, cap_id, functions=('dp0', 'dp1'), points=points)

            # set displacements to zero in cases where the layer functions
            # where not available
            qdur2_p [np.isnan(qdur2_p )] = 0
            qduphi_p[np.isnan(qduphi_p)] = 0

            # calculate intersection displacements in cartesian coordinates,
            # apply particle charge and helix curvature factors
            dx = (ur2_x * qdur2_p + uphi_x * qduphi_p) * q / hel_p_abs
            dy = (ur2_y * qdur2_p + uphi_y * qduphi_p) * q / hel_p_abs

            # set displacements to zero in regions where we have no learned data
            beyond_outer = (r2 > (self._algo.spec.caps.caps_r2_max[cap_id] + 10.0))
            beyond_inner = (r2 < (self._algo.spec.caps.caps_r2_min[cap_id] - 10.0))
            beyond = beyond_inner | beyond_outer
            dx[beyond] = 0
            dy[beyond] = 0

            # correct intersection points by adding the displacements
            xi += dx
            yi += dy

        return (xi, yi)

    def correctCylinderIntersections(self, xi, yi, zi, cyl_id, mask):
        """Correct intersections points with cylinders (and re-calculate
        helix parameters if needed).
        Args:
            xi, yi, zi (np.float64 array (N,)): coordinates of the intersections
                before corrections
            cyl_id (int array (N,)): identifies the cylinder containing 
                the respective intersection
            mask (bool array (N,)): specifies which entries of the input arrays
                xi, yi, zi, cyl_id are valid. Entries with mask[i] == False are
                ignored.
        where
            N...is the number of helices this corrector is working on.
        Returns:
            xi, yi, zi (np.float64 array (N,)): corrected intersection coordinates,
                aligned like the inputs.
                Note: For entries with mask[i] == False, the output is arbitrary
                      and should not be used.
        """
        if self._run.layer_functions is not None and 'dp0' in self._run.layer_functions.functions:
            # polar coordinates of the uncorrected intersection point in the x,y-plane
            r2 = np.sqrt(np.square(xi) + np.square(yi))
            phi = np.arctan2(yi, xi)

            # unit vectors in radial and azimuthal direction at the uncorrected
            # intersection point
            ur2_x = xi / r2
            ur2_y = yi / r2
            uphi_x = -ur2_y
            uphi_y =  ur2_x

            # get helix params and infer particle charge, get absolute pitch and curvature radius
            hel_dz, hel_xm, hel_ym, hel_r, hel_pitch = self.helixParams()
            q = -np.sign(hel_dz * hel_pitch)
            hel_p_abs = np.abs(hel_pitch / (2*np.pi))
            hel_cr = (np.square(hel_r) + np.square(hel_p_abs)) / hel_r

            # interpolate intersection displacement functions
            last_r2 = np.sqrt(np.square(self._hel_points[-1][0]) + np.square(self._hel_points[-1][1]))
            points = np.stack([last_r2, zi, phi], axis=1)
            qdz_p, qduphi_p = self._run.neighbors.evaluateLayerFunctions(
                None, None, None, True, cyl_id, functions=('dp0', 'dp1'), points=points)

            # set displacements to zero in cases where the layer functions
            # where not available
            qdz_p   [np.isnan(qdz_p   )] = 0
            qduphi_p[np.isnan(qduphi_p)] = 0

            # calculate intersection displacements in cartesian coordinates,
            # apply particle charge and helix curvature factors
            dx = (uphi_x * qduphi_p) * q / hel_cr
            dy = (uphi_y * qduphi_p) * q / hel_cr
            dz = qdz_p               * q / hel_cr

            # set displacements to zero in regions where we have no learned data
            beyond_outer = (zi > ( self._algo.spec.cylinders.cyl_absz_max[cyl_id] + 10.0))
            beyond_inner = (zi < (-self._algo.spec.cylinders.cyl_absz_max[cyl_id] - 10.0))
            beyond = beyond_inner | beyond_outer
            dx[beyond] = 0
            dy[beyond] = 0
            dz[beyond] = 0

            # correct intersection points by adding the displacements
            xi += dx
            yi += dy
            zi += dz

        return (xi, yi, zi)

    def updateHelices(self, xi, yi, zi, mask=None):
        """Calculate new helix fits assuming the given intersections points shall lie on
        the new helices.
        Args:
            xi, yi, zi (np.float64 array (N,)): coordinates of the new intersection
                points
            mask (None or bool array (N,)): if given, specifies which entries of
                the input arrays xi, yi, zi are valid. Entries with mask[i] == False
                are ignored (no parameter update will be done for these).
                Note: .hel_points is updated even for the helices not selected by the
                      mask. Therefore, if a mask is used, subsequent updates and
                      derived subset correctors should only use narrower masks (i.e.
                      masks which imply the mask given here) to avoid propagation
                      of invalid data.
        where
            N...is the number of helices this corrector is working on.
        """
        # get coordinates of the previous two points on each helix and add
        # the new points
        # Note: Within this function, self._hel_points will temporarily have 3 entries
        #       but we restore it to two before returning.
        self._hel_points.append((xi, yi, zi))
        p = self._hel_points
        # for the fit calculation, potentially mask out invalid data to avoid problems
        if mask is not None:
            p = [tuple(coord[mask] for coord in coords) for coords in p]
        # calculate new helix fit
        # XXX unify this with track fitting code in algorithm.py
        hel_xm, hel_ym, hel_r = circleFromThreePoints(*p[0][:2], *p[1][:2], *p[2][:2])
        hel_pitch, _, hel_dz = helixPitchFromTwoPoints(*p[1], *p[2], hel_xm, hel_ym)
        new_params = (hel_dz, hel_xm, hel_ym, hel_r, hel_pitch)
        # store new helix parameters
        update_mask = submask(self._hel_params_mask, mask)
        if update_mask is None:
            for dst, src in zip(self._hel_params, new_params):
                np.copyto(dst, src)
        else:
            for dst, src in zip(self._hel_params, new_params):
                np.place(dst, update_mask, src)
        # remember only the latest two points for the new fits
        self._hel_points = self._hel_points[1:]

