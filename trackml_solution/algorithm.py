"""The main solution algorithm.

See the method `findTracks` at the end of the `Algorithm` class
for the outermost loops of the solution algorithm.

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
import re
from collections import OrderedDict
from string import ascii_lowercase
import pprint

import numpy as np
import pandas as pd
from scipy import stats

from trackml.score import score_event

from trackml_solution.geometry import (Intersector, circleFromThreePoints, helixPitchFromTwoPoints,
                                       helixPitchLeastSquares, helixDirectionFromTwoPoints,
                                       helixWithTangentVector, helixNearestPointDistance,
                                       helixUnitTangentVector)
from trackml_solution.neighbors import Neighbors
from trackml_solution.candidates import Candidates
from trackml_solution.corrections import HelixCorrector
from trackml_solution.cells import CellFeatures
from trackml_solution.logging import Logger

class Algorithm(Logger):
    """This class holds the (hyper-)parameterization and learned data for running
    the solution algorithm and contains the methods which implement the algorithm.
    After constructing an instance, invoke the `findTracks` method to run the
    solution algorithm.
    Note: Individual methods of the Algorithm class are also called by the
    Supervisor (see supervised.py) during learning.
    """
    class Run:
        """Represents one run (i.e. the application to one event) of the algorithm.
        Note: If the algorithm does multiple iterations of the commit loop ("rounds"),
              a new Run object will be created for each round.
        Note: The Algorithm class does some direct attribute setting on the Run object.
        """
        def __init__(self, algo, supervisor, event, params,
                     layer_functions=None, used=None, hit_in_cyl=None, hit_layer_id=None,
                     fit_columns=None):
            self.algo = algo
            self.supervisor = supervisor
            self.event = event
            self.params = params
            self.neighbors = Neighbors(self.algo.spec, params=self.params, layer_functions=layer_functions)
            available_hits_df = self.event.hits_df
            if used is not None:
                available_hits_df = available_hits_df.loc[~used[available_hits_df['hit_id']]]
            self.available_hits_fraction = len(available_hits_df) / len(self.event.hits_df)
            self.neighbors.fit(available_hits_df, hit_in_cyl=hit_in_cyl, hit_layer_id=hit_layer_id)
            self.hit_in_cyl = hit_in_cyl
            self.hit_layer_id = hit_layer_id
            self.candidates = Candidates(self.event, fit_columns=fit_columns)
            self.layer_functions = layer_functions
        def hasLayerFunction(self, function):
            return self.layer_functions is not None and function in self.layer_functions.functions

    default_params = OrderedDict([
        # params for the track candidate evaluation function
        ('value__hit_bonus'     , 0.000524), # per-hit bonus in track evaluation
        ('value__cross_bonus'   , 0.000000), # per-layer-crossing bonus in track evaluation
        ('value__bayes_weight'  , 0.00005 ), # weight of "Bayesian" probability score in track evaluation
        ('value__p0'            , 0.01    ), # probability of a particle crossing a layer without leaving a hit
        ('value__fit_weight_hcs', 0.0     ), # weight of helix parameter evaluation score in track evaluation
        ('value__ploss_weight'  , 0.0     ), # weight of helix-pitch-fitting loss in track evaluation
        ('value__ploss_bias'    , 0.0     ), # expected helix-pitch-fitting loss for a good track
        ('value__cells_weight'  , 0.005   ), # weight of cell features in track evaluation
        ('value__cells_bias'    , 1.0     ), # expected inner product between helix direction and cell feature direction

        # params for defining the neighborhoods to consider for the second hit of candidate tracks
        ('nb__origin_dz'        , 20.0    ), # for second hit neighborhoods: displacement of origin for finding second hit neighborhoods
        ('nb__cap_origin_radius', 25.0    ), # radius of virtual origin region for defining second hit neighborhoods on caps
        ('nb__cyl_origin_area'  , 392.7   ), # area of virtual origin region for defining second hit neighborhoods on cylinders
        ('nb__nlayers'          , 1       ), # number of layers considered for second hit neighborhoods
        ('nb__radius_exp'       , 1.0     ), # exponent for scaling second hit neighborhoods

        # params for defining the neighborhoods searched for extension candidate hits
        ('nb__dist_threshold'   , 0.0250  ), # factor for calculating the helix distance cut-off (old extension heuristics)
        ('nb__dist_trust'       , 0.8     ), # fraction of the cut-off distance at which to consider extensions dubious (old and new)
        ('nb__cut_factor'       , 1.0     ), # factor for calculating the cut-off for the new extension heuristics
        ('nb__cells_cut'        , 0.09    ), # maximum difference of helix direction and cell feature direction

        # params for finding "paired" hits (i.e. multiple hits for one track in a layer crossing)
        ('pair__dist_threshold' , 0.0050  ), # factor for calculating the distance cut-off for pairing (old extension heuristics)
        ('pair__diff_threshold' , 0.0025  ), # factor for calculating the distance-difference cut-off for pairing (old extension heuristics)
        ('pair__cut'            , 1.0     ), # factor for calculating the pairing cut-off (new pairing heuristics)

        # params for truncating the ranked candidates list
        ('rank__ntop'           , 200000  ), # maximum number of candidates to keep in the candidates list
        ('rank__ntop_qu'        , 2e-5    ), # coefficient of (nhits**2) in determining the max. number of candidates
        ('rank__ntop_li'        , 1.0     ), # coefficient of (nhits) in determining the max.number of candidates

        # params for the commit loop
        ('commit__nmax'         , 1000    ), # maximum number of candidates to commit to submission per round (except in the last round)
        ('commit__niter'        , 1       ), # maximum number of commit rounds
        ('commit__min_nhits'    , 3       ), # minimum number of hits needed to commit a candidate track
        ('commit__max_nloss'    , 3       ), # maximum number   of hits a candidate may loose to higher prio candidates before being skipped
        ('commit__max_loss_fraction', 0.2 ), # maximum fraction of hits a candidate may loose to higher prio candidates before being skipped

        # params for the follow tracks loop
        ('follow__niter'        , 11      ), # number of steps of track extension to perform
        ('follow__nskip_max'    , 2       ), # maximum number of layers a candidate track may skip before being dropped
        ('follow__pairs_k'      , 4       ), # number of k-neighbors considered when pairing hits
        ('follow__drop_start'   , 1000    ), # number of extension steps after which to start dropping redundant track candidates
                                             # Note: 1000 used as effective np.inf.
        ('follow__weird_triples', False   ), # If True, choose triples for starting candidate tracks very loosely.
        ('follow__weird_k'      , 8       ), # Number of k-neighbors to consider for building the loosely selected triples.

        # params related to helix fitting
        ('fit__hel_r_min'       , 0.0     ), # minimum plausible helix radius in x,y-plane

        # miscellaneous params
        ('post__nonphys_odd'    , False   ), # If True, apply the non-physical score optimization proposed by Grzegorz Sionkowski
    ]   + list(Neighbors.default_params.items())
    )

    def __init__(self, spec, params={}, max_log_indent=None, layer_functions=None):
        """
        Create the Algorithm object.
        Args:
            spec (geometry.DetectorSpec): Defines the geometry of the detector.
            params (dict or OrderedDict): hyperparameters to override defaults.
            max_log_indent (None or int): maximum log indent level to show.
            layer_functions (None or LayerFunctions): data for per-layer interpolated
                functions
        """
        super(Algorithm, self).__init__(max_log_indent=max_log_indent)
        self.spec = spec
        self.intersector = Intersector(self.spec)
        self.params = self.default_params
        self.params.update(params)
        self.layer_functions = layer_functions

    def shortStats(self, data, fmt='%.4f'):
        """Return a string giving some very brief descriptive statistics of the given array."""
        return ("mean " + fmt + " std " + fmt + " [" + fmt + "; " + fmt + "]") % (
            np.mean(data), np.std(data), np.amin(data, axis=0), np.amax(data, axis=0))

    def chooseLikelyFirstHits(self, run):
        """Find hits which are likely to be the first hit of a track.
        Args:
            run (Run): the Run object holding the data structures for this round
        Returns:
            nh (pd.DataFrame): dataframe with a 'hit_id' column listing the candidate hits.
                Note: Best not to rely on any other columns being present.
                XXX This should really be changed into an array of hit_ids.
        """
        # XXX would be cleaner to return a np.array from this function
        nh = run.neighbors.findFirstHitNeighborhood()
        self.log("chosen candidates for first hits: ", len(nh))
        return nh

    def chooseLikelySecondHits(self, run, hit_id, origin_coords=(0.0, 0.0, 0.0)):
        """Given an array of potential first hits of tracks, choose a set of likely second
        hits for each first hit to get seeds for starting candidate tracks.
        Args:
            run (Run): the Run object holding the data structures for this round
            hit_id (int32 array or pd.Series): the hit_ids of the first candidate hits
            origin_coords (3-tuple of coordinates): coordinates to assume for the
                 most likely origin of particle tracks.
        Returns:
            nb_df (pd.DataFrame): Dataframe with two columns:
                'hit_id': hit id of the first hit of the candidate
                'nb_hit_id': hit id of the second hit of the candidate
                XXX change to a list of two arrays?
        """
        nb_dfs = []
        x0, y0, z0 = run.event.hitCoordinatesById(hit_id)
        # pick a very large helix pitch (if it is large enough, the sign (-charge * sign(hel_dz)) should not matter)
        hel_pitch = np.full(x0.shape, 1e6) # XXX refine
        considered = []
        zo = origin_coords[2]
        for origin_z in (zo, zo - run.params['nb__origin_dz'], zo + run.params['nb__origin_dz']): # XXX refine
            # assume (almost) straight lines through the origin
            ux0, uy0, uz0 = x0 - origin_coords[0], y0 - origin_coords[1], z0 - origin_z
            (hel_xm, hel_ym, hel_r) = helixWithTangentVector(x0, y0, z0, ux0, uy0, uz0, hel_pitch)
            last_x, last_y, last_z = x0, y0, z0
            last_hel_s = 0.0
            for i_layer in range(int(run.params['nb__nlayers'])): # XXX refine second hit neighborhood
                # find next intersections with detector elements
                xi, yi, zi, cyl_closer, next_id, dphi, hel_s, _ = self.intersector.findNextHelixIntersection(
                    last_x, last_y, last_z, uz0, hel_xm, hel_ym, hel_r, hel_pitch)
                hel_s += last_hel_s # accumulate arc length
                last_hel_s = hel_s
                last_x, last_y, last_z = xi, yi, zi
                # factor for estimating the neighborhood size
                r20_sqr = np.square(x0) + np.square(y0)
                r20 = np.sqrt(r20_sqr)
                r0 = np.sqrt(r20_sqr + np.square(z0))
                factor = (hel_s / r0) ** run.params['nb__radius_exp']
                # dz neighborhood size for cylinder neighborhoods
                cyl_dz = np.sqrt(run.params['nb__cyl_origin_area']
                                 * run.params['nb__cyl_scale'] / np.pi) * factor # XXX refine
                # dr2 neighborhood size for cap neighborhoods
                cos_theta = z0 / r0
                abs_sin_theta = np.sqrt(1 - np.square(cos_theta))
                cap_dr2 = run.params['nb__cap_origin_radius'] * abs_sin_theta * factor # XXX refine
                radius = cap_dr2
                radius[cyl_closer] = cyl_dz[cyl_closer]
                # select intersections points we did not consider, yet
                mask_new = np.full(len(xi), True)
                for prev_cyl_closer, prev_next_id in considered:
                    mask_new[(cyl_closer == prev_cyl_closer) & (next_id == prev_next_id)] = False
                self.log("new intersections for origin_z = ", origin_z, ": ", np.sum(mask_new))
                nb_df = run.neighbors.findIntersectionNeighborhood(
                    xi[mask_new], yi[mask_new], zi[mask_new],
                    cyl_closer[mask_new], next_id[mask_new], radius[mask_new])
                if nb_df is not None:
                    vind = nb_df['vind'].values
                    nb_df.drop(columns='vind', inplace=True)
                    nb_df['hit_id'] = hit_id.values[mask_new][vind]
                    nb_df['xi'] = xi[mask_new][vind]
                    nb_df['yi'] = yi[mask_new][vind]
                    nb_df['zi'] = zi[mask_new][vind]
                    nb_dfs.append(nb_df)
                considered.append((cyl_closer, next_id))
        return pd.concat(nb_dfs, ignore_index=True) if nb_dfs else None

    def projectNeighborDisplacement(self, x, y, z, xn, yn, zn):
        """Project the displacement between a hit position and the nearest helix point
        onto two orthogonal directions utheta, uphi.
        Directions:
            utheta is represented by a unit vector in the direction of increasing polar
                angle theta (spherical coordinates).
            uphi is represented by a unit vector in the direction of increasing azimuthal
                angle phi (cylindrical coordinates).
        Args:
            x, y, z (float64 array(N,)): coordinates of the actual hit
            xn, yn, zn (float64 array(N,)): coordinates of the nearest helix point
        Returns:
            d_utheta, d_uphi (float64 array(N,)): (signed) displacements in the
                directions utheta, iphi, respectively.
        """
        # set up unit vectors normal to the ray from the origin
        # XXX merge this code with the one in supervised.py?
        r2 = np.sqrt(np.square(x) + np.square(y))
        r = np.sqrt(np.square(x) + np.square(y) + np.square(z))
        utheta_factor = np.abs(z) / (r2 * r)
        utheta_x = utheta_factor * x
        utheta_y = utheta_factor * y
        utheta_z = -np.sign(z) * r2 / r
        uphi_x = -y / r2
        uphi_y =  x / r2
        uphi_z = 0
        # calculate coordinate differences
        xd = x - xn
        yd = y - yn
        zd = z - zn
        # project difference vectors onto the normal unit vectors
        d_utheta   = utheta_x * xd   + utheta_y * yd   + utheta_z * zd
        d_uphi     = uphi_x   * xd   + uphi_y   * yd   + uphi_z   * zd
        return (d_utheta, d_uphi)

    def predictRandomError(self, run, hel_s, e_theta_meas, e_phi_meas):
        """Predict the random errors of helix intersection prediction.
        """
        e_predict = 0.007884 * hel_s # XXX refine error estimate
        e_theta = np.sqrt(np.square(e_theta_meas) + np.square(e_predict * 0.2096))
        e_phi   = np.sqrt(np.square(e_phi_meas  ) + np.square(e_predict * 0.3853))
        return e_theta, e_phi

    def bayesianNeighborEvaluation(self, run, nb_df, mask=None, step=0, details=None):
        """Evaluate the neighbor hits found as potential extensions of the candidate hits.
        Args:
            run (Algorithm.Run): data and parameters for this algorithm run
            nb_df (pd.DataFrame): dataframes listing the neighbors found
                used columns are:
                    extend_hit_id: the hit_id of the neighbor suggested
                    cyl_closer, next_id: layer identification of neighbor
                    xn, yn, zn: point on predicted helix nearest to neighbor
                    hel_s: helix arc length from previous layer crossing
                        (currently used for error estimation)
            mask (None or bool array(run.candidates.n,)): if given, True for the
                candidates which were active in the neighbor search. (The
                'extend_index' in nb_df is aligned with the masked candidates.)
            step (integer): identifies the algorithm step for logging, analysis, etc.
            details (None or SimpleNamespace, etc.): if given, some detailed info
                for the supervisor will be stored in this object.
        Returns:
            good_neighbor (bool array(len(nb_df),)): True if the neighbor should
                be considered as an extension of the candidate track
            dubious (bool array(len(nb_df),)): True if the neighbor is dubious
                as the right extension of the candidate track.
        """
        # coordinates of the neighbor hits found
        x, y, z = run.event.hitCoordinatesById(nb_df['extend_hit_id'])

        # coordinates of the predicated helix points closest to the neighbors found
        xn = nb_df['xn'].values
        yn = nb_df['yn'].values
        zn = nb_df['zn'].values

        # project neighbor displacements to spherical coordinate directions
        d_theta, d_phi = self.projectNeighborDisplacement(x, y, z, xn, yn, zn)

        # evalute neighbor functions at the locations of the neighbor hits
        ld_utheta_abs, ld_uphi_abs, ldnt_utheta_abs, ldnt_uphi_abs = run.neighbors.evaluateLayerFunctions(
            x, y, z, nb_df['cyl_closer'].values, nb_df['next_id'].values,
            functions=('d_utheta_abs', 'd_uphi_abs', 'dnt_utheta_abs', 'dnt_uphi_abs'))

        # calculate formal estimated errors
        e_theta, e_phi = self.predictRandomError(run, hel_s=nb_df['hel_s'],
            e_theta_meas=ldnt_utheta_abs, e_phi_meas=ldnt_uphi_abs)

        if details:
            details.bay_index  = nb_df['extend_index'].values
            details.bay_hit_id = nb_df['extend_hit_id'].values
            details.bay_b_theta   = ld_utheta_abs
            details.bay_b_phi     = ld_uphi_abs
            details.bay_d_theta   = d_theta
            details.bay_d_phi     = d_phi
            details.bay_e_theta   = e_theta
            details.bay_e_phi     = e_phi

        # convert neighbor displacement and background hit distances to
        # multiples of the respective error estimates
        de_theta  = d_theta       / e_theta
        de_phi    = d_phi         / e_phi
        dbe_theta = ld_utheta_abs / e_theta
        dbe_phi   = ld_uphi_abs   / e_phi

        # quadrature sum of error-scaled neighbor displacement
        # XXX could save some np.sqrt evaluations here for speed by comparing the squares
        de = np.sqrt(np.square(de_theta) + np.square(de_phi))

        # calculate cut-off from Bayesian analysis
        # Note: The cut values are added in quadrature for the two orthogonal directions.
        #       This addition has been pulled into the argument of the logarithm
        #       as a multiplication.
        cut = run.params['nb__cut_factor'] * np.sqrt(
            2 * np.log( (2*np.square(dbe_theta)/np.pi + 1)
                      * (2*np.square(dbe_phi  )/np.pi + 1)
                      ))
        good_neighbor = (de < cut)
        dubious       = (de > run.params['nb__dist_trust'] * cut)

        if details:
            details.bay_cut = cut
            details.bay_de  = de

        return good_neighbor, dubious

    def getHelixParams(self, run, i_fit0, i_fit1, i_fit2, mask=None,
                       pitch_from=(1,2), origin_coords=(0.0, 0.0, 0.0)):
        """Get helix parameters for each candidate track.
        XXX document arguments
        """
        x0, y0, z0 = run.candidates.hitCoordinates(i_fit0, mask=mask)
        x1, y1, z1 = run.candidates.hitCoordinates(i_fit1, mask=mask)
        x2, y2, z2 = run.candidates.hitCoordinates(i_fit2, mask=mask)
        # we assume the origin as the first point if we have only two points so far
        for coord, origin_coord in zip((x0, y0, z0), origin_coords):
            coord[np.isnan(coord)] = origin_coord

        # look up x,y-plane part of the helix fit for each candidate to extend
        hel_xm    = run.candidates.getFit(i_fit2, 'hel_xm'   , mask=mask).astype(np.float64)
        hel_ym    = run.candidates.getFit(i_fit2, 'hel_ym'   , mask=mask).astype(np.float64)
        hel_r     = run.candidates.getFit(i_fit2, 'hel_r'    , mask=mask).astype(np.float64)
        if pitch_from != (1,2):
            # fit helices XXX merge with self.fitTracks?
            # XXX pitch from three points if available?
            xs = (x0, x1, x2)
            ys = (y0, y1, y2)
            zs = (z0, z1, z2)
            pitch_coords = (coords[index] for index in pitch_from for coords in (xs, ys, zs))
            hel_pitch, _, hel_dz = helixPitchFromTwoPoints(*pitch_coords, hel_xm, hel_ym)
            assert hel_pitch.shape == hel_dz.shape
        else:
            # look up helix pitch fit for each candidate to extend
            hel_pitch = run.candidates.getFit(i_fit2, 'hel_pitch', mask=mask).astype(np.float64)
            hel_dz    = helixDirectionFromTwoPoints(hel_xm, hel_ym, hel_pitch, x1, y1, z1, x2, y2, z2)
        return (hel_dz, hel_xm, hel_ym, hel_r, hel_pitch)

    def dropNeighborsInconsistentWithCellFeatures(self, run, nb_df):
        """Discard neighbor hit candidates the cell features of which are inconsistent
        with the predicted helix parameters.
        Precondition:
            run.cell_features is not None: cell features are available
        Args:
            run (Run): the Run object holding the data structures for this round
            nb_df (pd.DataFrame): dataframes listing the neighbors
        Returns:
            nb_df (pd.DataFrame): filtered dataframe listing the neighbors
                which passed the consistency check.
        """
        assert run.cell_features is not None
        # get helix parameters, hit coordinates, and hit layer for the candidates
        hel_params = tuple(nb_df[col] for col in ('hel_dz', 'hel_xm', 'hel_ym', 'hel_r', 'hel_pitch'))
        hit_id = nb_df['extend_hit_id'].values
        x, y, z = run.event.hitCoordinatesById(hit_id)
        in_cyl = run.hit_in_cyl[hit_id]
        layer_id = run.hit_layer_id[hit_id]
        # get the unit tangent vector to the helix at the hit
        udir = helixUnitTangentVector(x, y, z, *hel_params)
        udir_cells, inner, d = run.cell_features.estimateClosestDirection(hit_id, udir)
        # only apply cut for hits in the innermost four cylinders # XXX for now
        is_inner_cyl = in_cyl & (layer_id < 4)
        # only keep neighbors which are consistent enough
        keep = ~is_inner_cyl | (d <= run.params['nb__cells_cut'])
        nb_df = nb_df.loc[keep]
        return nb_df

    def findHelixIntersectionNeighbors(self, run, i_fit0, i_fit1, i_fit2, last_x, last_y, last_z,
                                       last_dphi=0.0, k=2, nmax_per_crossing=2, pitch_from=(1,2),
                                       revisit=False, mask=None,
                                       force_cyl_closer=None,
                                       origin_coords=(0.0, 0.0, 0.0),
                                       corrector=None,
                                       step=0, details=None):
        """Predict helix intersections for candidate tracks and find the hits closest
        to the predicted intersections.
        Args:
            run (Algorithm.Run): data and parameters for this algorithm run
            i_fit0, i_fit1, i_fit2 (int): crossing indices along the candidate track to use
                for fitting the helix parameters. XXX remove since we always use the last three?
            last_x, last_y, last_z (float64 array (N,)): coordinates of the last known
                point on each track (do not have to be coordinates of a hit).
            last_dphi (float64 scalar or array (N,)): helix phase difference from
                the last known hit to the point (last_x, last_y, last_z)
            k (int): number of neighbors to look up per intersection
            nmax_per_crossing (int): maximum number of neighbors to pair together for one
                extension
            pitch_from (tuple of two int): indicates which of the i_fit* crossings to
                use for helix pitch fitting (and in which order).
            revisit (bool): Whether we are revisiting an already found intersection.
                If True, move backwards along each helix
                before predicting the next intersection (the intention is to revisit
                the latest layer crossing). If False, do the default move forwards
                before predicition the next intersection in order to prevent getting
                stuck at an intersection.
            mask (None or bool array (run.candidates.n,)): If given, consider only the
                track candidates for which mask is True.
            force_cyl_closer (None or bool array (N,)): If given, force predicted
                intersection to be on a cylinder (for True) or on a cap (for False).
            origin_coords (3-tuple of coordinates): coordinates to assume as the first
                 point in helix fitting if only two points are known.
            corrector (None or HelixCorrector): If given, use this corrector to
                 predict perturbations of the helices.
            step (integer): identifies the algorithm step for logging, analysis, etc.
                 XXX also use for follow__weird_triples, not so clean.
            details (None or SimpleNamespace, etc.): if given, some detailed info
                for the supervisor will be stored in this object.
        Returns:
            has_intersection (bool array (N,)): True if an intersection with the
                detector geometry has been predicted for the respective candidate track.
            xi, yi, zi, dphi, hel_s (float64 array (N,)): predicted intersection
                coordinates, phase differences, and arc length (the latter two
                relative to (last_x, last_y, last_z).
            nb_df (pd.DataFrame): dataframe with found extension candidates.
                nb_df['extend_index'] index of candidate track (in the full candidates
                    list if mask is None, otherwise into the masked candidates list).
        where:
            N...is run.candidates.n if mask=None, otherwise N == np.sum(mask)
        """
        # XXX this function is much too complicated. Refactor into subfunctions.
        # assert some sanity checks
        assert np.isscalar(last_dphi) or last_dphi.shape == last_x.shape

        # get helix parameters: from the corrector, if given, or from the candidate fits
        if corrector is not None:
            assert corrector.isAlignedWith(mask)
            hel_dz, hel_xm, hel_ym, hel_r, hel_pitch = corrector.helixParams()
        else:
            hel_dz, hel_xm, hel_ym, hel_r, hel_pitch = self.getHelixParams(
                run, i_fit0, i_fit1, i_fit2,
                pitch_from=pitch_from, mask=mask, origin_coords=origin_coords)

        # sanity check shapes
        assert (hel_dz.shape == hel_xm.shape == hel_ym.shape == hel_r.shape == hel_pitch.shape
                == last_x.shape == last_y.shape == last_z.shape)

        # intersect helices with detector geometry
        pre_move = 'back' if revisit else None
        (xi, yi, zi, cyl_closer, next_id, dphi, hel_s, mult) = self.intersector.findNextHelixIntersection(
            last_x, last_y, last_z, hel_dz, hel_xm, hel_ym, hel_r, hel_pitch,
            cyl_pre_move=pre_move, cap_pre_move=pre_move, missable=(not revisit),
            force_cyl_closer=force_cyl_closer, corrector=corrector)
        assert (xi.shape == yi.shape == zi.shape == cyl_closer.shape == next_id.shape
                == dphi.shape == hel_s.shape == mult.shape == last_x.shape)
        if details:
            details.cyl_closer = cyl_closer
            details.next_id    = next_id
            details.mult       = mult

        # get updated helix parameter predictions from the corrector
        if corrector is not None:
            corrector.updateHelices(xi, yi, zi, mask=np.isfinite(xi))
            hel_dz, hel_xm, hel_ym, hel_r, hel_pitch = corrector.helixParams()

        # reject intersections which are half a helix turn or more distant,
        # as we have no chance to fit tracks to them
        # XXX do this in a cleaner way
        total_dphi = np.where(np.isnan(dphi), np.inf, last_dphi + dphi)
        reject = (np.abs(total_dphi) >= np.pi)
        if details:
            details.nno_intersection = np.sum(np.isnan(xi))
            details.nrejected_dphi = np.sum(reject) - details.nno_intersection
            details.reject_total_dphi = reject & ~np.isnan(xi)
        xi   [reject] = np.nan
        yi   [reject] = np.nan
        zi   [reject] = np.nan
        hel_s[reject] = np.inf
        dphi [reject] = np.nan
        mult [reject] = 0

        has_intersection = np.isfinite(hel_s)
        n_intersection = np.sum(has_intersection, axis=0)

        # {{{ all arrays we have so far are aligned with the (possibly masked) candidate list }}}

        # find k-neighbors of the predicted intersection points
        nb_df = run.neighbors.findIntersectionNeighborhoodK(xi, yi, zi, cyl_closer, next_id, k=k)
        if nb_df is None:
            self.log("no intersection neighbors found (nb_df is None)")
            return has_intersection, xi, yi, zi, dphi, hel_s, nb_df

        if details:
            self.log("number of seeds, intersections, and neighbors: "
                     + "%d -> %d (%d, %.2f%%, %d !i/s, %d dphi) -> %d (%.1f each)"
                     % (len(xi), n_intersection,  n_intersection - len(xi),
                     (n_intersection - len(xi))/ len(xi) * 100.0,
                     details.nno_intersection, details.nrejected_dphi,
                     len(nb_df), len(nb_df) / n_intersection))

        # {{{ nb_df has exaktly k rows for each valid intersection, ordered by
        #     (intersection index, neighbor distance) }}}
        # Note: Due to the idealization of the detector geometry in the neighborhood search,
        #       the neighbor distance is only a very approximate distance measure. We do not
        #       use it for selection of candidates.
        assert len(nb_df) == k * n_intersection

        # build a dataframe for re-indexing our data to align it with the neighbor data below
        # fit_df itself is aligned with the (possibly masked!) candidate list
        # The "vind" column in the neighborhood data points into fit_df's index.
        fit_df = pd.DataFrame(data=OrderedDict([
            ('last_x' , last_x),
            ('last_y' , last_y),
            ('last_z' , last_z),
            ('hel_xm' , hel_xm),
            ('hel_ym' , hel_ym),
            ('hel_r' , hel_r),
            ('hel_pitch' , hel_pitch),
            ('hel_dz' , hel_dz),
            ('xi' , xi),
            ('yi' , yi),
            ('zi' , zi),
            ('ri' , np.sqrt(np.square(xi) + np.square(yi) + np.square(zi))),
            ('ri2' , np.sqrt(np.square(xi) + np.square(yi))),
            ('cyl_closer' , cyl_closer),
            ('next_id' , next_id),
            ('dphi' , dphi),
            ('hel_s' , hel_s),
            ('mult', mult),
            ('prev_hit_id' , run.candidates.hitIds(i_fit2, mask=mask)),
        ]))
        fit_df.name = 'fit_df'

        # reindex helix parameters and intersection results to align them with nb_df
        # Note: "extend_index" will point into the (possibly masked!) candidate list,
        #       and thereby also into our intersection data from above.
        nb_df = (nb_df.join(fit_df, on='vind', how='left', sort=False)
                      .rename(columns={'vind': 'extend_index', 'nb_hit_id': 'extend_hit_id'}))
        nb_df.name = 'nb_df'
        del fit_df

        # calculate helix distance of all the neighbors found
        x, y, z = run.event.hitCoordinatesById(nb_df['extend_hit_id'])
        hel_params = [nb_df[col].values for col in 
            ('last_x', 'last_y', 'last_z', 'hel_xm', 'hel_ym', 'hel_r', 'hel_pitch')]
        (xn, yn, zn, dist) = helixNearestPointDistance(*hel_params, x, y, z)
        nb_df['dist'] = dist
        nb_df['xn'] = xn
        nb_df['yn'] = yn
        nb_df['zn'] = zn

        new_pairs = revisit and run.hasLayerFunction('pair_theta')
        if new_pairs:
            # calculate a distance measure taking the local background hit density
            # into account, separately for azimuthal and polar directions
            d_theta, d_phi = self.projectNeighborDisplacement(x, y, z, xn, yn, zn)
            db_theta, db_phi, e_theta, e_phi = run.neighbors.evaluateLayerFunctions(
                x, y, z, nb_df['cyl_closer'].values, nb_df['next_id'].values,
                functions=('d_utheta_abs', 'd_uphi_abs', 'pair_theta', 'pair_phi'))
            de_theta  = np.abs(d_theta) / e_theta
            de_phi    = np.abs(d_phi  ) / e_phi
            dbe_theta = db_theta / e_theta
            dbe_phi   = db_phi   / e_phi
            weight_theta = np.square(dbe_theta)
            weight_phi   = np.square(dbe_phi  )
            de = (weight_theta * de_theta + weight_phi * de_phi) / (weight_theta + weight_phi)
            dist = de
            nb_df['dist'] = dist

        # sort neighbors within each neighbor group by distance to the fitted helix
        dist_reshaped = np.reshape(dist, (dist.shape[0] // k, k))
        dist_argsort = np.argsort(dist_reshaped, axis=1)
        dist_argsort = np.reshape(dist_argsort, dist.shape[0])
        dist_argsort = dist_argsort + np.repeat(np.arange(0, dist.shape[0], k), k)
        nb_df = nb_df.iloc[dist_argsort]
        nb_df.reset_index(drop=True, inplace=True)

        if not revisit:
            # detect intersections which are too close to the previous intersection point
            # Note: This means we can loose some gracing cylinder hits, but they cause problems, anyway.
            stuck = np.full(xi.shape, False)
            stuck[nb_df.loc[(nb_df['extend_hit_id'] == nb_df['prev_hit_id']), 'extend_index']] = True
            # remove all neighbors found for stuck intersections
            nstuck = np.sum(stuck)
            if nstuck > 0:
                self.log("stuck intersections: ", nstuck)
                nb_df = nb_df.loc[~stuck[nb_df['extend_index']]]

        # {{{ nb_df has exaktly k rows for each valid intersection,
        #     ordered by (intersection index, helix distance) }}}

        # we do not want to hit the same spot twice in a row (when moving forward along the track)
        assert revisit or not np.any(nb_df['extend_hit_id'] == nb_df['prev_hit_id'])

        # drop neighbor hits whose cell features are not consistent with the
        # helix parameters of the tracks we want to extend
        if run.cell_features is not None:
            nb_df = self.dropNeighborsInconsistentWithCellFeatures(run, nb_df)

        # {{{ nb_df has up to k rows for each valid intersection,
        #     ordered by (intersection index, helix distance) }}}

        # XXX refactor the pairing into a separate method
        # XXX we should not waist time on pairing here during the extension finding step
        #     when we use the new pairing algorithm which runs in a separate step.
        self.log("neighbors before pairing: ", len(nb_df))
        npaired = []
        forbidden_module_ids = [run.event.hitModuleIdById(nb_df['extend_hit_id'].values)]
        for i_neighbor in range(1, nmax_per_crossing):
            # try to detect paired hits
            good_pair = (
                  (nb_df['extend_index'] == np.roll(nb_df['extend_index'], -1)) # next entry same intersection
                & (nb_df['extend_index'] != np.roll(nb_df['extend_index'], 1))) # first entry for intersection

            if new_pairs:
                good_pair &= (np.roll(nb_df['dist'].values, -1) < run.params['pair__cut'])
            else:
                # find distance differences between successive nearest neighbors
                # (will be invalid in the last neighbor of each group!)
                nb_df['dist_diff'] = np.roll(nb_df['dist'].values, -1) - nb_df['dist'].values
                good_pair &= (
                  (nb_df['dist'] < run.params['pair__dist_threshold'] * nb_df['ri2'])       # XXX refine
                & (nb_df['dist_diff'] < run.params['pair__diff_threshold'] * nb_df['ri2'])) # XXX refine

            # disallow pairing hits with the same module_id
            # first check which neighbors have a forbidden module_id
            module_id = np.roll(forbidden_module_ids[0], -1)
            same_module_id = np.zeros(len(module_id), dtype=np.bool)
            for forbidden_module_id in forbidden_module_ids:
                same_module_id |= (module_id == forbidden_module_id)

            # now drop those neighbors which would have been paired but
            # have an already used module_id
            # XXX these neighbors would actually make nice candidates for
            #     alternative extensions of the track (or they are noise).
            # Note: This step is very annoying, but it's necessary in our current
            #       formalism if we do not want to be blocked from attempting
            #       further pairings as soon as we hit a repeated module_id.
            same_module_id &= good_pair
            keep_nb = ~np.roll(same_module_id, 1)
            nb_df = nb_df.loc[keep_nb]
            good_pair = good_pair[keep_nb] & ~same_module_id[keep_nb]
            module_id = module_id[keep_nb]
            for i in range(len(forbidden_module_ids)):
                forbidden_module_ids[i] = forbidden_module_ids[i][keep_nb]

            # copy hits from paired neighbor entry
            # XXX The data handling with the strangely named columns here is much
            #     too complicated. I do not have time now to clean it up, though.
            #     Probably we should be putting hit_ids into a multi-dimensional
            #     array.
            to_copy = good_pair.copy()
            for i_pair in range(1, i_neighbor + 1):
                hit_col = 'extend_hit_id_' + ascii_lowercase[i_pair]
                if i_pair == i_neighbor:
                    nb_df[hit_col] = 0 # create the new column
                is_free_slot = (nb_df[hit_col] == 0)
                use_this_slot = to_copy & is_free_slot & (i_pair < nb_df['mult'])
                nb_df.loc[use_this_slot, hit_col] = np.roll(nb_df['extend_hit_id'].values, -1)[use_this_slot]
                to_copy &= ~use_this_slot

            # remember module_id of paired hits; for non-paired hits store a redundant module_id
            module_id[~good_pair] = forbidden_module_ids[0][~good_pair] # redundant, no new module_id used
            forbidden_module_ids.append(module_id)

            # remove paired hits from the list of neighbors (and module_id arrays)
            nbefore = len(nb_df)
            not_paired_away = ~np.roll(good_pair, 1)
            nb_df = nb_df.loc[not_paired_away]
            for i in range(len(forbidden_module_ids)):
                forbidden_module_ids[i] = forbidden_module_ids[i][not_paired_away]
            npaired.append(nbefore - len(nb_df))

        self.log("paired: " + ", ".join(map(str, npaired)))

        if step==0 and run.params['follow__weird_triples']:
            nb_df['dubious'] = True
        else:
            # apply cuts to select only neighbors which are good candidates for extension
            # (unless we are revisiting an intersection for pairing)
            if not revisit:
                if run.hasLayerFunction('d_utheta_abs'):
                    # use the Bayesian evaluation to choose good neighbors for extension
                    good_neighbor, dubious = self.bayesianNeighborEvaluation(
                        run, nb_df, mask=mask, step=step, details=details)
                    nb_df['dubious'] = dubious
                    nb_df = nb_df.loc[good_neighbor]
                else:
                    # reject neighbors which are too far from the helix
                    # Note: Moved before picking the first neighbor in preparation for use of oracles.
                    dist_threshold = nb_df['hel_s'] * run.params['nb__dist_threshold'] # XXX refine
                    good_neighbor = (nb_df['dist'] < dist_threshold) # XXX refine
                    nb_df = nb_df.loc[good_neighbor]
                    dist_threshold = dist_threshold[good_neighbor]
                    nb_df['dubious'] = nb_df['dist'] > (run.params['nb__dist_trust'] * dist_threshold) # XXX refine

            # pick the first neighbor (and possibly its pair) of each neighbor group # XXX refine
            # Note: We use some custom code instead of .groupby(...).first(), as the latter is slow.
            if (not nb_df.empty) and nb_df['extend_index'].iat[0] == nb_df['extend_index'].iat[-1]:
                # special case: only one index
                nb_df = nb_df.head(1)
            else:
                is_first = (nb_df['extend_index'] != np.roll(nb_df['extend_index'], 1))
                nb_df = nb_df.loc[is_first]

        return has_intersection, xi, yi, zi, dphi, hel_s, nb_df

    def chooseLikelyNextHits(self, run, k=2, nmax_per_crossing=2, ncross_min_keep=2, nhits_min_keep=2, step=0,
                             origin_coords=(0.0, 0.0, 0.0)):
        df = run.candidates.df
        last_x, last_y, last_z, last_dphi, last_hel_s, last_nskipped = [
            df[col].values for col in ('xf', 'yf', 'zf', 'dphi', 'hel_s', 'nskipped')]

        corrector = HelixCorrector(self, run)

        has_intersection, xi, yi, zi, dphi, hel_s, nb_df = self.findHelixIntersectionNeighbors(
            run, -3, -2, -1, last_x, last_y, last_z,
            last_dphi=last_dphi,
            k=k, nmax_per_crossing=nmax_per_crossing,
            origin_coords=origin_coords,
            corrector=corrector,
            step=step)

        # {{{ all arrays are aligned with the full candidate list }}}
        # {{{ 'extend_index' points into the full candidate list (and thereby into the arrays) }}}

        will_be_extended = np.zeros(run.candidates.n, dtype=np.bool)
        dubious_extension = np.zeros(run.candidates.n, dtype=np.bool)

        if nb_df is not None:
            nneighbors = len(nb_df)
            self.log("neighbors after selection: ", nneighbors)
            # do not extend those which skipped too many intersections in-between
            # XXX this kind of cut could also (maybe better) be done based on
            #     accumulated arc length and error, density estimates
            self.log("skipping stats: ", self.shortStats(last_nskipped))
            skipped_too_many = (last_nskipped > run.params['follow__nskip_max']) # XXX refine
            nb_df = nb_df.loc[~skipped_too_many[nb_df['extend_index'].values]]
            self.log("dropped because they skipped too many intersections: ", nneighbors - len(nb_df))

            # find which previous candidates will be extended
            extend_index = nb_df['extend_index'].values
            will_be_extended[extend_index] = True
            dubious_extension[extend_index] = nb_df['dubious']
            # find hit_ids by which to extend
            extend_hit_cols = ['extend_hit_id'] + [
                'extend_hit_id_' + ch for ch in ascii_lowercase[1:nmax_per_crossing]]
            extend_hit_ids = [nb_df[col].values for col in extend_hit_cols]
        else:
            self.log("no neighbors")
            extend_index = []
            extend_hit_ids = []

        # update candidates data with intersection info
        # for keepers which had an intersection, we proceed next time from this time's intersection point
        # accumulate phase difference and arc length
        # increment skip counters for candidates which intersected but were kept
        df.loc[has_intersection, 'xf'] = xi[has_intersection]
        df.loc[has_intersection, 'yf'] = yi[has_intersection]
        df.loc[has_intersection, 'zf'] = zi[has_intersection]
        df.loc[has_intersection, 'dphi'] += dphi[has_intersection]
        df.loc[has_intersection, 'hel_s'] += hel_s[has_intersection]
        assert np.amax(last_nskipped, axis=0) < np.iinfo(last_nskipped.dtype).max
        df.loc[has_intersection, 'nskipped'] += 1

        # update candidates list: extend tracks by the good neighbors we found
        close_mask = ~has_intersection & (run.candidates.ncross >= 4)
        keep_mask = ( (~will_be_extended | dubious_extension)
                    & (run.candidates.ncross >= ncross_min_keep)
                    & (run.candidates.nHits() >= nhits_min_keep) )
        n_kept = np.sum(keep_mask, axis=0)
        self.log("keeping ", n_kept, ", of which ", np.sum(keep_mask & has_intersection, axis=0),
                 " had an intersection")

        run.candidates.update(close_mask=close_mask, keep_mask=keep_mask,
                              extend_index=extend_index, extend_hit_ids=extend_hit_ids)

        if nb_df is not None:
            # {{{ new index of candidates list: first the n_kept old ones,
            #     then len(extend_index) == len(nb_df) new ones }}}
            assert run.candidates.n == n_kept + len(nb_df)

            # reset and store data for extended candidates
            xf, yf, zf = run.candidates.hitCoordinates(-1) # XXX use mean of pairs?
            df = run.candidates.df
            df.loc[n_kept:,'xf'] = xf[n_kept:]
            df.loc[n_kept:,'yf'] = yf[n_kept:]
            df.loc[n_kept:,'zf'] = zf[n_kept:]
            df.loc[n_kept:,'nskipped'] = 0
            df.loc[n_kept:,'dphi'] = 0.0
            df.loc[n_kept:,'hel_s'] = 0.0
            df.loc[n_kept:,'dist'] = nb_df['dist'].values

    def findPairs(self, run, k=4, for_ncross=3, step=0,
                  origin_coords=(0.0, 0.0, 0.0)):
        """Find paired hits for layer crossings.
        Note:
            With "paired hits" we mean the set of (1 to 4) hits created by a track crossing
            and interacting with one layer of the detector.
        Args:
            run (Run): the Run object holding the data structures for this round
            k (int): number of neighbors to find for each intersection and pick the
                paired hits from
            for_ncross (int == 2, 3, or -1):
                2...find paired hits for the first crossing in tracks with exactly
                        two crossings
                3...find paired hits for the first two crossings in tracks with exactly
                        three crossings
                -1...find paired hits for the most recent crossing in all tracks with
                        at least three crossings
            origin_coords (3-tuple of coordinates): coordinates to assume as the first
                 point in helix fitting if only two points are known.
            step (int): algorithm step index for logging and off-line analysis
        """
        if for_ncross == -1:
            mask_seeds = (run.candidates.ncross >= 3) & (run.candidates.df['donePairs'].values < run.candidates.ncross)
        else:
            mask_seeds = (run.candidates.ncross == for_ncross) & (run.candidates.df['donePairs'].values == 0)
        nseeds = np.sum(mask_seeds)
        self.log("step %d: number of seeds to find pairs for: " % step, nseeds)
        if nseeds == 0:
            return

        nmax_pairs = min(k, run.candidates.nmax_per_crossing)
        extend_hit_cols = ['extend_hit_id'] + ['extend_hit_id_' + ch for ch in ascii_lowercase[1:nmax_pairs]]

        for i_step in range(1 if for_ncross == -1 else 2):
            if for_ncross == -1:
                crossing = -1
                pitch_from = (1, 2)
            else:
                crossing = -(for_ncross - 1) - i_step
                pitch_from = (4-for_ncross, 3-for_ncross)
            last_coords = run.candidates.hitCoordinates(crossing, mask=mask_seeds)
            has_intersection, xi, yi, zi, _, hel_s, nb_df = self.findHelixIntersectionNeighbors(run, -3, -2, -1, *last_coords,
                k=k, nmax_per_crossing=nmax_pairs, pitch_from=pitch_from, revisit=True, mask=mask_seeds,
                force_cyl_closer=run.hit_in_cyl[run.candidates.hitIds(crossing, mask=mask_seeds)],
                origin_coords=origin_coords,
                step=(step + i_step))
            # {{{ the arrays are aligned with the masked(!) candidate list: has_intersection, xi, yi, zi, hel_s }}}
            # {{{ nb_df['extend_index'] points into the masked(!) candidate list (and thereby also into the arrays) }}}

            keep_mask = np.full(run.candidates.n, True)
            if nb_df is not None:
                existing_hit_id = run.candidates.hitIds(crossing, mask=mask_seeds)
                cand_df = pd.DataFrame(data={
                    'existing_hit_id': existing_hit_id
                })
                nb_df = nb_df.join(cand_df, on='extend_index', how='left')

                # CAUTION: We must not assume that 'extend_index' is sorted! Therefore
                #          we are not allowed to align nb_df columns with candidate
                #          data which has been masked by 'has_any_neighbor'!

                # tanslate extend_index into an index pointing into the UNMASKED candidate list
                extend_index = nb_df['extend_index'].values
                candidate_index = np.nonzero(mask_seeds)[0][extend_index]

                # Note: All the boolean arrays we build now are aligned with the UNMASKED candidate list.

                has_any_neighbor = np.full(run.candidates.n, False)
                has_any_neighbor[candidate_index] = True

                existing_is_any = np.full(run.candidates.n, False)
                for col in extend_hit_cols:
                    existing_is_this = (nb_df['existing_hit_id'].values == nb_df[col].values)
                    existing_is_any[candidate_index] = existing_is_any[candidate_index] | existing_is_this

                self.log("existing is not any neighbor found: ", np.sum(mask_seeds & ~existing_is_any))
                self.log("existing is not any neighbor found (but neighbors found): ", np.sum(has_any_neighbor & ~existing_is_any))

                # At the end, drop candidates for which we did not properly find the expected neighbors backwards
                keep_mask[mask_seeds & ~has_any_neighbor] = False
                # XXX we might want to do this, but currently it reduces the score:
                #keep_mask[mask_seeds & (~has_any_neighbor | ~existing_is_any)] = False

                for i_pair, col in enumerate(extend_hit_cols):
                    hit_to_set = np.zeros(run.candidates.n, dtype=np.int32)
                    hit_to_set[candidate_index] = nb_df[col].values
                    # XXX could be optimized, maybe, by implementing 'place'-like semantics for setHitIds.
                    run.candidates.setHitIds(crossing, hit_to_set, pair=i_pair, mask=has_any_neighbor)

            # drop bad candidates
            self.log("dropping bad candidates: ", np.sum(~keep_mask))
            run.candidates.update(keep_mask=keep_mask)
            mask_seeds = mask_seeds[keep_mask]

            keep_mask = self.filterInvalidTrackCandidates(run, step=(10*step + i_step), mask=mask_seeds)
            mask_seeds = mask_seeds[keep_mask]

        # mark pair finding done
        if for_ncross == -1:
             run.candidates.df.loc[mask_seeds, 'donePairs'] = run.candidates.ncross[mask_seeds]
        else:
             run.candidates.df.loc[mask_seeds, 'donePairs'] = for_ncross - 1

        self.log("remaining candidates: ", run.candidates.n)

    def fitTracks(self, run, crossing=-1, mask=None, origin_coords=(0.0, 0.0, 0.0), pitch_from=(1,2), step=0):
        """Fit helix parameters for candidate tracks.
        XXX document arguments
        """
        x0, y0, z0 = run.candidates.hitCoordinates(crossing - 2, mask=mask)
        x1, y1, z1 = run.candidates.hitCoordinates(crossing - 1, mask=mask)
        x2, y2, z2 = run.candidates.hitCoordinates(crossing - 0, mask=mask)
        # we assume the origin as the first point if we have only two points so far
        for coord, origin_coord in zip((x0, y0, z0), origin_coords):
            coord[np.isnan(coord)] = origin_coord

        # fit helices
        hel_xm, hel_ym, hel_r = circleFromThreePoints(x0, y0, x1, y1, x2, y2)
        assert hel_xm.shape == hel_ym.shape == hel_r.shape == x0.shape
        # XXX pitch from three points if available?
        xs = (x0, x1, x2)
        ys = (y0, y1, y2)
        zs = (z0, z1, z2)
        pitch_coords = (coords[index] for index in pitch_from for coords in (xs, ys, zs))
        hel_pitch, _, hel_dz = helixPitchFromTwoPoints(*pitch_coords, hel_xm, hel_ym)
        assert hel_pitch.shape == hel_dz.shape

        hel_pitch_ls, _, _, loss = helixPitchLeastSquares(x0, y0, z0, x1, y1, z1, x2, y2, z2, hel_xm, hel_ym)

        # store helix fit parameters
        run.candidates.setFit(crossing, 'hel_xm'      , hel_xm      , mask=mask)
        run.candidates.setFit(crossing, 'hel_ym'      , hel_ym      , mask=mask)
        run.candidates.setFit(crossing, 'hel_r'       , hel_r       , mask=mask)
        run.candidates.setFit(crossing, 'hel_pitch'   , hel_pitch   , mask=mask)
        run.candidates.setFit(crossing, 'hel_pitch_ls', hel_pitch_ls, mask=mask)
        run.candidates.setFit(crossing, 'hel_ploss'   , loss        , mask=mask)

        # drop implausible fits with extreme parameters
        is_hel_r_too_small = (hel_r < run.params['fit__hel_r_min'])
        if np.any(is_hel_r_too_small):
            if mask is not None:
                keep_mask = np.ones(run.candidates.n, dtype=np.bool)
                keep_mask[mask] = ~is_hel_r_too_small
            else:
                keep_mask = ~is_hel_r_too_small
            self.log("dropping candidates with too small hel_r: %d" % np.sum(is_hel_r_too_small))
            run.candidates.update(keep_mask=keep_mask)

    def evaluateLocalBayesianValue(self, run, x, y, z, cyl_closer, layer_id, xn, yn, zn, hel_s):
        # XXX unify much of this code with bayesianNeighborEvaluation
        # project neighbor displacements to spherical coordinate directions
        d_theta, d_phi = self.projectNeighborDisplacement(x, y, z, xn, yn, zn)

        # evalute neighbor functions at the locations of the neighbor hits
        ld_utheta_abs, ld_uphi_abs, ldnt_utheta_abs, ldnt_uphi_abs = run.neighbors.evaluateLayerFunctions(
            x, y, z, cyl_closer, layer_id,
            functions=('d_utheta_abs', 'd_uphi_abs', 'dnt_utheta_abs', 'dnt_uphi_abs'))

        # calculate formal estimated errors
        e_theta, e_phi = self.predictRandomError(run, hel_s=hel_s,
            e_theta_meas=ldnt_utheta_abs, e_phi_meas=ldnt_uphi_abs)

        # convert neighbor displacement and background hit distances to multiples of the respective error estimates
        de_theta  = d_theta       / e_theta
        de_phi    = d_phi         / e_phi
        dbe_theta = ld_utheta_abs / e_theta
        dbe_phi   = ld_uphi_abs   / e_phi

        # quadrature sums of error-scaled neighbor displacement and error-scaled expected background hit distance
        # XXX could save some np.sqrt evaluations here for speed by comparing the squares
        s = np.sqrt(np.square(de_theta ) + np.square(de_phi ))
        b = np.sqrt(np.square(dbe_theta) + np.square(dbe_phi))

        p0 = run.params['value__p0']
        xsb = np.exp(-np.square(s) / 2) * (2 * np.square(b) + np.pi) * (1 - p0) / np.pi + p0

        value = np.log(xsb)
        return value

    def evaluateLocally(self, run, i_fit, i_test, pair_test=0, fit_crossing=0, analysis=True):
        mask = run.candidates.has([(i_test, pair_test), i_fit, fit_crossing], allow_negative=False)
        x0, y0, z0 = run.candidates.hitCoordinates(i_test, pair=pair_test, mask=mask)
        xf, yf, zf = run.candidates.hitCoordinates(i_fit, mask=mask)
        hel_xm    = run.candidates.getFit(fit_crossing, 'hel_xm'   , mask=mask).astype(np.float64)
        hel_ym    = run.candidates.getFit(fit_crossing, 'hel_ym'   , mask=mask).astype(np.float64)
        hel_r     = run.candidates.getFit(fit_crossing, 'hel_r'    , mask=mask).astype(np.float64)
        hel_pitch = run.candidates.getFit(fit_crossing, 'hel_pitch', mask=mask).astype(np.float64)
        hel_params = (xf, yf, zf, hel_xm, hel_ym, hel_r, hel_pitch)
        (xn, yn, zn, dist) = helixNearestPointDistance(*hel_params, x0, y0, z0)

        if run.hasLayerFunction('d_utheta_abs'):
            hit_ids = run.candidates.hitIds(i_test, pair=pair_test, mask=mask)
            cyl_closer = run.hit_in_cyl[hit_ids]
            layer_id = run.hit_layer_id[hit_ids]
            # XXX produce better error estimates than those based on this pseudo-arc-length
            pseudo_hel_s = np.sqrt(np.square(x0 - xf) + np.square(y0 - yf) + np.square(z0 - zf))
            local_value_masked = run.params['value__bayes_weight'] * self.evaluateLocalBayesianValue(
                run, x0, y0, z0, cyl_closer, layer_id, xn, yn, zn, pseudo_hel_s)
        else:
            r = np.sqrt(np.square(x0) + np.square(y0) + np.square(z0))
            dist /= r
            local_value_masked = -np.square(dist)
        local_value = np.zeros(run.candidates.n, dtype=np.float64)
        np.place(local_value, mask, local_value_masked)
        if analysis:
            def uncompress(x, mask=mask):
                y = np.full(run.candidates.n, np.nan, dtype=np.float64)
                np.place(y, mask, x)
                return y
            local_info = OrderedDict(
                [(k, uncompress(v)) for (k,v) in [('dist', dist), ('locval', local_value)]])
        else:
            local_info = None
        assert not np.any(np.isnan(local_value))
        return (local_value, mask, local_info)

    def evaluateCellFeatureConsistency(self, run):
        """Evaluate how consistent the candidate tracks are with features
        extracted from the cells data.
        Args:
            run (Run): the Run object holding the data structures for this round
        Returns:
            value (np.float32 array (run.candidates.n,)): value of candidate track
                (the higher the value, the better the track)
        """
        value = np.zeros(run.candidates.n, dtype=np.float32)
        # we need at least three crossings to have proper helix fits
        mask = (run.candidates.ncross >= 3)
        # get helix fit parameters
        hel_xm    = run.candidates.getFit(2, 'hel_xm'   ).astype(np.float64)
        hel_ym    = run.candidates.getFit(2, 'hel_ym'   ).astype(np.float64)
        hel_r     = run.candidates.getFit(2, 'hel_r'    ).astype(np.float64)
        hel_pitch = run.candidates.getFit(2, 'hel_pitch').astype(np.float64)
        # Note: Currently we do not need to know hel_dz (the direction of movement along
        #       the z-axis) for this consistency check, as the sign of the direction
        #       cannot be extracted from the cell features, anyway.
        #       This could change if we start to evaluate consistency differently
        #       depending on the direction of incidence.
        hel_dz    = np.full(len(hel_pitch), 1.0, dtype=np.float64)
        hel_params = (hel_dz, hel_xm, hel_ym, hel_r, hel_pitch)
        # iterate over the first 4 layer crossings and paired hits
        for i_crossing in range(4):
            for i_pair in range(1): # XXX iterate over all paired hits
                # only check tracks which have the crossing we look at
                # and which are in one of the innermost four cylinders
                mask_check = mask & run.candidates.has([i_crossing, i_pair])
                hit_id = run.candidates.hitIds(i_crossing, pair=i_pair, mask=mask_check)
                is_inner_cyl = run.hit_in_cyl[hit_id] & (run.hit_layer_id[hit_id] < 4)
                mask_check[mask_check] = is_inner_cyl
                hit_id = hit_id[is_inner_cyl]
                assert np.sum(mask_check) == len(hit_id)
                # get the unit tangent vector to the helix at the hit
                x, y, z = run.event.hitCoordinatesById(hit_id)
                hel_params_masked = tuple(par[mask_check] for par in hel_params)
                udir = helixUnitTangentVector(x, y, z, *hel_params_masked)
                udir_cells, inner, _ = run.cell_features.estimateClosestDirection(hit_id, udir)
                # calculate a concistency score/loss
                value[mask_check] += (inner - run.params['value__cells_bias'])
        return value

    def evaluateTracks(self, run, analysis=True):
        """Calculate a heuristic value of each candidate track.
        Args:
            run (Run): the Run object holding the data structures for this round
            analysis (bool): If True, store data for off-line analysis.
        Returns:
            value (np.float32 array (run.candidates.n,)): value of candidate track
                (the higher the value, the better the track)
            eval_df (pd.DataFrame or None): if analysis==True, this is a dataframe
                aligned with the candidates list. `None` if not `analysis`.
        """
        ncross = run.candidates.ncross
        max_ncross = np.amax(ncross, axis=0)
        value = np.zeros(len(ncross), dtype=np.float32)
        nvalues = np.zeros(len(ncross), dtype=np.int8)

        # for collecting info for offline analysis (optional)
        info = OrderedDict()

        # evaluate distance of hits from backwards-projected helices
        for i in range(max_ncross - 3):
            for i_pair in range(2):
                local_value, has_value, local_info = self.evaluateLocally(
                    run, i+1, i, pair_test=i_pair, fit_crossing=i+3,
                    analysis=analysis)
                value += local_value
                nvalues += has_value
                if analysis and i_pair == 0:
                    info.update(OrderedDict([(('%d_' % i) + key, value) for key, value in local_info.items()]))

        # evaluate distance of latest hit from forwards-projected helix going
        # through the fourth-latest hit
        for i_pair in range(2):
            local_value, has_value, local_info = self.evaluateLocally(
                run, ncross-4, ncross-1, pair_test=i_pair, fit_crossing=ncross-2,
                analysis=analysis)
            value += local_value
            nvalues += has_value
            if analysis and i_pair == 0:
                info.update(OrderedDict([('f_' + key, value) for key, value in local_info.items()]))

        # evaluate loss in least-squares fitting of helix pitch
        if run.params['value__ploss_weight'] != 0:
            value_hel_ploss = np.zeros(run.candidates.n, dtype=np.float32)
            for i in range(2, max_ncross):
                hel_ploss = run.candidates.getFit(i, 'hel_ploss')
                hel_ploss_valid = np.isfinite(hel_ploss)
                value_hel_ploss[hel_ploss_valid] += run.params['value__ploss_bias'] - hel_ploss[hel_ploss_valid]
            value += run.params['value__ploss_weight'] * value_hel_ploss

        # evaluate changes of helix parameters along the tracks
        # XXX this seriously needs improvement
        for i_ncross in range(4, max_ncross + 1):
            mask = (ncross == i_ncross)
            hel_pitches = [run.candidates.getFit(i, 'hel_pitch', mask=mask) for i in range(2, i_ncross)]
            hel_pitches = np.stack(hel_pitches, axis=1)
            hel_rs = [run.candidates.getFit(i, 'hel_r', mask=mask) for i in range(2, i_ncross)]
            hel_rs = np.stack(hel_rs, axis=1)

            hel_curvatures = hel_rs / (np.square(hel_pitches / (2*np.pi)) + np.square(hel_rs))
            hel_mean = np.mean(np.abs(hel_curvatures), axis=1)
            hel_std = np.std(np.sign(hel_pitches) * hel_curvatures, axis=1)
            fit_value = -run.params['value__fit_weight_hcs'] * (hel_std / hel_mean) / i_ncross

            value[mask] += fit_value

        # tracks with only two crossings ("doubles") are not scored, currently
        mask_doubles = (ncross == 2)
        assert np.all(value[mask_doubles] == 0)

        # some special handling for tracks with only three crossings ("triples")
        mask_triples = (ncross == 3)
        if run.params['follow__weird_triples']:
            value[mask_triples] = -run.candidates.getFit(2, 'hel_ploss', mask=mask_triples)
        else:
            value[mask_triples] -= run.candidates.df['dist'][mask_triples] # XXX refine
        nvalues[mask_triples] += run.candidates.nHits()[mask_triples]

        # evaluate consistency of candidate tracks with cells data features
        if run.cell_features is not None:
            value += run.params['value__cells_weight'] * self.evaluateCellFeatureConsistency(run)

        # evaluate per-hit and per-layer-crossing boni
        value = value + run.params['value__hit_bonus'] * nvalues + run.params['value__cross_bonus'] * ncross

        # optionally prepare data for offline analysis
        # XXX this should probably be put into run.candidates.df insteaf of in a separate dataframe
        if analysis:
            info['value'] = value
            eval_df = pd.DataFrame(data = info)
        else:
            eval_df = None

        return (value, eval_df)

    def dropRedundantTracks(self, run, eval_df, nlayers=3):
        """Drop candidates which are redundant in that they have the same hits
        recorded for the given number of layer crossings at the start.
        """
        keep_mask = run.candidates.findUnique(crossings = list(range(nlayers)))
        self.log("dropping redundant candidates: ", np.sum(~keep_mask))
        run.candidates.update(keep_mask = keep_mask)
        self.log("remaining candidates: ", run.candidates.n)
        # XXX eval_df data should probably be put in run.candidates.df
        if eval_df is not None:
            eval_df = eval_df.loc[keep_mask].reset_index(drop=True)
        return eval_df

    def filterInvalidTrackCandidates(self, run, mask=None, step=0, prev_candidates=None):
        """Filter out candidates which repeatedly hit exactly the same point within the
        latest three layer crossings.
        Returns:
            keep_mask (bool array): aligned with the candidates list *before* dropping
                the candidates, this array is true for the kept candidates.
        """
        xyz0 = run.candidates.hitCoordinates(-3, mask=mask)
        xyz1 = run.candidates.hitCoordinates(-2, mask=mask)
        xyz2 = run.candidates.hitCoordinates(-1, mask=mask)
        is_same_xyz_01 = tuple(a == b for a, b in zip(xyz0, xyz1))
        is_same_xyz_02 = tuple(a == b for a, b in zip(xyz0, xyz2))
        is_same_xyz_12 = tuple(a == b for a, b in zip(xyz1, xyz2))
        is_same_xy_01 = np.all(is_same_xyz_01[:2], axis=0)
        is_same_xy_02 = np.all(is_same_xyz_02[:2], axis=0)
        is_same_all_01 = is_same_xy_01 & is_same_xyz_01[2]
        is_same_all_02 = is_same_xy_02 & is_same_xyz_02[2]
        is_same_all_12 = np.all(is_same_xyz_12, axis=0)
        ###for i, is_same in enumerate((is_same_01, is_same_02, is_same_12)):
        ###    if np.any(is_same):
        ###        print("step %d: is_same %d" % (step, i))
        ###        if mask is None:
        ###            problem_mask = is_same
        ###        else:
        ###            problem_mask = np.zeros(run.candidates.n, dtype=np.bool)
        ###            problem_mask[mask] = is_same
        ###        for name, cand in (('prev', prev_candidates), ('this', run.candidates)):
        ###            if cand is not None:
        ###                print(name, ": ", cand.__str__(mask=problem_mask, show_coords=True, show_r2=True, show_fit=True,
        ###                      show_pid=True, show_df=True, break_cross=True,
        ###                      hit_in_cyl=run.hit_in_cyl, hit_layer_id=run.hit_layer_id))
        is_invalid = is_same_all_01 | is_same_all_02 | is_same_all_12 | (is_same_xy_01 & is_same_xy_02)
        if np.any(is_invalid):
            self.log("dropping candidates with repeated hit coordinates: %d" % np.sum(is_invalid))
        if mask is None:
            keep_mask = ~is_invalid
        else:
            keep_mask = np.ones(run.candidates.n, dtype=np.bool)
            keep_mask[mask] = ~is_invalid
        run.candidates.update(keep_mask=keep_mask)
        return keep_mask

    def nonPhysicalPostprocessOddHits(self, run, used, submission_df, k=4):
        """Add best-guess unused hits to candidates with an even number of hits.
        This is a pure score optimization which is not physically motivated. It exploits
        artifacts of the scoring function used in the trackml competition:
          - The score is a monotonically rising function of the summed weight of hits.
          - The score is a monotonically falling function of floor(nhits / 2).
        Due to the second property, adding a hit to a candidate track with an even number
        of hits can only ever increase the candidate's score, never decrease it.
        This fact was originally observed by Grzegorz Sionkowski and published on the
        kaggle discussion forum:
        https://www.kaggle.com/c/trackml-particle-identification/discussion/60638#354053
        """
        # select tracks with an even number of hits
        tracks_df = submission_df.loc[submission_df['event_id'] == run.event.event_id].copy()
        tracks_df['nhits'] = tracks_df.groupby('track_id', sort=False)['hit_id'].transform('count')
        tracks_df = tracks_df.loc[tracks_df['nhits'] % 2 == 0]
        # get coordinates of unused hits
        unused_hit_ids, = np.where(~used)
        if len(unused_hit_ids) and unused_hit_ids[0] == 0:
            unused_hit_ids = unused_hit_ids[1:]
        unused_hit_coords = run.event.hitCoordinatesById(unused_hit_ids)
        # find k-neighbors of unused hits
        # Note: We use the neighborhoods filled with all hits of the event for this search.
        unused_hit_layers = run.hit_in_cyl[unused_hit_ids], run.hit_layer_id[unused_hit_ids]
        nb_df = run.full_neighbors.findIntersectionNeighborhoodK(*unused_hit_coords, *unused_hit_layers, k=k)
        # XXX Note: Would be better to also search unsuccessful layer intersection points, here.
        nb_df['unused_hit_id'] = unused_hit_ids[nb_df['vind'].values]
        nb_df.rename(columns={'nb_hit_id': 'hit_id'}, inplace=True)
        nb_df = nb_df.merge(tracks_df, on='hit_id', how='inner', sort=False)
        nb_df.drop(columns='hit_id', inplace=True)
        # XXX should use helix distance
        nb_df = nb_df.sort_values('nb_dist')
        nb_df = nb_df.groupby('unused_hit_id', sort=False, as_index=False).first()
        nb_df = nb_df.groupby('track_id', sort=False, as_index=False).first()
        nb_df.rename(columns={'unused_hit_id': 'hit_id'}, inplace=True)
        add_df = nb_df[['event_id', 'hit_id', 'track_id']].sort_values('hit_id')
        # mark hits as used
        assert not np.any(used[add_df['hit_id'].values])
        used[add_df['hit_id'].values] = True
        return add_df

    def paramsForRun(self, i_commit=0, sunset=False):
        """Determine parameters to use for the given commit round.
        This function implements a kind of mini-language in the system for
        hyper-parameters that allows things like:
            *) override/modify specific parameters only for the "sunset" round
            *) override/modify specific parameters "upto" or starting "from"
                a given round index
        Args:
            i_commit (int): zero-based index of the commit round
            sunset (bool): if True, this is the last ("sunset") round, in which
                we take desparate measures
        """
        params = self.params.copy()
        conditions = OrderedDict([
            ('upto'  , lambda match, i=i_commit   : 1 if i <= int(match.group(2)) else 0),
            ('from'  , lambda match, i=i_commit   : 1 if i >= int(match.group(2)) else 0),
            ('sunset', lambda match, sunset=sunset: 2 if sunset                   else 0),
        ])
        # Note: Within each bucket, operations are applied in the order listed.
        operations = OrderedDict([
            # Damn Python purists: Why not simply allow assignments in lambda?
            ('ADD__', lambda params, key, value: params.__setitem__(key, params[key] + value)),
            ('SUB__', lambda params, key, value: params.__setitem__(key, params[key] - value)),
            (''     , lambda params, key, value: params.__setitem__(key,               value)),
        ])
        regex = ( '(' + '|'.join(conditions.keys()) + ')(\d*)__'
                + '(' + '|'.join(operations.keys()) + ')(.*)')
        # bucket 0 is for inactive parameters, the rest of the buckets are applied in index order
        buckets = [{ op: OrderedDict() for op in operations.keys() } for i in range(3)]
        for key, value in params.items():
            match = re.match(regex, key)
            if match:
                bucket = conditions[match.group(1)](match)
                name = match.group(4)
                if name not in params:
                    print("warning: '%s' adds new param '%s'" % (key, name), file=sys.stderr)
                buckets[bucket][match.group(3)][name] = value
        for bucket in buckets[1:]:
            for op, executor in operations.items():
                for key, value in bucket[op].items():
                    executor(params, key, value)
        return params

    def setupRun(self, event, i_commit=0, sunset=False, used=None, first_run=None,
                 fit_columns=None, supervisor=None,
                 cell_details=None):
        """Setup data structures for running a round of the algorithm.
        Args:
            event (data.Event): the event for which to run the algorithm
            i_commit (int): commit round index (0 for first round)
            sunset (bool): True if this will be the sunset round
            used (None or bool array(1 + max_hit_id,)): if given, used[hit_id] == True for
                hit_ids which have been committed already in earlier rounds
            first_run (None or Run): None for setting up the first round, otherwise
                the Run object for the first round
            fit_columns (None or list of str): If given, this list defines the
                fit parameters that will be stored in the candidates list.
            supervisor (supervised.Supervisor): supervisor object (XXX remove?)
            cell_details (None or SimpleNamespace): If given, this is filled with some
                detailed data about cell features for off-line analyis.
                (Only if cell features are actually calculated in this call.)
        Returns:
            run (Run): the Run object holding the data structures for this round
        """
        # XXX should most of the data setup be moved to Run.__init__?
        if i_commit == 0:
            # prepare arrays for remembering in which layer each hit resides
            hit_in_cyl   = np.zeros(1 + event.max_hit_id, dtype=np.bool)
            hit_layer_id = np.full (1 + event.max_hit_id, -1, dtype=np.int8)
        else:
            # these have already been calculated and will be reused from the first run
            hit_in_cyl = None
            hit_layer_id = None

        run = self.Run(self, supervisor, event,
                       params=self.paramsForRun(i_commit=i_commit, sunset=sunset),
                       hit_in_cyl=hit_in_cyl, hit_layer_id=hit_layer_id,
                       layer_functions=self.layer_functions,
                       used=used, fit_columns=fit_columns)

        if i_commit == 0:
            # this is the first round; remember the full neighborhoods
            run.full_neighbors = run.neighbors

            # calculate cell features if we have cell data
            cell_features = None
            if event.has_cells:
                cell_features = CellFeatures(self, run, details=cell_details)
            run.cell_features = cell_features
        else:
            # take over some information from the first round
            assert first_run is not None
            run.hit_in_cyl = first_run.hit_in_cyl
            run.hit_layer_id = first_run.hit_layer_id
            run.full_neighbors = first_run.neighbors
            run.cell_features = first_run.cell_features

        return run

    def createOrAppendSubmissionFile(self, submission_filename, submission_df, append=False):
        """Create a submission file from the given dataframe, or append to it.
        Args:
            submission_filename (str): path of the submission file
            submission_df (pd.DataFrame): submission dataframe
            append (bool): If True, append to the submission file, which is assumed
                to exist. If False, create or overwrite the submission file
        """
        submission_columns = np.array(['event_id', 'hit_id', 'track_id'])
        if append:
            mode = 'a'
            header = False
        else:
            mode = 'w'
            header = True
        assert np.all(submission_df.columns == submission_columns)
        submission_df.to_csv(submission_filename, index=False, mode=mode, header=header)

    def findTracks(self, supervisor, events_test,
                   submission_filename=None,
                   analysis=True, score_intermediate=True, score_final=True):
        """
        Args:
            supervisor: XXX remove this argument? XXX
            events_test (list of data.Event): The test data to work on.
            submission_filename (str or None): If given, write the submission
                to the given path
            analysis (bool): If True, store data for off-line analysis.
            score_intermediate (bool): If True, calculate score at the end of each
                track extension iteration (if ground truth is available).
            score_final (bool): If True, calculate score for each processed
                event (if ground truth is available).
        Returns:
            scores (list): list of scores for the given events, if ground
                truth was available and score_final==True,
                otherwise an empty list.
        """
        # print parameters and layer_functions used to the log file
        self.log("params:")
        with self.indent:
            self.log(pprint.PrettyPrinter(indent=4).pformat(self.params))
        if self.layer_functions is not None:
            self.log("layer_functions: " + ", ".join(self.layer_functions.functions))

        scores = []
        # Loop over events
        for i_event, event_test in enumerate(events_test):
            submission_df = None
            score = None
            event_test.open()
            self.log("event_test: ", event_test.summary())

            with self.timed('event %d' % event_test.event_id):

                used = np.zeros(1 + event_test.max_hit_id, dtype=np.bool)
                submission_df_parts = []
                min_track_id = 1
                sunset = False
                first_run = None

                # Commit loop:
                #     In each commit loop iteration ("round"), we commit a subset of
                #     the candidate tracks to the submission dataframe. The hits used
                #     by the committed candidates will be marked as used so that
                #     they are removed from the neighborhoods searched by
                #     subsequent commit iterations.
                #     Basic parameterizations of the algorithm use only a single
                #     commit loop.
                for i_commit in range(int(self.params['commit__niter']) + 1):
                    with self.timed('round %d (%6d hits)' % (i_commit, np.sum(~used) - 1) + (' (sunset)' if sunset else '')):
                        # set up run data for this round
                        run = self.setupRun(event_test, i_commit=i_commit, sunset=sunset,
                                            used=used, first_run=first_run,
                                            supervisor=supervisor)
                        if i_commit == 0:
                            first_run = run
                        with self.indent:
                            self.log(pprint.PrettyPrinter(indent=4).pformat(run.params))

                        # calculate the "quadratic funnel" endpoints for this round
                        nhits = float(event_test.max_hit_id)
                        ntop_quadratic = run.params['rank__ntop_qu'] * (nhits ** 2)
                        ntop_linear    = run.params['rank__ntop_li'] *  nhits
                        self.log("ntop_quadratic = %.0f, ntop_linear = %.0f" % (ntop_quadratic, ntop_linear))

                        with self.timed('chooseLikelyFirstHits'):
                            nh = self.chooseLikelyFirstHits(run)

                        with self.timed('chooseLikelySecondHits'):
                            secnh = self.chooseLikelySecondHits(run, nh['hit_id'])
                            assert secnh is not None # XXX handle secnh is None
                            nseeds = len(secnh)
                            self.log("chosen candidates for second hits: ", nseeds)

                        # XXX this stuff should really be factored into a method:
                        xf, yf, zf = run.event.hitCoordinatesById(secnh['nb_hit_id'])
                        df = pd.DataFrame(data=OrderedDict([
                            ('xf', xf), ('yf', yf), ('zf', zf),
                            ('dphi', np.zeros(nseeds, dtype=np.float64)),
                            ('hel_s', np.zeros(nseeds, dtype=np.float64)),
                            ('nskipped', np.zeros(nseeds, dtype=np.int8)),
                            ('donePairs', np.zeros(nseeds, dtype=np.int8)),
                        ]))
                        run.candidates.addSeeds([secnh['hit_id'], secnh['nb_hit_id']], df)
                        self.filterInvalidTrackCandidates(run, step=0)

                        with self.timed('fitting'):
                            self.fitTracks(run, step=-1)

                        with self.timed('follow tracks'):
                            # Follow loop:
                            #     In each iteration of this loop, track candidates will be
                            #     extended by at most one layer crossing.
                            for i in range(int(run.params['follow__niter'])): # XXX refine
                                # choose extension candidates
                                with self.timed('chooseLikelyNextHits'):
                                    self.chooseLikelyNextHits(run,
                                        k=(int(run.params['follow__weird_k']) if i==0 and run.params['follow__weird_triples'] else 4),
                                        nmax_per_crossing=4, step=5*i,
                                        nhits_min_keep = 2 if i < run.params['follow__nskip_max'] else 3)
                                    self.filterInvalidTrackCandidates(run, step=(50*i+1))
                                    self.log("number of candidates: ", run.candidates.n)

                                with self.timed('fitting'):
                                    self.fitTracks(run, step=5*i)

                                with self.timed('findPairs'):
                                    # complete paired hits for triples
                                    self.findPairs(run, step=(5*i+1), k=int(run.params['follow__pairs_k']))
                                    if run.hasLayerFunction('pair_theta'):
                                        # find paired hits for the latest layer crossing
                                        self.findPairs(run, for_ncross=-1, step=(5*i+3), k=int(run.params['follow__pairs_k']))

                                with self.timed('evaluating'):
                                    value, eval_df = self.evaluateTracks(run, analysis=analysis)
                                    self.log("value ", self.shortStats(value))

                                with self.timed('ranking'):
                                    ranking = np.argsort(-value)

                                    # interpolate between quadratic and linear funnel sizes
                                    # XXX should be refactored into a method
                                    nip = int(run.params['follow__niter']) - 1
                                    xip =     i     / nip
                                    yip = (nip - i) / nip
                                    ntop = xip * ntop_linear + yip * ntop_quadratic
                                    ntop = int(min(ntop, int(run.params['rank__ntop'])))
                                    self.log("ntop[%2d] = %6d" % (i, ntop))

                                    ranking = ranking[:ntop]
                                    run.candidates.permute(ranking)
                                    if eval_df is not None:
                                        eval_df = eval_df.iloc[ranking].reset_index()

                                with self.timed('dropRedundantTracks'):
                                    if i >= run.params['follow__drop_start']:
                                        eval_df = self.dropRedundantTracks(run, eval_df, nlayers = 3 if i < run.params['follow__nskip_max'] else 2)

                                # save data for off-line analysis (optional)
                                if eval_df is not None:
                                    run.candidates.df['%d_rank' % i] = np.arange(run.candidates.n)
                                    cand_df = run.candidates.analysisDataframe()
                                    eval_df = pd.concat([eval_df, cand_df], axis=1)
                                    eval_df.to_csv('eval-%d.dat' % (i - 1), index=False)

                                # calculate intermediate score per follow iteration (optional)
                                if score_intermediate and event_test.has_truth:
                                    submission_df = run.candidates.submit()
                                    score = score_event(event_test.truth_df, submission_df)
                                    self.log("")
                                    self.log("score: %.4f" % score, " nhits = ", self.shortStats(run.candidates.nHits(), fmt='%.1f'))

                        # XXX add general way to coerce params to integer type
                        overfull = run.candidates.n > int(run.params['commit__nmax'])
                        last_round = sunset or (i_commit == int(self.params['commit__niter']) - 1) or not overfull
                        if overfull and not last_round:
                            # restrict to the top commit__nmax candidates
                            run.candidates.permute(np.arange(int(run.params['commit__nmax'])))

                        # XXX clean up handling of sunset parameters
                        if last_round and not sunset and any(x.startswith('sunset__') for x in self.params.keys()):
                            last_round = False
                            sunset = True

                        # build the part of the submission dataframe for this round
                        # Note: In the final round, we do this in two steps. The first one
                        #       works the same as in the previous rounds. The second one
                        #       is a desparate attempt to throw all the remaining stuff
                        #       we have into the submission and hope it is better than noise.
                        for desperation in range(1 + last_round):
                            submission_df_part = run.candidates.submit(fill=False, used=used, min_track_id=min_track_id,
                                min_nhits        =(3    if desperation else run.params['commit__min_nhits']),
                                max_nloss        =(None if desperation else run.params['commit__max_nloss']),
                                max_loss_fraction=(1.0  if desperation else run.params['commit__max_loss_fraction']),
                                reserve_skipped=False)
                            min_track_id += run.candidates.n
                            used[submission_df_part['hit_id'].values] = True
                            submission_df_parts.append(submission_df_part)

                        if last_round: break

            submission_df = pd.concat(submission_df_parts, axis=0, ignore_index=True)

            # "non-physical" score optimization
            if self.params['post__nonphys_odd']:
                add_df = self.nonPhysicalPostprocessOddHits(run, used, submission_df)
                submission_df = pd.concat([submission_df, add_df], axis=0, ignore_index=True)

            # fill the submission up so it uses all hits
            fill_df = Candidates(event_test).submit(fill=True, used=used, min_track_id=min_track_id)
            submission_df = pd.concat([submission_df, fill_df], axis=0, ignore_index=True)

            # write the submission file
            if submission_filename is not None:
                self.createOrAppendSubmissionFile(submission_filename, submission_df, append=(i_event > 0))

            # calculate the score for this event (if we have ground truth)
            if score_final and event_test.has_truth:
                score = score_event(event_test.truth_df, submission_df)
                scores.append(score)
                self.log("")
                self.log("event %4d score: %.4f" % (event_test.event_id, score))

            event_test.close()

        return scores
