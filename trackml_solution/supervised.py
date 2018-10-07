"""Supervised learning.

The code in this file tries to extract useful information from training
events and turns it into mathematical models to be used by the track
finding algorithm.

Currently no sophisticated models are build. Only the following data
is learned from training events:

* `Supervisor.learnLayerFunctions`:
  The average displacement between the true hit position and the measured
  hit position and the expected distance from a point to the nearest
  hit.
  See `learn_layer_functions.bat` for how to invoke.

* `Supervisor.learnIntersectionDisplacements`,
  `Supervisor.learnIntersectionDisplacementsLayer`:
  Learns the systematic deviations of particle trajectories from idealized
  helix trajectories.
  See `learn_displacements.bat` for how to invoke.

* `Supervisor.analyzePairs`, `Supervisor.analyzePairsFirstLayers`,
  `Supervisor.learnPairing`, `Supervisor.learnPairingLayer`:
  Tries to find good cut-off values for identifying "paired" hits
  (meaning the group of 1 to 4 hits that a particle can cause in
  a single detector layer crossing).
  See `learn_pairs.bat` for how to invoke.

There is some code which trains models to predict the errors of
extrapolated trajectories but these models ended up not being used in the
final solution.

The code also gathers a lot of statistics that were used to guide the
development of the algorithm but which are not used directly to find solutions.

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

import os
from collections import OrderedDict
from types import SimpleNamespace
from string import ascii_lowercase

import numpy as np
import pandas as pd
import scipy.stats as stats

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor
from sklearn.cluster import KMeans
from xgboost import XGBRegressor

from trackml.score import score_event

from trackml_solution.neighbors import Neighbors
from trackml_solution.geometry import helixNearestPointDistance, helixDirectionFromTwoPoints
from trackml_solution.corrections import HelixCorrector
from trackml_solution.algorithm import Algorithm
from trackml_solution.logging import Logger

def lookupTable(values, keys, max_key=None, default=0):
    if max_key is None:
        max_key = keys.max()
    lut = np.full(1 + max_key, default, dtype=values.dtype)
    lut[keys] = values
    return lut

def frequencyInRows(array, empty_value=-1):
    """Find unique values in each row, sort and report their frequencies per row.
    Args:
        array (array (N,M)): the array to analyse.
        empty_value (scalar of array.dtype): a special value to be considered an "empty" entry
            in the array.
    Returns:
        freq (array (N,M)): element (i,j) is the frequency of unique(i,j) in row i.
        unique (array (N,M)): element (i,j) is the (j+1)-th most frequent value in row i.
            (j=0 for the most frequent item).
    """
    array_sorted = np.flip(np.sort(array, axis=1), axis=1)
    array_sorted = np.concatenate([array_sorted, np.full((array_sorted.shape[0],1), empty_value,
                                                         dtype=array_sorted.dtype)], axis=1)
    unequal = (array_sorted[:,1:] != array_sorted[:,:-1])
    assert unequal.shape == array.shape
    where_unequal = np.nonzero(unequal)
    start_new_row = (where_unequal[0] != np.roll(where_unequal[0], 1))
    row_offset = np.zeros(len(start_new_row), dtype=np.int32)
    row_offset[start_new_row] = np.arange(len(start_new_row))[start_new_row]
    row_offset = np.maximum.accumulate(row_offset)
    col_index = np.arange(len(start_new_row)) - row_offset
    prev_run_end = np.roll(where_unequal[1], 1)
    prev_run_end[start_new_row] = -1
    run_lengths = where_unequal[1] - prev_run_end
    freq = np.zeros(array.shape, dtype=np.int8)
    freq[where_unequal[0], col_index] = run_lengths
    unique = np.full(array.shape, empty_value, dtype=array.dtype)
    unique[where_unequal[0], col_index] = array_sorted[where_unequal[0], where_unequal[1]]
    freq_sort = np.argsort(-freq, axis=1)
    freq = freq[np.arange(freq_sort.shape[0])[:,np.newaxis], freq_sort]
    unique = unique[np.arange(freq_sort.shape[0])[:,np.newaxis], freq_sort]
    return (freq, unique)

class LayerFunctions:
    """Class for storing per-layer interpolated functions."""
    def __init__(self, df=None, dfname=None):
        self._function_data = OrderedDict()
        self._dfs = []
        if df is not None:
            self.add(df, dfname=dfname)

    @property
    def functions(self):
        return self._function_data.keys()

    def getGridValues(self, is_cylinder, layer_id, functions):
        """Return grid points and values for the given layer functions.
        Args:
            is_cylinder (bool): whether to look for a cylinder layer (True)
                or a cap layer (False).
            layer_id (int): identifies the layer: cyl_id for is_cylinder == True,
                cap_id for is_cylinder == False.
                XXX maybe clean up our terminology to distinguish this better
                    from the "layer_id" column given in the trackml data.
            functions (list of str): names of the functions for which to return
                data.
        Returns:
            data (list of tuples (points, values)): for each function in the
                list given as the `functions` argument:
                *) points is a tuple of of arrays giving the grid points for
                   each of the grid dimensions
                *) values is an array of shape (len(points[0]), len(points[1], ...)
                   giving the function values at the grid points
                Note: The data layout of points and values is as expected by
                scikit.interpolate.RegularGridInterpolator.
        """
        data = []
        for function in functions:
            fn = self._function_data[function]
            df = fn['df']
            layer_df = df.loc[(df['is_cylinder'] == is_cylinder) & (df['layer_id'] == layer_id)]
            if len(layer_df) > 0:
                layer_data = self._convertDataFrame(fn, layer_df)
            else:
                layer_data = (None, None)
            data.append(layer_data)
        return data

    def add(self, df, dfname=None, filename=None):
        """Add layer functions stored in the given data frame to this object.
        """
        # find the independent variables (corresponding to grid dimensions)
        # in the given dataframe
        indep_cols = [col for col in df.columns if col.startswith('indep')]
        # find all functions defined in the given dataframe
        non_function_cols = ['is_cylinder', 'layer_id'] + indep_cols
        for col in df.columns:
             if col not in non_function_cols:
                 if col in self._function_data:
                     msg = "duplicate layer function '" + col + "'"
                     if dfname is not None:
                         msg += " in dataframe '" + dfname + "'"
                     if filename is not None:
                         msg += " in file '" + filename + "'"
                     old_filename = self.functions[col]['filename']
                     if old_filename is not None:
                         msg += " (already loaded from file '" + old_filename + "')"
                     raise RuntimeError(msg)
                 self._function_data[col] = {
                     'df': df,
                     'indeps': indep_cols,
                     'filename': filename,
                     'column': col,
                 }
        self._dfs.append({
            'df': df,
            'dfname': dfname,
        })

    def from_csv(prefixes):
        lf = LayerFunctions()
        if not (isinstance(prefixes, list) or isinstance(prefixes, tuple)):
            prefixes = prefixes,
        for prefix in prefixes:
            path = prefix + '.csv'
            lf.load_csv(path)
        return lf

    def load_csv(self, filename, dfname=None):
        df = pd.read_csv(filename, dtype={'is_cylinder': np.bool, 'layer_id': np.int8})
        self.add(df, filename=filename, dfname=dfname)

    def to_csv(self, prefix):
        for dfdata in self._dfs:
            path = prefix
            if dfdata['dfname'] is not None:
                path += '-' + dfdata['dfname']
            path += '.csv'
            dfdata['df'].to_csv(path, index=False)

    def _convertDataFrame(self, fn, layer_df):
        n = len(layer_df)
        shape = []
        points = []
        stride = 1
        for indep_col in reversed(fn['indeps']):
            assert n % stride == 0
            indep = layer_df[indep_col].values[::stride]
            next_dim_indices, = np.where(indep == indep[0])
            if len(next_dim_indices) > 1:
                dimsize = next_dim_indices[1]
                point_coords = indep[:dimsize]
                assert len(indep) % dimsize == 0
                assert np.all(indep == np.tile(point_coords, len(indep) // dimsize))
            else:
                dimsize = len(indep)
                point_coords = indep
            points.insert(0, point_coords)
            shape.insert(0, dimsize)
            stride *= dimsize
        values = layer_df[fn['column']].values.reshape(shape)
        return (tuple(points), values)

def discardOutliersBeforeTrackSorting(run, df, particle_weight_df):
    # discard particles which underwent extreme changes of z-momemtum
    # Note: I do not really know how to deal with these intelligently and
    #       these buggers have cost me too much time already, so let's
    #       toss them out.
    mom_df = df.merge(run.event.particles_df[['particle_id', 'px', 'py', 'pz', 'nhits']],
                      on='particle_id', how='left', sort=False)
    mom_df['p'] = np.sqrt(np.sum([np.square(mom_df[col]) for col in ('px', 'py', 'pz')], axis=0))
    mom_df.loc[mom_df['pz'] == 0, 'pz'] = 0.001 # dummy momemtum to avoid division by zero
    mom_df['eta'] = np.arctanh( mom_df['pz'] / mom_df['p'] )
    mom_df['reltpz'] = mom_df['tpz'] / mom_df['pz']
    pz_df = mom_df.groupby('particle_id', sort=False, as_index=False).agg(
        {'reltpz': ['min', 'max'], 'eta': 'first', 'nhits': 'first'})
    pz_df.columns = ['_'.join(col).rstrip('_') for col in pz_df.columns.values]
    reltpz_tolerance = np.where(pz_df['eta_first'].abs() > 0.1 , 0.2, # XXX refine this
                       np.where(pz_df['eta_first'].abs() > 0.01, 1.0,
                                                                 np.inf))
    has_crazy_reltpz = ( (pz_df['reltpz_max'] > (1 + reltpz_tolerance))
                       | (pz_df['reltpz_min'] < (0 - reltpz_tolerance)) )
    crazy_pz_df = pz_df.loc[has_crazy_reltpz]
    crazy_pz_df = crazy_pz_df.join(particle_weight_df, on='particle_id', how='left', sort=False)
    print("    rejecting %d particles (total weight %.4f) due to extreme z-momentum changes"
        % (len(crazy_pz_df), crazy_pz_df['weight'].sum()))
    ###with pd.option_context('display.max_rows', None, 'display.width', None):
    ###    print(crazy_pz_df.loc[crazy_pz_df['weight'] > 0])
    print("nhits before momentum selection: %d" % len(df))
    df = df.loc[~df['particle_id'].isin(crazy_pz_df['particle_id'])]
    return df

def crossIdGenerator(ntracks, nlay_id):
    def crossId(track_id, in_cyl, lay_id):
        cross_id = (track_id.astype(np.int32)
                   + (1 + ntracks) * (lay_id.astype(np.int32) + nlay_id * in_cyl.astype(np.int32)))
        return cross_id
    return crossId

def getTrueHitsAndTracks(run):
    # XXX move data frame building into a function?
    # get true hits which aren't noise
    df = run.event.truth_df.loc[run.event.truth_df['particle_id'] != 0].copy()

    # calculate particle weight
    particle_weight_df = (run.event.truth_df[['particle_id', 'weight']]
                          .groupby('particle_id')['weight'].sum().to_frame())

    # discard some crazy particles before we attempt track sorting
    df = discardOutliersBeforeTrackSorting(run, df, particle_weight_df)

    # assign shorter unique ids to the true particles which caused hits
    unique_pids, inverse_pids = np.unique(df['particle_id'].values, return_inverse=True)
    assert inverse_pids.max() < np.iinfo(np.int16).max
    df['track_id'] = 1 + inverse_pids.astype(np.int16)
    ntracks = df['track_id'].max()
    print("ntracks: ", ntracks)
    nhits = len(df)
    print("nhits: %d (average %.2f per track_id)" % (nhits, nhits / ntracks))
    # organize particle ground truth by the new track_id
    tracks_df = pd.DataFrame(data=OrderedDict([
        ('track_id', np.arange(1, 1 + ntracks, dtype=np.int16)),
        ('particle_id', unique_pids),
    ]))
    tracks_df = tracks_df.merge(run.event.particles_df, on='particle_id', how='left', sort=False)
    tracks_df = tracks_df.join(particle_weight_df, on='particle_id', how='left', sort=False)
    df.drop(columns='particle_id', inplace=True)

    # find detector layers for the hits
    df['in_cyl'] = run.hit_in_cyl[df['hit_id'].values]
    df['lay_id'] = run.hit_layer_id[df['hit_id'].values]
    # assign a compact id (unique within the event) to layer crossings
    # Note: The cross_id identifies track and layer, but may be repeated along a
    #       track if the track crosses the same layer multiple times.
    nlay_id = 1 + df['lay_id'].max()
    cross_id_gen = crossIdGenerator(ntracks, nlay_id)
    df['cross_id'] = cross_id_gen(df['track_id'], df['in_cyl'], df['lay_id'])

    # count hits per layer crossing (before track-order sorting)
    unique_cross_id, count_per_cross_id = np.unique(df['cross_id'].values, return_counts=True)
    unique_counts, count_per_count = np.unique(count_per_cross_id, return_counts=True)
    print(unique_counts, count_per_count)

    hit_to_track_id = lookupTable(df['track_id'].values, df['hit_id'].values, max_key=run.event.max_hit_id)

    nhits_per_cross_id = lookupTable(count_per_cross_id, unique_cross_id)

    return df, tracks_df, hit_to_track_id, cross_id_gen, nhits_per_cross_id

def getTrueHitCoordinateTables(run):
    assert run.event.has_truth
    hit_ids = run.event.truth_df['hit_id'].values
    hit_tx = lookupTable(run.event.truth_df['tx'].values, hit_ids, max_key=run.event.max_hit_id)
    hit_ty = lookupTable(run.event.truth_df['ty'].values, hit_ids, max_key=run.event.max_hit_id)
    hit_tz = lookupTable(run.event.truth_df['tz'].values, hit_ids, max_key=run.event.max_hit_id)
    return hit_tx, hit_ty, hit_tz

def trackOrderHits(run, df, tracks_df):
    # coordinates of the hits
    x, y, z = run.event.hitCoordinatesById(df['hit_id'].values)

    # true coordinates relative to the original vertex, inverted over the unit circle in x,y-plane
    vx = tracks_df['vx'].values[df['track_id'] - 1]
    vy = tracks_df['vy'].values[df['track_id'] - 1]
    vz = tracks_df['vz'].values[df['track_id'] - 1]
    rx = df['tx'] - vx
    ry = df['ty'] - vy
    rz = df['tz'] - vz
    r2sqr = np.square(rx) + np.square(ry)
    at_vertex = (r2sqr == 0)
    print("true hit directly at the vertex: ", np.sum(at_vertex))
    r2sqr[at_vertex] = 0.001 # dummy value to avoid division by zero
    # Note: We properly handle the at_vertex case below.
    rxinv = rx / r2sqr
    ryinv = ry / r2sqr

    # original momentum, aligned with df
    px = tracks_df['px'].values[df['track_id'].values - 1]
    py = tracks_df['py'].values[df['track_id'].values - 1]
    pz = tracks_df['pz'].values[df['track_id'].values - 1]
    p = np.sqrt(np.square(px) + np.square(py) + np.square(pz))

    # pseudo-rapidity
    eta = np.arctanh( pz / p )

    # order for central particles
    ordparam = -(rxinv * px + ryinv * py)
    ordparam[at_vertex] = -np.inf # hits at the original vertex come first

    # override order for particles in the forward regions
    forward  = (eta >  0.10)
    backward = (eta < -0.10)
    ordparam[forward ] =  z[forward ]
    ordparam[backward] = -z[backward]

    # sort hits per track_id and within the track_id in supposed track order
    df['ordparam'] = ordparam
    df.sort_values(['track_id', 'ordparam'], inplace=True)
    df.reset_index(drop=True, inplace=True)

def buildCandidatesListFromTracks(run, df, tracks_df):
    # within each track, we want to get an index for each layer crossing,
    # and within each layer crossing, an index for the hit
    # XXX move this into a sub-function?
    starts_next_track = df['track_id'].values != np.roll(df['track_id'].values, 1)
    starts_next_cross = df['cross_id'].values != np.roll(df['cross_id'].values, 1)
    assert starts_next_track[0]
    assert starts_next_cross[0]
    cross_index = np.cumsum(starts_next_cross) - 1
    first_cross_index_in_track = np.maximum.accumulate(np.where(starts_next_track, cross_index, 0))
    cross_index -= first_cross_index_in_track
    hit_index = np.arange(len(df))
    first_hit_index_in_cross = np.maximum.accumulate(np.where(starts_next_cross, hit_index, 0))
    hit_index -= first_hit_index_in_cross
    df['cross_index'] = cross_index
    df['hit_index'] = hit_index
    print(pd.crosstab(index=hit_index, columns='count'))
    # XXX check maximum pairwise distance within each layer crossing group
    # Note: We do not check all pairs but instead use the triangle inequality
    #       to get an upper bound.

    ### XXX caution: arrays will not remain aligned with df below! best move code above into a function

    # limit number of hits per layer crossing to the maximum supported by the candidates list
    nmax_per_crossing = run.candidates.nmax_per_crossing
    is_save = df['hit_index'].values < nmax_per_crossing
    if np.any(~is_save):
        print("WARNING: dropping ", np.sum(~is_save), " hit(s) with hit_index > ", nmax_per_crossing)
    df = df.loc[is_save]

    ###with pd.option_context('display.width', None, 'display.max_columns', None, 'display.max_rows', None):
    ###    print(df.head(30))

    # build candidates list
    # XXX move this into a sub-function?
    hit_row = df['track_id'].values - 1
    hit_col = nmax_per_crossing * df['cross_index'] + df['hit_index']
    max_ncross = 1 + df['cross_index'].max()
    hit_matrix_shape = (hit_row.max() + 1, max_ncross * nmax_per_crossing)
    print("maximum number of layer crossings: ", max_ncross)
    print("shape of hit matrix: ", hit_matrix_shape)
    assert np.all(hit_col < hit_matrix_shape[1])
    hit_matrix = np.zeros(hit_matrix_shape, dtype=np.int32)
    hit_matrix[hit_row, hit_col] = df['hit_id'].values

    # prepare companion data frame
    cand_df = tracks_df[['track_id', 'weight', 'particle_id', 'vx', 'vy', 'vz', 'px', 'py', 'pz', 'q', 'nhits']]

    # fill candidates list
    run.candidates.initialize(hit_matrix, cand_df)

    # as a sanity check, prepare a test submission from the perfect candidate list
    submission_df = run.candidates.submit(fill=True)
    score = score_event(run.event.truth_df, submission_df)
    print("score of retained hits = %.4f" % score)

    # tabulate weight with number of crossings
    run.candidates.df['ncross'] = run.candidates.ncross
    print("weight of tracks with ncross layer crossings:")
    print(run.candidates.df.groupby('ncross', sort=True)['weight'].sum())
    print("total weight: %.4f of which %.4f is in tracks with less than 3 crossings"
          % (run.candidates.df['weight'].sum(), run.candidates.df.loc[run.candidates.ncross < 3]['weight'].sum()))

def reportOnXGBoostModel(booster, feature_names):
    f_to_feature_map = { "f"+str(i) : name for i, name in enumerate(feature_names) }
    for importance_type in 'weight', 'gain', 'cover':
        f_scores = booster.get_score(importance_type=importance_type)
        feature_scores = pd.Series(data={ f_to_feature_map[k] : v for k, v in f_scores.items() })
        feature_scores.sort_values(ascending=False, inplace=True)
        with pd.option_context('display.max_rows', None):
            print("Feature scores, %s:" % importance_type)
            print(feature_scores)
    dump = booster.get_dump()
    print("Number of trees: ", len(dump))
    print("Trees:")
    for tree in dump:
        print(tree)

def removeOutlierCrossings(event, ana_df):
    nsr_theta = ana_df['de_theta'].abs() / ana_df['dbe_theta']
    nsr_phi   = ana_df['de_phi'  ].abs() / ana_df['dbe_phi'  ]

    nsr_theta_too_high = (nsr_theta > 3.0)
    nsr_phi_too_high   = (nsr_phi   > 3.0)

    is_outlier = nsr_theta_too_high | nsr_phi_too_high

    outliers_df = ana_df.loc[is_outlier]
    outlier_particles = np.unique(outliers_df['particle_id'].values)
    outlier_particles_weight = outliers_df.groupby('particle_id', sort=False)['weight'].first().sum()
    outlier_pct = len(outliers_df) / len(ana_df) * 100.0

    print("    removing ", len(outliers_df), " outlier(s) (%.1f%%) from " % outlier_pct,
          len(outlier_particles),
          " crazon(s) with total weight %.4f" % outlier_particles_weight)
    if 0: # XXX make an option
        print("    crazons in event ", event.event_id, ":")
        for pid in outlier_particles:
            print("        %d" % pid)

    ana_df = ana_df.loc[~is_outlier]
    return ana_df

def formatConditionCounts(conditions, weight=None):
    text = ""
    for i, (name, cond) in enumerate(conditions):
        if i > 0:
            text += ", "
        n = np.sum(cond)
        text += "%5d" % n
        if weight is not None:
            text += (" (%5.1f%% $%04.0f)"
                    % (n / len(weight) * 100 if len(weight) > 0 else 0, np.sum(weight[cond]) * 10000))
        text += " " + name
    return text

class Supervisor(Logger):
    default_params = OrderedDict([
        ('learn__true_coords'   , True    ),
        ('learn__analyze_pairs' , False   ),
    ]   + list(Algorithm.default_params.items())
    )

    def __init__(self, spec, params={}):
        """
        Args:
            spec (geometry.DetectorSpec): Defines the geometry of the detector.
            params (dict or OrderedDict): hyperparameters to override defaults.
        """
        super(Supervisor, self).__init__(max_log_indent=None)
        self.spec = spec
        self.params = self.default_params
        self.params.update(params)
        self._data = {}
        self.models = OrderedDict([
            ('d_theta_sqr', None),
            ('d_phi_sqr'  , None),
        ])

    def loadModels(self, prefix):
        for name in self.models.keys():
            filename = prefix + name + '.joblib.dat'
            self.log("loading model '%s' from file %s" % (name, filename))
            model = joblib.load(filename)
            self.models[name] = model

    def saveModels(self, prefix, models=None):
        for name, model in self.models.items():
            if models is None or name in models:
                filename = prefix + name + '.joblib.dat'
                self.log("saving model '%s' in file %s" % (name, filename))
                joblib.dump(model, filename)

    def learnLayerFunctions(self, events_train):
        with self.timed('loading hits'):
            nhits = 0
            hits_dfs = []
            nevents = len(events_train)
            min_event_id = min((event.event_id for event in events_train))
            for event in events_train:
                event.open()
                df = event.hits_df
                # get true hit positions, if available, and remember the differences to the measured coordinates
                if event.has_truth:
                    df = df.merge(event.truth_df[['hit_id', 'tx', 'ty', 'tz']], on='hit_id', how='left', sort=False)
                    df['dxnt'] = df['x'] - df['tx']
                    df['dynt'] = df['y'] - df['ty']
                    df['dznt'] = df['z'] - df['tz']
                    df.drop(columns=['tx', 'ty', 'tz'], inplace=True)
                # make hit_id unique over all events
                N = 1000000
                assert event.max_hit_id < N
                assert (event.event_id - min_event_id + 1) * N <= np.iinfo(df['hit_id'].dtype).max
                df['hit_id'] += (event.event_id - min_event_id) * N
                hits_dfs.append(df)
                nhits += event.max_hit_id
                event.close()
                print(".", end="", flush=True)
            print("\nnumber of events loaded: ", nevents)
            print("\nnumber of hits loaded: %d (%.0f on average per event)" % (nhits, nhits / nevents))
        with self.timed('concatenating'):
            hits_df = pd.concat(hits_dfs, axis=0, ignore_index=True)
            hits_df.info(memory_usage='deep')
        with self.timed('fitting neighbors'):
            neighbors = Neighbors(self.spec, params=self.params)
            neighbors.fit(hits_df)

        # set for dumping raw data for a specific layer
        dump_is_cylinder = None
        dump_layer_id = None

        layer_functions_dfs = []
        for is_cylinder in True, False:
            for layer_id in range((len(self.spec.caps), len(self.spec.cylinders))[is_cylinder]):
                with self.timed('layer (%s, %d)' % (is_cylinder, layer_id)):
                    with self.timed('calculating grid'):
                        n = 5000000 # XXX parameterize
                        # use a defined random state for reproducibility
                        random_state = np.random.RandomState(seed=layer_id)
                        if is_cylinder:
                            # random sample points for a cylinder
                            cylinder_ser = self.spec.cylinders.cylinders_df.iloc[layer_id]
                            cyl_r2 = cylinder_ser['cyl_r2']
                            z_grid = random_state.uniform(
                                low=-cylinder_ser['absz_max'],
                                high=cylinder_ser['absz_max'],
                                size=n)
                            x_norm = random_state.normal(size=n)
                            y_norm = random_state.normal(size=n)
                            r2_norm = np.sqrt(np.square(x_norm) + np.square(y_norm))
                            x_grid = x_norm / r2_norm * cyl_r2
                            y_grid = y_norm / r2_norm * cyl_r2
                        else:
                            # random sample points for a cap (uniform (in area) samples on the disc)
                            r2_grid = np.sqrt(random_state.uniform(
                                low=self.spec.caps.caps_r2_min_sqr[layer_id],
                                high=self.spec.caps.caps_r2_max_sqr[layer_id],
                                size=n))
                            phi_grid = random_state.uniform(low=0, high=2*np.pi, size=n)
                            x_grid = r2_grid * np.cos(phi_grid)
                            y_grid = r2_grid * np.sin(phi_grid)
                            cap_z = self.spec.caps.cap_z[layer_id]
                            z_grid = np.full(n, cap_z)
                        xyz_grid = np.stack([x_grid, y_grid, z_grid], axis=1)
                        r_grid = np.linalg.norm(xyz_grid, axis=1)
                    with self.timed('finding neighbors'):
                        if is_cylinder:
                            layer_neighbors = neighbors.cyln
                        else:
                            layer_neighbors = neighbors.capn
                        nb_df = layer_neighbors.findNeighborhoodK(layer_id, xyz_grid, k=1)
                        assert len(nb_df) == xyz_grid.shape[0]
                        nb_df.rename(columns={'nb_hit_id': 'hit_id'}, inplace=True)
                    with self.timed('calculating neighbor properties (%d neighbors)' % len(nb_df)):
                        nb_df = nb_df.merge(hits_df, on='hit_id', how='left', sort=False)
                        # coordinates of nearest neighbor points and differences to true positions
                        xn = nb_df['x'].values
                        yn = nb_df['y'].values
                        zn = nb_df['z'].values
                        dxnt = nb_df['dxnt'].values
                        dynt = nb_df['dynt'].values
                        dznt = nb_df['dznt'].values

                        # vertex difference between nearest neighbor and idealized intersection point
                        xd = xn - x_grid
                        yd = yn - y_grid
                        zd = zn - z_grid
                        # scale neighbor difference to values representative for a single event
                        xd *= np.sqrt(nevents)
                        yd *= np.sqrt(nevents)
                        zd *= np.sqrt(nevents)
                        # vertex difference projected onto the ray from the origin:
                        d_in_r = xd * x_grid + yd * y_grid + zd * z_grid
                        xd_radial = x_grid * d_in_r / np.square(r_grid)
                        yd_radial = y_grid * d_in_r / np.square(r_grid)
                        zd_radial = z_grid * d_in_r / np.square(r_grid)

                        x = x_grid
                        y = y_grid
                        z = z_grid
                        r2 = np.sqrt(np.square(x) + np.square(y))
                        r = np.sqrt(np.square(x) + np.square(y) + np.square(z))

                        # set up unit vectors normal to the ray from the origin
                        utheta_factor = np.abs(z) / (r2 * r)
                        utheta_x = utheta_factor * x
                        utheta_y = utheta_factor * y
                        utheta_z = -np.sign(z) * r2 / r
                        uphi_x = -y / r2
                        uphi_y =  x / r2
                        uphi_z = 0

                        # project difference vectors onto the normal unit vectors
                        d_utheta   = utheta_x * xd   + utheta_y * yd   + utheta_z * zd
                        d_uphi     = uphi_x   * xd   + uphi_y   * yd   + uphi_z   * zd
                        dnt_utheta = utheta_x * dxnt + utheta_y * dynt + utheta_z * dznt
                        dnt_uphi   = uphi_x   * dxnt + uphi_y   * dynt + uphi_z   * dznt

                        # take absolute values and group the functions
                        # XXX clean up functions names after deciding on RMS vs. mean(abs)
                        layer_functions = [
                            ('d_utheta_abs'  , np.abs(d_utheta  )),
                            ('d_uphi_abs'    , np.abs(d_uphi    )),
                            ('dnt_utheta_abs', np.abs(dnt_utheta)),
                            ('dnt_uphi_abs'  , np.abs(dnt_uphi  )),
                        ]

                        if is_cylinder:
                            indep = z
                        else:
                            indep = r2

                        layer_functions_data = []
                        bins = 200 # XXX parameterize
                        for name, values in layer_functions:
                            # Note: By storing bins, we make sure all functions will use the same bin edges
                            # XXX should we not do RMS stats here (maybe after outlier removal?)
                            binned, bins, _ = stats.binned_statistic(indep, values, statistic='mean', bins=bins)
                            bin_centers = (bins[1:] + bins[:-1]) / 2
                            layer_functions_data.append(binned)
                        layer_functions_df = pd.DataFrame(data=OrderedDict(
                            [('is_cylinder', np.full(len(bin_centers), is_cylinder, dtype=np.bool)),
                             ('layer_id'   , np.full(len(bin_centers), layer_id   , dtype=np.int8)),
                             ('indep'      , bin_centers)]
                            + [(name, data) for (name, values), data in zip(layer_functions, layer_functions_data)]
                        ))
                        layer_functions_dfs.append(layer_functions_df)

                        # XXX test interpolation error?

                        # dump data for off-line analysis
                        if is_cylinder == dump_is_cylinder and layer_id == dump_layer_id:
                            # vertex difference normal to the ray from the origin
                            xd_normal = xd - xd_radial
                            yd_normal = yd - yd_radial
                            zd_normal = zd - zd_radial
                            d_normal = np.sqrt(np.square(xd_normal) + np.square(yd_normal) + np.square(zd_normal))
                            samples_df = pd.DataFrame(data=OrderedDict([
                                ('x', x_grid), ('y', y_grid), ('z', z_grid),
                                ('xn', xn), ('yn', yn), ('zn', zn),
                                ('dxnt', dxnt), ('dynt', dynt), ('dznt', dznt),
                                ('xd_normal', xd_normal), ('yd_normal', yd_normal), ('zd_normal', zd_normal),
                                ('d_normal', d_normal),
                            ]))
                            samples_df.to_csv('samples_%s%d_df.csv' % ('cyl' if is_cylinder else 'cap', layer_id), index=False)

        layer_functions_df = pd.concat(layer_functions_dfs, axis=0, ignore_index=True)
        layer_functions_df.info(memory_usage='deep')
        layer_functions = LayerFunctions(df=layer_functions_df)
        return layer_functions

    def analyzePairs(self, algo, run, crossing, last_crossing, fit_crossing, fit2_crossing):
        mask_next = run.candidates.has([crossing])

        # get hit ids for the layer crossing and the coordinates of the first hit
        next_hit_ids = [run.candidates.hitIds(crossing, pair=i_pair, mask=mask_next)
                      for i_pair in range (run.candidates.nmax_per_crossing)]
        next_hit_ids = np.stack(next_hit_ids, axis=1)
        x, y, z = run.candidates.hitCoordinates(crossing, mask=mask_next)
        hit_id = run.candidates.hitIds(crossing, mask=mask_next)
        cyl_closer = run.hit_in_cyl[hit_id]
        layer_id = run.hit_layer_id[hit_id]

        last_x, last_y, last_z = run.candidates.hitCoordinates(last_crossing, mask=mask_next)

        # get helix params
        if isinstance(fit_crossing, tuple):
            hel_dz, hel_xm, hel_ym, hel_r, hel_pitch = fit_crossing
        else:
            hel_xm, hel_ym, hel_r, hel_pitch = tuple(
                run.candidates.getFit(fit_crossing, par, mask=mask_next).astype(np.float64)
                for par in ('hel_xm', 'hel_ym', 'hel_r', 'hel_pitch'))
            hel_dz = helixDirectionFromTwoPoints(hel_xm, hel_ym, hel_pitch,
                         last_x, last_y, last_z, x, y, z)

        # evalute neighbor functions at the locations of the neighbor hits
        e_theta, e_phi = run.neighbors.evaluateLayerFunctions(
            x, y, z, cyl_closer, layer_id,
            functions=('dnt_utheta_abs', 'dnt_uphi_abs'))

        ana_pairs_dict = OrderedDict()
        ana_pairs_dict['e_theta'] = e_theta
        ana_pairs_dict['e_phi'  ] = e_phi

        # get updated helix params taking the next hit into account
        updated_helix = tuple(run.candidates.getFit(fit2_crossing, par, mask=mask_next)
            for par in ('hel_xm', 'hel_ym', 'hel_r', 'hel_pitch'))

        dists = []
        for i_pair in range(run.candidates.nmax_per_crossing):
            # distance from predicted helix to paired hit
            pair_coords = run.event.hitCoordinatesById(next_hit_ids[:,i_pair])
            _, _, _, pair_dist = helixNearestPointDistance(
                last_x, last_y, last_z, hel_xm, hel_ym, hel_r, hel_pitch, *pair_coords)
            pair_dist[np.isnan(pair_dist)] = np.inf
            dists.append(pair_dist)
        dists = np.stack(dists, axis=1)
        dist_order = np.argsort(dists, axis=1)

        # sort distances and hits by distance to the predicted helix
        row_indices = np.arange(dists.shape[0])[:,np.newaxis]
        dists = dists[row_indices, dist_order]
        next_hit_ids_dist_order = next_hit_ids[row_indices, dist_order]
        # put back NANs as they are more handy for pandas analysis
        dists[~np.isfinite(dists)] = np.nan

        for i_pair in range(run.candidates.nmax_per_crossing):
            ana_pairs_dict['dist%d' % i_pair] = dists[:,i_pair]

            # distance from updated helix to paired hit
            pair_coords = run.event.hitCoordinatesById(next_hit_ids_dist_order[:,i_pair])
            xnu, ynu, znu, pair_dist_updated = helixNearestPointDistance(
                x, y, z, *updated_helix, *pair_coords)
            pair_d_theta, pair_d_phi = algo.projectNeighborDisplacement(*pair_coords, xnu, ynu, znu)
            ana_pairs_dict['dist_uh%d' % i_pair] = pair_dist_updated
            ana_pairs_dict['d_theta%d' % i_pair] = pair_d_theta
            ana_pairs_dict['d_phi%d'   % i_pair] = pair_d_phi

        return ana_pairs_dict

    def analyzePairsFirstLayers(self, algo, run, crossing, last_crossing):
        mask_next = run.candidates.has([crossing])

        # get hit ids for the layer crossing and the coordinates of the first hit
        x, y, z = run.candidates.hitCoordinates(crossing, mask=mask_next)
        hit_id = run.candidates.hitIds(crossing, mask=mask_next)
        cyl_closer = run.hit_in_cyl[hit_id]
        layer_id = run.hit_layer_id[hit_id]

        ana_pairs_dict = self.analyzePairs(
            algo, run, crossing=crossing, last_crossing=2, fit2_crossing=2,
            fit_crossing=2)

        ana_df = pd.DataFrame(data=OrderedDict([
            # hit location data
            ('x', x), ('y', y), ('z', z),
            #('r2', r2),
            ('cyl_closer', cyl_closer), ('layer_id', layer_id),

            # intersection data
            ('crossing'             , crossing),
        ] + list(ana_pairs_dict.items())))
        return ana_df

    def learnTracks(self, algo, events_train,
                    save_models_prefix=None,
                    save_displacements_prefix=None,
                    save_pairs_prefix=None,
                    save_analytics_path=None):
        with self.timed('learning track properties'):
            nevents = len(events_train)
            ana_dfs = []
            for event in events_train:
                event.open(use_true_coords=self.params['learn__true_coords'])
                self.log(event.summary())
                assert event.has_truth

                # setup algorithm run (including neighborhoods) for this event
                run = algo.setupRun(event, supervisor=self,
                        fit_columns=['hel_xm', 'hel_ym', 'hel_r', 'hel_pitch', 'hel_pitch_ls', 'hel_ploss',
                                     'dist', 'de_theta', 'de_phi', 'dbe_theta', 'dbe_phi'])

                # get ground truth non-noise hits and their tracks
                df, tracks_df, hit_to_track_id, cross_id_gen, nhits_per_cross_id = getTrueHitsAndTracks(run)
                hit_tx, hit_ty, hit_tz = getTrueHitCoordinateTables(run)

                # find a geometric ordering of the hits along each track
                trackOrderHits(run, df, tracks_df)

                # build a perfect candidate list from the ground truth tracks
                # Note: It is perfect only up to cases where we might get the order of the hits
                #       along the track wrong or where we dropped hits due to limitations.
                buildCandidatesListFromTracks(run, df, tracks_df)

                # iterate over the layer crossings
                max_ncross = run.candidates.ncross.max()
                per_cross_dfs = []
                wl_dfs = []
                for i in range(2, max_ncross):
                    mask = (i < run.candidates.ncross)

                    # fit tracks at crossing index i
                    algo.fitTracks(run, crossing=i, mask=mask)

                    # select tracks which have a true extension into crossing index i+1
                    mask_next = mask & run.candidates.has([i+1])
                    nnext = np.sum(mask_next)
                    self.log("crossing %d: tracks continuing: %d ($%04.0f)"
                          % (i, nnext, run.candidates.df['weight'][mask_next].sum() * 10000))

                    # get data about the true extension hit
                    hit_id = run.candidates.hitIds(i+1, mask=mask_next)
                    cyl_closer = run.hit_in_cyl[hit_id]
                    layer_id = run.hit_layer_id[hit_id]
                    use_true_hit_coords = False
                    if use_true_hit_coords:
                        x, y, z = hit_tx[hit_id], hit_ty[hit_id], hit_tz[hit_id]
                    else:
                        x, y, z = run.candidates.hitCoordinates(i+1, mask=mask_next)

                    # get helix fit parameters
                    # XXX refactor this into an algorithm method?
                    last_x, last_y, last_z = run.candidates.hitCoordinates(i, mask=mask_next)
                    hel_xm    = run.candidates.getFit(i, 'hel_xm'   , mask=mask_next).astype(np.float64)
                    hel_ym    = run.candidates.getFit(i, 'hel_ym'   , mask=mask_next).astype(np.float64)
                    hel_r     = run.candidates.getFit(i, 'hel_r'    , mask=mask_next).astype(np.float64)
                    hel_pitch = run.candidates.getFit(i, 'hel_pitch', mask=mask_next).astype(np.float64)
                    hel_dz    = helixDirectionFromTwoPoints(hel_xm, hel_ym, hel_pitch,
                                    last_x, last_y, last_z, x, y, z)

                    corrector = HelixCorrector(algo, run, mask=mask_next, crossing=i)

                    # get helix intersection neighbors found by our algorithm
                    # XXX handle skipped intersections
                    details = SimpleNamespace()
                    has_intersection, xi, yi, zi, dphi, hel_s, nb_df = algo.findHelixIntersectionNeighbors(
                        run, i-2, i-1, i, last_x, last_y, last_z,
                        k=4, nmax_per_crossing=4, mask=mask_next, corrector=corrector,
                        details=details)

                    cand_no_intersection = np.zeros(run.candidates.n, dtype=np.bool)
                    np.place(cand_no_intersection, mask_next, ~has_intersection)

                    nintersections = np.sum(has_intersection)
                    nmissing_intersections = nnext - nintersections
                    if nmissing_intersections > 0:
                        self.log("missing intersection for %d of %d tracks ($%04.0f)"
                              % (nmissing_intersections, nnext, run.candidates.df['weight'][cand_no_intersection].sum() * 10000))
                        if 0:
                            run.candidates.df.insert(0, 'i_masked', -1)
                            run.candidates.df.insert(0, 'due_to_dphi', False)
                            run.candidates.df.loc[mask_next, 'i_masked'] = np.arange(nnext)
                            run.candidates.df.loc[mask_next, 'due_to_dphi'] = details.reject_total_dphi
                            print(run.candidates.__str__(
                                mask=cand_no_intersection, show_coords=True, break_cross=True,
                                show_df=True, show_fit=True, show_r2=True,
                                hit_in_cyl=run.hit_in_cyl, hit_layer_id=run.hit_layer_id))
                            run.candidates.df.drop(columns=['i_masked', 'due_to_dphi'], inplace=True)

                    same_layer = (details.cyl_closer == cyl_closer) & (details.next_id == layer_id)
                    wrong_layer = ~has_intersection | ~same_layer

                    nwrong_layer = np.sum(wrong_layer)
                    if nwrong_layer > 0:
                        self.log("wrong intersection layers: %d of %d intersections (%.1f%%, $%04.0f)"
                            % (nwrong_layer, nintersections, nwrong_layer / nintersections * 100.0,
                            run.candidates.df['weight'][mask_next][wrong_layer].sum() * 10000))
                        if 0: # XXX make this an option for whether to collect wrong-layer stats
                            wl_df = pd.DataFrame(data=OrderedDict([
                                ('cyl'  , cyl_closer        ),
                                ('lay'  , layer_id          ),
                                ('pcyl' , details.cyl_closer),
                                ('play' , details.next_id   ),
                                ('wrong', wrong_layer       ),
                            ]))
                            wl_df = wl_df.loc[wl_df['wrong']]
                            wl_dfs.append(wl_df)

                    # Notes for event 1000 on missing intersections:
                    # 225186372279861248: hits very close to gap in inner caps
                    #     752103542952558592: same?
                    #     887212425126871040: same?
                    # 274723906596634624: hits cap 23 close to the inner rim, helix intersection is predicted beyond
                    #     the inner rim -> missing intersection
                    #     954766831854288896: same?
                    #     220679542837084160: same?
                    #     297239430832324600: same?
                    #     517927769762430976: same?
                    #     526923149267173376: same?
                    #     837673104103702528: same?
                    #     873709941301444608: same?
                    #     716074333616734208: same, but in negative z-direction?
                    # 495400700654649344: nice track with close miss of cap 1 beyond the outer rim?
                    #     585470013142466560: same for cap 17 in other direction?
                    #     752104573744709632: same for cap 19?
                    # 598981361780391936: fake gap in cap 0 at r2=~755 mm
                    #     662036223329566720: same for cap 23?
                    # 27047505006952450: very nice track. escapes through the hole between cyl 9 and cap 6?
                    #     (hits cap 6 within 5mm of the outer rim)
                    #     436849301293891584: same?
                    # 112599405269356540: looks like a huge nice helix, moving out of the detector from
                    #     the cylinders and back into the caps. not found due to too-large dphi
                    #     671055929529073664: similar?

                    # build an array with the correct hit_ids in the present crossing for
                    # each candidate
                    cand_next_hit_ids = [run.candidates.hitIds(i+1, pair=i_pair)
                                       for i_pair in range (run.candidates.nmax_per_crossing)]
                    cand_next_hit_ids = np.stack(cand_next_hit_ids, axis=1)
                    next_hit_ids = cand_next_hit_ids[mask_next]
                    next_ntrue   = np.count_nonzero(next_hit_ids, axis=1)
                    next_nmissed = next_ntrue.copy()
                    next_ngood   = np.zeros_like(next_ntrue)
                    next_nbad    = np.zeros_like(next_ntrue)
                    next_nfound  = np.zeros_like(next_ntrue)

                    bay_df = None
                    if nb_df is not None:
                        hit_cols = ['extend_hit_id'] + ['extend_hit_id_' + ascii_lowercase[i_pair]
                                    for i_pair in range (1, run.candidates.nmax_per_crossing)]

                        # index pointing back from the neighbor data frame into the candidates list
                        candidate_index = np.nonzero(mask_next)[0][nb_df['extend_index'].values]

                        nb_df['weight'] = run.candidates.df['weight'].values[candidate_index]

                        # Analytics:
                        #     ntrue......number of true particle hits in the next layer crossing
                        #     nfound.....number of hits found to extend the track
                        #     ngood......number of hits in nfound that actually belong to the true particle
                        #         XXX: within the correct layer, consider only hits in the present
                        #              layer crossing to be good?
                        #     nmissed....number of unfound hits that the true particle left on the
                        #                    layer of the intersection we found
                        # Cases:
                        # 1) intersection found on correct layer
                        #    {1 <= ngood + nmissed}
                        #     1.1) ngood >= 1, at least one correct hit found
                        #         OK...analyse mispairing
                        #     1.2) ngood == 0, no correct hit found
                        #         1.2.1) nfound >= 1, all wrong hits taken
                        #             !!!
                        #         1.2.2) nfound == 0, missed hits on this layer (we know nmissed >= 1)
                        #             ! to !!, depending on whether continuation of
                        #             track is found on following layers or not
                        # 2) intersection found on wrong layer
                        #    {0 <= ngood + nmissed}
                        #     2.1) ngood >= 1, we were lucky to find another layer of the right track
                        #         !, ...analyse mispairing
                        #     2.2) ngood == 0
                        #         2.2.1) nfound >= 1, all wrong hits taken
                        #             2.2.1.1) marked as dubious
                        #                 ! (similar to nfound == 0, but slightly worse)
                        #             2.2.1.2) not marked as dubious
                        #                 !!!
                        #         2.2.2) nfound == 0
                        #             2.2.2.1) nmissed == 0, OK, nothing to see on this layer
                        #             2.2.2.2) nmissed >= 1
                        #                 !
                        # 3) no intersection found
                        #     !!!, but seems very rare

                        # aligned with candidates:
                        #     cand_next_hit_ids
                        #     <- (sel_)candidate_index
                        # aligned with candidates[mask_next], #=nnext:
                        #     hit_id
                        #     has_intersection, same_layer, cond
                        #     details.cyl_closer, details.next_id
                        #     next_layisc_index -> candidates[mask_next][cond]
                        #     next_hit_ids
                        #     next_ntrue, next_ngood, next_nbad, next_nmissed
                        #     <- (sel_)nb_df['extend_index']
                        #     <- details.bay_index
                        # aligned with candidates[mask_next][cond], #=ncond: (continued candidates for which the intersection exists and meets cond):
                        #     layisc_weight
                        #     layisc_next_hit_ids
                        #     layisc_ntrue, layisc_ngood, layisc_nbad, layisc_nmissed
                        #     <- next_layisc_index
                        # aligned with full nb_df:
                        #     candidate_index -> candidates
                        # aligned with sel_nb_df (neighbors selected by their intersection meeting the cond condition):
                        #     sel_candidate_index -> candidates
                        #     sel_nb_df['extend_index'] -> candidates[mask_next]
                        #     sel_correct_hit_ids
                        #     nb_ngood
                        #     nb_col_has_hit, nb_col_is_correct, nb_col_is_correct: specific to the current column of sel_nb_df

                        next_layisc_index = np.zeros(nnext, dtype=np.int32)

                        for on_right_layer, cond, desc in (True, same_layer, "correct"), (False, ~same_layer, "wrong"):
                            cond &= has_intersection
                            layisc_weight = run.candidates.df['weight'][mask_next][cond]
                            cond_weight = layisc_weight.sum()
                            sel_nb_df = nb_df.loc[cond[nb_df['extend_index'].values]]
                            self.log("looking on %s layer $%04.0f:" % (desc, cond_weight * 10000))

                            # index pointing back from the neighbor data frame into the candidates list
                            sel_candidate_index = np.nonzero(mask_next)[0][sel_nb_df['extend_index'].values]

                            sel_correct_hit_ids = cand_next_hit_ids[sel_candidate_index]

                            # number and indices of continuing tracks with match the right/wrong layer condition
                            ncond = np.sum(cond)
                            next_layisc_index[cond] = np.arange(ncond, dtype=np.int32)

                            nb_ngood  = np.zeros(len(sel_nb_df), np.int8)
                            nb_nbad   = np.zeros(len(sel_nb_df), np.int8)
                            nb_nfound = np.zeros(len(sel_nb_df), np.int8)
                            for i_pair, hit_col in enumerate(hit_cols):
                                found_hit_id = sel_nb_df[hit_col].values
                                nb_col_has_hit = (found_hit_id != 0)

                                if on_right_layer:
                                    same_hit_id = (found_hit_id[:,np.newaxis] == sel_correct_hit_ids)
                                    nb_col_ngood = np.sum(same_hit_id, axis=1) * nb_col_has_hit
                                    assert np.all(nb_col_ngood <= 1)
                                    nb_col_is_correct = (nb_col_ngood > 0)
                                    del nb_col_ngood
                                else:
                                    nb_col_is_correct = (hit_to_track_id[found_hit_id]
                                        == run.candidates.df['track_id'].values[sel_candidate_index])
                                nb_ngood  += nb_col_is_correct
                                nb_nbad   += nb_col_has_hit & ~nb_col_is_correct
                                nb_nfound += nb_col_has_hit

                                nfound = np.sum(nb_col_has_hit)
                                ngood =  np.sum(nb_col_has_hit &  nb_col_is_correct)
                                nbad  =  np.sum(nb_col_has_hit & ~nb_col_is_correct)
                                ndub  =  np.sum(nb_col_has_hit & ~nb_col_is_correct & sel_nb_df['dubious'])
                                if on_right_layer:
                                    self.log("        %15s: %5d ext" % (hit_col, nfound), end='')
                                    if nfound > 0:
                                        self.log(" %5d (%5.1f%%) good, %3d (%4.1f%%) bad [%3d (%4.1f%%) dub]"
                                              % (ngood, ngood / nfound * 100.0, nbad, nbad / nfound * 100.0,
                                               ndub, (ndub / nbad * 100.0 if nbad > 0 else 0)), end='')
                                    self.log("")

                            nb_next_index = sel_nb_df['extend_index'].values
                            if on_right_layer:
                                next_nmissed[nb_next_index] -= nb_ngood
                                next_ngood  [nb_next_index] += nb_ngood
                                next_nbad   [nb_next_index] += nb_nbad
                                next_nfound [nb_next_index] += nb_nfound
                                layisc_ntrue   = next_ntrue  [cond]
                                layisc_nmissed = next_nmissed[cond]
                                layisc_ngood   = next_ngood  [cond]
                                layisc_nbad    = next_nbad   [cond]
                                layisc_nfound  = next_nfound [cond]
                            else:
                                layisc_track_id = run.candidates.df['track_id'].values[mask_next][cond]
                                layisc_cyl_closer = details.cyl_closer[cond]
                                layisc_next_id    = details.next_id[cond]
                                layisc_cross_id = cross_id_gen(layisc_track_id, layisc_cyl_closer, layisc_next_id)
                                layisc_ntrue = nhits_per_cross_id[layisc_cross_id]
                                layisc_ngood   = np.zeros(ncond, dtype=np.int32)
                                layisc_nbad    = np.zeros(ncond, dtype=np.int32)
                                layisc_nfound  = np.zeros(ncond, dtype=np.int32)
                                layisc_ngood  [next_layisc_index[nb_next_index]] = nb_ngood
                                layisc_nbad   [next_layisc_index[nb_next_index]] = nb_nbad
                                layisc_nfound [next_layisc_index[nb_next_index]] = nb_nfound
                                self.log("layisc_ntrue ", layisc_ntrue.shape, " layisc_ngood", layisc_ngood.shape)
                                layisc_nmissed = layisc_ntrue - layisc_ngood

                            self.log("    " + formatConditionCounts(
                                [('miss%d' % n, (layisc_nmissed == n) & (layisc_nmissed < layisc_ntrue))
                                                for n in range(run.candidates.nmax_per_crossing)]
                                + [('missall', (layisc_nmissed == layisc_ntrue))],
                                weight=layisc_weight))

                            self.log("    " + formatConditionCounts(
                                [('part%d' % n, (layisc_ngood == n) & (layisc_ngood < layisc_ntrue))
                                                for n in range(1, run.candidates.nmax_per_crossing)],
                                weight=layisc_weight))

                            self.log("    " + formatConditionCounts(
                                [('bad%d' % n, (layisc_nfound >= 1) & (layisc_nbad == n) & (layisc_nbad < layisc_nfound))
                                                for n in range(run.candidates.nmax_per_crossing)]
                                + [('allbad', (layisc_nfound >= 1) & (layisc_nbad == layisc_nfound))],
                                weight=layisc_weight))

                            self.log("    " + formatConditionCounts(
                                [('put%d' % n, (layisc_nfound == n))
                                                for n in range(1 + run.candidates.nmax_per_crossing)]
                                + [('putany', (layisc_nfound > 0))],
                                weight=layisc_weight))

                            if on_right_layer:
                                bay_df = pd.DataFrame(data=OrderedDict([
                                    ('bay_index'    , details.bay_index    ),
                                    ('bay_cut'      , details.bay_cut      ),
                                    ('bay_de'       , details.bay_de       ),
                                    ('bay_e_theta'  , details.bay_e_theta  ),
                                    ('bay_e_phi'    , details.bay_e_phi    ),
                                    ('bay_b_theta'  , details.bay_b_theta  ),
                                    ('bay_b_phi'    , details.bay_b_phi    ),
                                    ('bay_d_theta'  , details.bay_d_theta  ),
                                    ('bay_d_phi'    , details.bay_d_phi    ),
                                ]))

                                # Note about the details.bay_ arrays:
                                #   * There can be 0 or more entries corresponding to one candidates[mask_next]
                                #         (pointed to by details.bay_index).
                                #   * There can be 0 or more entries corresponding to an individual hit_id.
                                #   * BUT: For each candidates[mask_next] there can be at most 1 entry that
                                #     matches both in details.bay_index and in details.bay_hit_id for a
                                #     particular hit_id column.
                                #   * For each candidates[mask_next], there can be 0 to nmax_per_crossing
                                #     entries which have at least one correct hit_id

                                bay_right_hit_ids = (details.bay_hit_id[:,np.newaxis] == next_hit_ids[details.bay_index])
                                bay_any_right_hit_id = np.any(bay_right_hit_ids, axis=1)

                                bay_df = bay_df.loc[bay_any_right_hit_id]

                                # for each bay_index, select the first entry with right hit ids
                                unique_bay_index, bay_df_index = np.unique(bay_df['bay_index'].values, return_index=True)
                                bay_df = bay_df.iloc[bay_df_index].reset_index(drop=True)

                            nextended = len(sel_nb_df)
                            extended_weight = sel_nb_df['weight'].sum()
                            if ncond > 0:
                                nempty = ncond - nextended
                                empty_weight = cond_weight - extended_weight
                                self.log("    total: %5d ext of %5d (%5.1f%%, %5.1f%% $%04.0f of %5d), %3d (%5.1f%%, %5.1f%% $%04.0f of %5d) empty;"
                                    % (nextended, ncond,
                                       nextended / ncond * 100.0, nextended / nnext * 100.0,
                                       extended_weight * 10000, nnext,
                                       nempty, nempty / ncond * 100.0, nempty / nnext * 100.0,
                                       empty_weight * 10000, nnext))
                                if nextended > 0:
                                    is_any_correct = (nb_ngood > 0)
                                    nany_correct = np.sum(is_any_correct)
                                    nall_bad = nextended - nany_correct
                                    any_correct_weight = sel_nb_df['weight'][is_any_correct].sum()
                                    all_bad_weight = extended_weight - any_correct_weight
                                    self.log(" %5d (%5.1f%%) any good, %5d (%5.1f%% $%04.0f) all bad"
                                        % (nany_correct, nany_correct / nextended * 100.0,
                                           nall_bad, nall_bad / nextended * 100.0, all_bad_weight * 10000), end='')
                                if not on_right_layer and nextended > 0:
                                    ndubious = np.sum(sel_nb_df['dubious'])
                                    self.log(" %5d (%5.1f%%) dubious"
                                        % (ndubious, ndubious / nextended * 100.0), end='')
                                self.log("")
                            else:
                                self.log("    no true track proceeds; %5d spurious extensions found" % nextended)
                        else:
                            if ncond > 0:
                                self.log("    %d true tracks proceed, but no neighbors found" % ncond)


                    # get updated helix parameter predictions from the corrector
                    if corrector is not None:
                        hel_dz, hel_xm, hel_ym, hel_r, hel_pitch = corrector.helixParams()

                    # calculate distance of the true extension hit to the fitted helix
                    (xn, yn, zn, dist) = helixNearestPointDistance(
                        last_x, last_y, last_z, hel_xm, hel_ym, hel_r, hel_pitch, x, y, z)

                    # project neighbor displacements to spherical coordinate directions
                    d_theta, d_phi = algo.projectNeighborDisplacement(x, y, z, xn, yn, zn)

                    # evalute neighbor functions at the locations of the neighbor hits
                    ld_utheta_abs, ld_uphi_abs, ldnt_utheta_abs, ldnt_uphi_abs = run.neighbors.evaluateLayerFunctions(
                        x, y, z, cyl_closer, layer_id,
                        functions=('d_utheta_abs', 'd_uphi_abs', 'dnt_utheta_abs', 'dnt_uphi_abs'))

                    e_theta = ldnt_utheta_abs
                    e_phi   = ldnt_uphi_abs

                    # convert neighbor displacement and background hit distances
                    # to multiples of the respective error estimates
                    de_theta  = d_theta       / e_theta
                    de_phi    = d_phi         / e_phi
                    dbe_theta = ld_utheta_abs / e_theta
                    dbe_phi   = ld_uphi_abs   / e_phi

                    run.candidates.setFit(i+1, 'dist'     , dist     , mask=mask_next)
                    run.candidates.setFit(i+1, 'de_theta' , de_theta , mask=mask_next)
                    run.candidates.setFit(i+1, 'de_phi'   , de_phi   , mask=mask_next)
                    run.candidates.setFit(i+1, 'dbe_theta', dbe_theta, mask=mask_next)
                    run.candidates.setFit(i+1, 'dbe_phi'  , dbe_phi  , mask=mask_next)

                    # analyze distances of paired hits from the predicted and the updated helix
                    ana_pairs_dict = OrderedDict()
                    if self.params['learn__analyze_pairs']:
                        # fit tracks to get updated helix params taking the next hit into account
                        # XXX reuse track fits in next iteration?
                        algo.fitTracks(run, crossing=i+1, mask=mask_next)

                        ana_pairs_dict = self.analyzePairs(
                            algo, run, crossing=i+1, last_crossing=i, fit2_crossing=i+1,
                            fit_crossing=(hel_dz, hel_xm, hel_ym, hel_r, hel_pitch))

                    # look up ground truth attributes for off-line analysis
                    track_id = run.candidates.df.loc[mask_next]['track_id'].values
                    particle_id = tracks_df['particle_id'].values[track_id - 1]
                    particle_q = tracks_df['q'].values[track_id - 1]
                    weight = tracks_df['weight'].values[track_id - 1]

                    # calculate features
                    r2 = np.sqrt(np.square(x) + np.square(y))
                    phi = np.arctan2(y, x)

                    # a variant of the helix pitch the sign of which correlates with
                    # the sign of the charge of the particle
                    hel_qpitch = np.sign(hel_dz) * hel_pitch

                    hel_phi = np.arctan2(y - hel_ym, x - hel_xm) - phi
                    hel_phi[hel_phi < 0      ] += 2*np.pi
                    hel_phi[hel_phi > 2*np.pi] -= 2*np.pi

                    # build data frame for later model fitting and analysis
                    # aligned with candidates[mask_next]
                    ana_df = pd.DataFrame(data=OrderedDict([
                        # hit location data
                        ('x', x), ('y', y), ('z', z),
                        #('r2', r2),
                        ('cyl_closer', cyl_closer), ('layer_id', layer_id),

                        # intersection data
                        ('crossing'             , i),
                        #('algo_has_intersection', has_intersection),
                        #('algo_cyl_closer'      , details.cyl_closer),
                        #('algo_next_id'         , details.next_id),
                        #('xi', xi), ('yi', yi), ('zi', zi),
                        #('hel_s', hel_s),

                        # helix data
                        ('xn', xn), ('yn', yn), ('zn', zn),
                        ('last_x', last_x), ('last_y', last_y),
                        ('last_z', last_z),
                        #('hel_phi', hel_phi),
                        ('hel_r', hel_r),
                        ('hel_pitch', hel_pitch),
                        #('hel_qpitch', hel_qpitch),

                        # prediction error data
                        #('d_theta', d_theta), ('d_phi', d_phi),
                        ('de_theta', de_theta), ('de_phi', de_phi),
                        ('dbe_theta', dbe_theta), ('dbe_phi', dbe_phi),

                        # neighbor analysis data
                        ('ngood', next_ngood),
                        ('ntrue', next_ntrue),

                        # particle ground truth
                        #('event_id', run.event.event_id),
                        ('particle_id', particle_id),
                        ('particle_q', particle_q),
                        ('weight', weight),

                        # link with crossing / intersection data
                        #('bay_index', np.arange(len(x))),
                    ] + list(ana_pairs_dict.items())))

                    #if bay_df is not None:
                    #    ana_df = ana_df.merge(bay_df, on='bay_index', how='left', sort=False)

                    per_cross_dfs.append(ana_df)

                if self.params['learn__analyze_pairs']:
                    per_cross_dfs.append(self.analyzePairsFirstLayers(algo, run, crossing=1, last_crossing=2))
                    per_cross_dfs.append(self.analyzePairsFirstLayers(algo, run, crossing=0, last_crossing=1))

                if wl_dfs:
                    wl_df = pd.concat(wl_dfs, ignore_index=True)
                    with pd.option_context('display.max_rows', None, 'display.max_columns', None,
                                           'display.width', None):
                        self.log(pd.crosstab(index=[wl_df['cyl'], wl_df['lay']],
                                          columns=[wl_df['pcyl'], wl_df['play']]))

                # assemble data frame with all layer crossings for this event
                ana_df = pd.concat(per_cross_dfs, ignore_index=True)
                self.log("    event %4d: number of layer crossings: %d" % (event.event_id, len(ana_df)))

                # remove crazy outliers
                if 1: # XXX remove or don't remove outliers depending on what we are learning
                    ana_df = removeOutlierCrossings(event, ana_df)

                # keep the data from this event that we need for model fitting, close the rest
                ana_dfs.append(ana_df)
                event.close()

            # assemble data frame containing data from all events in the training set
            ana_df = pd.concat(ana_dfs, ignore_index=True)
            ana_df.info(memory_usage='deep')

            if self.params['learn__analyze_pairs']:
                with self.timed('learning pairing functions'):
                    self.learnPairing(ana_df, save_pairs_prefix=save_pairs_prefix)
            else:
                with self.timed('learning helix displacements'):
                    self.learnIntersectionDisplacements(ana_df,
                        save_displacements_prefix=save_displacements_prefix)

                model_defs = OrderedDict([
                    #('d_theta_sqr', {
                    #    'target': 'bay_d_theta_sqr',
                    #    'target_fn': lambda ana_df: np.square(ana_df['bay_d_theta']),
                    #}),
                    #('d_phi_sqr', {
                    #    'target': 'bay_d_phi_sqr',
                    #    'target_fn': lambda ana_df: np.square(ana_df['bay_d_phi']),
                    #}),
                ])

                for model_name, model_def in model_defs.items():
                    model_type = 'nnb'
                    #model_type = 'bdt'
                    predict_features = [
                        'z', 'r2', #'cyl_closer', # location of the hit to predict # XXX calculate intersections?
                        'hel_qpitch',            # parameters of predicted helix (uncorrected)
                    ]
                    predict_target = model_def['target']
                    if 'target_fn' in model_def:
                        ana_df[predict_target] = model_def['target_fn'](ana_df)

                    steps = []
                    fit_params = {}
                    if model_type == 'bdt':
                        steps.append(('bdt', XGBRegressor(silent=False, max_depth=5, n_estimators=200, learning_rate=0.2)))
                    elif model_type == 'nnb':
                        steps.append(('nnb', KNeighborsRegressor(n_neighbors=10)))

                    pipeline = Pipeline(steps)

                    valid_data = np.isfinite(ana_df[predict_target])
                    learn_df = ana_df.loc[valid_data]

                    features = np.stack([learn_df[col].values.astype(np.float64) for col in predict_features], axis=1)
                    target = learn_df[predict_target].values.astype(np.float64)

                    if model_type == 'bdt':
                        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.33, random_state=1)

                        self.log("RMS y_train = ", np.sqrt(np.mean(np.square(y_train))))
                        self.log("RMS y_test  = ", np.sqrt(np.mean(np.square(y_test ))))

                        fit_params.update({
                            'bdt__eval_set': [(X_train, y_train), (X_test, y_test)],
                            'bdt__early_stopping_rounds': 10,
                            'bdt__verbose': True,
                        })
                    else:
                        X_train, y_train = features, target

                    with self.timed('fitting model'):
                        pipeline.fit(X_train, y_train, **fit_params)

                    # store the trained model
                    self.models[model_name] = pipeline
                    if save_models_prefix is not None:
                        with self.timed('saving model'):
                            self.saveModels(prefix=save_models_prefix, models=[model_name])

                    if model_type == 'bdt':
                        reportOnXGBoostModel(pipeline.named_steps.bdt.get_booster(), feature_names=predict_features)

                    with self.timed('predicting'):
                        self.log("predicting %d instances" % features.shape[0])
                        prediction = pipeline.predict(features)

                    full_prediction = np.full(len(ana_df), np.nan)
                    np.place(full_prediction, valid_data, prediction)
                    ana_df['predict_' + predict_target] = full_prediction
                    self.log("RMS prediction  = ", np.sqrt(np.mean(np.square(prediction))))

            if save_analytics_path is not None:
                with self.timed('writing CSV'):
                    ana_df.to_csv(save_analytics_path, index=False)

    def learnLayerFunctionsGeneric(self, ana_df, fun, dfname=None, save_prefix=None, **kw_args):
        """Learn some interpolated functions per detector layer.
        """
        layer_dfs = []
        for is_cylinder in True, False:
            for layer_id in range((len(self.spec.caps), len(self.spec.cylinders))[is_cylinder]):
                with self.timed('layer (%s, %d)' % (is_cylinder, layer_id)):
                    layer_df = fun(ana_df, is_cylinder, layer_id, **kw_args)
                    if layer_df is not None:
                        layer_dfs.append(layer_df)
        df = pd.concat(layer_dfs, ignore_index=True)
        df.info(memory_usage='deep')
        if save_prefix is not None:
            layer_functions = LayerFunctions(df=df, dfname=dfname)
            layer_functions.to_csv(save_prefix)

    def learnPairing(self, ana_df, save_pairs_prefix=None):
        """Learn systematic intersection displacements on all layers.
        """
        self.learnLayerFunctionsGeneric(ana_df, self.learnPairingLayer,
                                        dfname='pair', save_prefix=save_pairs_prefix)

    def learnIntersectionDisplacements(self, ana_df, save_displacements_prefix=None):
        """Learn systematic intersection displacements on all layers.
        """
        self.learnLayerFunctionsGeneric(ana_df, self.learnIntersectionDisplacementsLayer,
                                        dfname='dp', save_prefix=save_displacements_prefix)

    def aggregatePairingDistance(self, x):
        if len(x) > 0:
            return np.percentile(x, 99.0)
        else:
            return 0.0

    def learnPairingLayer(self, ana_df, is_cylinder, layer_id):
        """Learn functions for deciding on hit pairing for a given layer.
        """
        self.log(("cyl_id " if is_cylinder else "cap_id "), layer_id)

        ### fetch the data

        # select data for intersections on the given layer
        df = ana_df.loc[(ana_df['cyl_closer'] == is_cylinder) & (ana_df['layer_id'] == layer_id)]

        # select crossings which have at least one paired hit
        df = df.loc[df['d_theta1'].notnull()]

        if len(df) == 0:
            return None

        # hit position
        x = df['x'].values
        y = df['y'].values
        z = df['z'].values

        # absolute pair distances
        d_theta_abs = np.abs(df['d_theta1'].values)
        d_phi_abs   = np.abs(df['d_phi1'  ].values)

        ### preprocessing

        # polar coordinates of hit position in x,y-plane
        r2 = np.sqrt(np.square(x) + np.square(y))
        phi = np.arctan2(y, x)

        ### the grid

        # create bin edges for binned stats
        nbins_phi = 50
        nbins_r2  = 50
        nbins_z   = 50
        phi_bin_edges = np.linspace(-np.pi, np.pi, num=(1 + nbins_phi))
        if is_cylinder:
            z_min = min(np.amin(z), -self.spec.cylinders.cyl_absz_max[layer_id] - 25.0)
            z_max = max(np.amax(z),  self.spec.cylinders.cyl_absz_max[layer_id] + 25.0)
            z_bin_edges = np.linspace(z_min, z_max, num=(1 + nbins_z))
            bin_edges = z_bin_edges
            map_sample_coords = [z]
            cyl_r2 = self.spec.cylinders.cylinders_df.iloc[layer_id]['cyl_r2']
        else:
            r2_min = 0
            r2_max = max(np.amax(r2), self.spec.caps.caps_r2_max[layer_id]) + 25.0
            r2_bin_edges = np.linspace(r2_min, r2_max, num=(1 + nbins_r2))
            bin_edges = r2_bin_edges
            map_sample_coords = [r2]

        ### learn functions

        binned      , _, _ = stats.binned_statistic(*map_sample_coords, d_theta_abs, 'count', bins=bin_edges)
        binned_theta, _, _ = stats.binned_statistic(*map_sample_coords, d_theta_abs,
            self.aggregatePairingDistance, bins=bin_edges)
        binned_phi  , _, _ = stats.binned_statistic(*map_sample_coords, d_phi_abs,
            self.aggregatePairingDistance, bins=bin_edges)

        # bin center coordinates
        co0_centers = (bin_edges[1:] + bin_edges[:-1])/2

        ### build dataframe for this layer

        # build the part of the layer_functions data frame for this cluster of samples
        n = len(co0_centers)
        layer_df = pd.DataFrame(data=OrderedDict([
            ('is_cylinder', np.full(n, is_cylinder, dtype=np.bool)),
            ('layer_id'   , np.full(n, layer_id, dtype=np.int8)),
            ('indep0'     , co0_centers),
            ('pair_theta' , binned_theta),
            ('pair_phi'   , binned_phi),
        ]))
        return layer_df

    def averageDisplacement(self, nbr_data):
        """Extract the average (hopefully systematic) displacement from the noisy sample displacements.
        """
        d_mean = np.mean(nbr_data, axis=1)
        d_std = np.std(nbr_data, axis=1)
        d_z = np.abs(nbr_data - d_mean[:,np.newaxis]) / d_std[:,np.newaxis]
        is_outlier = d_z > 2 # XXX refine and/or make param
        d = nbr_data.copy()
        d[is_outlier] = 0
        d_n = np.sum(~is_outlier, axis=1)
        d_avg = np.sum(d, axis=1) / d_n
        return d_avg

    def learnIntersectionDisplacementsLayer(self, ana_df, is_cylinder, layer_id):
        """Learn systematic intersection displacements on a specific layer.
        """
        self.log(("cyl_id " if is_cylinder else "cap_id "), layer_id)

        ### fetch the data

        # select data for intersections on the given layer
        df = ana_df.loc[(ana_df['cyl_closer'] == is_cylinder) & (ana_df['layer_id'] == layer_id)]

        # hit position and difference to nearest predicted helix point
        x = df['x'].values
        y = df['y'].values
        z = df['z'].values
        dx = x - df['xn'].values
        dy = y - df['yn'].values
        dz = z - df['zn'].values

        # track properties used as additional dimensions for mapping displacements
        # Note: Currently we only use coordinates of the previous hit here.
        #       (z-coordinate for caps, cylinder radius r2 for cylinders)
        if is_cylinder:
            last_coord = np.sqrt(np.square(df['last_x'].values) + np.square(df['last_y'].values))
        else:
            last_coord = df['last_z'].values

        # particle charge
        q = df['particle_q'].values

        # absolute helix pitch and curvature radius
        hel_p_abs = np.abs(df['hel_pitch'].values / (2 * np.pi))
        hel_r = df['hel_r'].values
        hel_cr = (np.square(hel_r) + np.square(hel_p_abs)) / hel_r

        ### preprocessing

        # polar coordinates of hit position in x,y-plane
        r2 = np.sqrt(np.square(x) + np.square(y))
        phi = np.arctan2(y, x)

        # unit vectors in (cylinder) radial and azimuthal directions
        ur2_x = x / r2
        ur2_y = y / r2
        uphi_x = -ur2_y
        uphi_y = ur2_x

        # project displacements to radial and azimuthal directions
        dur2  = ur2_x  * dx + ur2_y  * dy
        duphi = uphi_x * dx + uphi_y * dy

        # factor out particle charge and helix curvature/pitch
        if is_cylinder:
            particle_factor = q * hel_cr
            dp0 = dz    * particle_factor
            dp1 = duphi * particle_factor
        else:
            particle_factor = q * hel_p_abs
            dp0 = dur2  * particle_factor
            dp1 = duphi * particle_factor

        ### build the grid

        # set up the grid. it must be the same for all clusters that we form later
        # Note: For the bins, we actually set up the bin *edges* here (therefore
        #       (1 + nbins) of them). The bin centers which will actually form the
        #       interpolation grid later will be calculated further below.
        nbins_phi = 100
        nbins_r2  = 100
        nbins_z   = 100
        phi_bin_edges = np.linspace(-np.pi, np.pi, num=(1 + nbins_phi))
        if is_cylinder:
            z_min = min(np.amin(z), -self.spec.cylinders.cyl_absz_max[layer_id] - 25.0)
            z_max = max(np.amax(z),  self.spec.cylinders.cyl_absz_max[layer_id] + 25.0)
            z_bin_edges = np.linspace(z_min, z_max, num=(1 + nbins_z))
            bin_edges = (z_bin_edges, phi_bin_edges)
            map_sample_coords = [z, phi]
            cartesian_sample_coords = np.stack([x, y, z], axis=1)
            cyl_r2 = self.spec.cylinders.cylinders_df.iloc[layer_id]['cyl_r2']
        else:
            r2_min = 0
            r2_max = max(np.amax(r2), self.spec.caps.caps_r2_max[layer_id]) + 25.0
            r2_bin_edges = np.linspace(r2_min, r2_max, num=(1 + nbins_r2))
            bin_edges = (phi_bin_edges, r2_bin_edges)
            map_sample_coords = [phi, r2]
            cartesian_sample_coords = np.stack([x, y], axis=1)

        # cluster last_coord values into a few means (which will correspond to the grid
        # points of the interpolated functions in one dimension)
        kmeans = KMeans(n_clusters=3, n_init=1, random_state=1)
        kmeans.fit(last_coord.reshape(-1, 1))
        last_coord_centers = kmeans.cluster_centers_.flatten()
        last_coord_order = np.argsort(last_coord_centers)
        last_coord_inv_order = np.argsort(last_coord_order)
        last_coord_means = last_coord_centers[last_coord_order]
        last_coord_labels = last_coord_inv_order[kmeans.labels_]

        # get an idea of the scale of cluster-to-cluster distances
        last_coord_std = np.std(last_coord_means)
        self.log("    last_coord_means: ", last_coord_means, " last_coord_std: ", last_coord_std, " samples: ", len(last_coord))

        ### learn values on the grid

        used = np.zeros(len(x), dtype=np.bool)
        cluster_dfs = []
        for i_cluster, last_coord_mean in enumerate(last_coord_means):
            # select samples in this cluster
            # selection = (np.abs(last_coord - last_coord_mean) / last_coord_std) < 0.1
            selection = (last_coord_labels == i_cluster)
            n = np.sum(selection)

            # mask arrays for the selected samples
            cartesian_sample_coords_sel, map0_sel, map1_sel, dp0_sel, dp1_sel, last_coord_sel = (
                ar[selection] for ar in (cartesian_sample_coords, *map_sample_coords, dp0, dp1, last_coord))

            self.log("    %s %7.1f last_coord_mean %7.1f selected %5d samples, min/mean/max, std = %.1f/%.1f/%.1f, %.1f"
                % ("cyl_r2" if is_cylinder else "cap_cz",
                   cyl_r2 if is_cylinder else self.spec.caps.caps_df.iloc[layer_id]['cap_cz'],
                   last_coord_mean, n,
                   last_coord_sel.min(), np.mean(last_coord_sel), last_coord_sel.max(), np.std(last_coord_sel)))

            if n < 20:
                self.log("        too few samples, skipping this cluster")
                continue

            # mark selected samples as used
            used |= selection

            # get binned statistics of the displacements
            # XXX Currently these binned stats are not really used, but they are replaced by
            #     the k-neighbors regression done below.
            #     If we want to activate the binned stats again in order to get better
            #     stats for very dense regions, the 'mean' aggregating function should
            #     be replaced with self.averageDisplacement (possibly wrapped).
            binned     = stats.binned_statistic_2d(map0_sel, map1_sel, dp0_sel, 'count', bins=bin_edges)
            binned_dp0 = stats.binned_statistic_2d(map0_sel, map1_sel, dp0_sel, 'mean' , bins=bin_edges)
            binned_dp1 = stats.binned_statistic_2d(map0_sel, map1_sel, dp1_sel, 'mean' , bins=bin_edges)

            # bin center coordinates
            co0_centers = (binned.x_edge[1:] + binned.x_edge[:-1])/2
            co1_centers = (binned.y_edge[1:] + binned.y_edge[:-1])/2

            # expand bin center coordinates into a full regular mesh
            points = np.meshgrid(co0_centers, co1_centers, indexing='ij')
            points = [coord.flatten() for coord in points]

            # minimum number of values we want to aggregate in a bin
            # Note: The idea is that if we have fewer values than these, we use k-neighbors
            #       regression to get a more representative sample.
            min_nvalues = 100
            # XXX Currently we set the counts to zero so all the learning is done
            #     by the k-neighbors regression below. See also XXX comment above.
            ##counts = binned.statistic
            counts = np.zeros_like(binned.statistic)
            where_too_few = np.where(counts < min_nvalues)
            self.log("    %6d of %6d grid points have too few values" % (len(where_too_few[0]), np.size(counts)))

            # use k-neighbors regression to learn the systematic displacements
            # on this layer
            radius = np.zeros_like(binned_dp0.statistic)
            if len(where_too_few[0]) > 0:
                # find nearest neighbors of each grid point for which we want to calculate values
                nbrs = NearestNeighbors(n_neighbors=min(n, min_nvalues))
                nbrs.fit(cartesian_sample_coords_sel)
                if is_cylinder:
                    z_where   = co0_centers[where_too_few[0]]
                    phi_where = co1_centers[where_too_few[1]]
                    x_where   = cyl_r2 * np.cos(phi_where)
                    y_where   = cyl_r2 * np.sin(phi_where)
                    cartesian_grid_coords = np.stack([x_where, y_where, z_where], axis=1)
                else:
                    phi_where = co0_centers[where_too_few[0]]
                    r2_where  = co1_centers[where_too_few[1]]
                    cartesian_grid_coords = r2_where[:,np.newaxis] * np.stack([np.cos(phi_where), np.sin(phi_where)], axis=1)
                nbr_dist, nbr_index = nbrs.kneighbors(cartesian_grid_coords)
                # get radius of farthest of the neighbors for off-line analysis
                radius[where_too_few[0], where_too_few[1]] = np.amax(nbr_dist, axis=1)
                # use neighbors to learn function values for the grid points
                for data, dst in zip((dp0_sel, dp1_sel), (binned_dp0.statistic, binned_dp1.statistic)):
                    nbr_data = data[nbr_index]
                    pred = self.averageDisplacement(nbr_data)
                    dst[where_too_few[0], where_too_few[1]] = pred

            # build the part of the layer_functions data frame for this cluster of samples
            n = len(co0_centers) * len(co1_centers)
            cluster_df = pd.DataFrame(data=OrderedDict([
                ('is_cylinder', np.full(n, is_cylinder, dtype=np.bool)),
                ('layer_id'   , np.full(n, layer_id, dtype=np.int8)),
                ('indep0'     , last_coord_mean),
                ('indep1'     , points[0]),
                ('indep2'     , points[1]),
                ('dp0'        , binned_dp0.statistic.flatten()),
                ('dp1'        , binned_dp1.statistic.flatten()),
                ('radius'     , radius.flatten()),
            ]))
            cluster_dfs.append(cluster_df)

        self.log("    %d of %d samples not used" % (np.sum(~used), len(used)))

        if cluster_dfs:
            layer_df = pd.concat(cluster_dfs, ignore_index=True)
        else:
            layer_df = None
        return layer_df
