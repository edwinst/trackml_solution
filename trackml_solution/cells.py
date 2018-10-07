"""Extracting features from the cell data.

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

class CellFeatures:
    """This class calculates and provides features derived from the
    cell data of the event.
    """
    def __init__(self, algo, run, details=None):
        """Create cell features data for the given algorithm run.
        Args:
            algo (Algorithm): main algorithm object
            run (Algorithm.Run): data and parameters for this algorithm run
            details (None or SimpleNamespace): If given, this is filled with some
                detailed data useful in off-line analyis.
        """
        self._algo = algo
        self._run = run
        with self._algo.timed("calculating cell features"):
            self._cell_features = self._calculateCellFeatures(details=details)

    def cellFeaturesByHitId(self, hit_id):
        """Return cell features for the given hit_ids.
        Args:
            hit_id (np.int32 array(N,)): hit_ids for which to get cell features
        Returns:
            a tuple containing the cell feature arrays:
            Note: All of these arrays are aligned with the given hit_ids.
            [0], udir_xyz_up (array (len(hit_id), 3)):
                    unit vector in (x,y,z) coordinates in the tangent direction
                    implied by the cell features (variant with positive w-component).
            [1], udir_xyz_down (array (len(hit_id), 3)):
                    unit vector in (x,y,z) coordinates in the tangent direction
                    implied by the cell features (variant with negative w-component).
            [2], ncells (int array (len(hit_id),)):
                    number of activated cells in the detection of the respective hit.
        """
        return tuple(feature[hit_id] for feature in self._cell_features)

    def estimateClosestDirection(self, hit_id, udir):
        """Find directions implied by the cells data such that the directions are
        as close as possible to the given ones and give an estimate for the deviation.
        Args:
            hit_id (np.int32 array(N,)): hit_ids for which to look up directions
            udir (np.float64 array(N,3)): unit vectors defining the expected direction
                for each hit_id.
                Note: The selection of the closest direction also works if udir has
                      a non-zero length different from 1. In such cases, consider
                      that the returned `inner` scales linearly with `udir` and
                      that the returned `d` only makes sense if ||udir|| == 1.
        Returns:
            udir_cell (np.float64 array(N,3)): unit vector in the direction implied
                by the cells data
            inner (np.float64 array(N,)): inner product of the given `udir` direction
                and the direction implied by the cell data.
            d (np.float64 array(N,)): a measure for the difference between the two
                directions. The larger `d`, the larger and more significant the
                difference between the directions.
            XXX should probably also return angle of incidence in order to get a better
                idea about the accuracy of the cells features
        """
        udir_cell_up   = self._cell_features[0][hit_id]
        udir_cell_down = self._cell_features[1][hit_id]
        ncells         = self._cell_features[2][hit_id]

        inner_up   = np.einsum('ij,ij->i', udir, udir_cell_up  )
        inner_down = np.einsum('ij,ij->i', udir, udir_cell_down)
        is_up_closer = (np.abs(inner_up) > np.abs(inner_down))
        udir_cell = np.where(is_up_closer[:,np.newaxis], udir_cell_up, udir_cell_down)
        inner     = np.where(is_up_closer, inner_up, inner_down)
        is_backwards = (inner < 0)
        udir_cell[is_backwards,:] *= -1
        inner    [is_backwards]   *= -1

        # calculate heuristic distance measure of directions
        # Note: We multiply by sqrt(ncells) to account for the fact that directions
        #       estimated based on a small number of activated cells are less
        #       significant statistically.
        d = np.sqrt(ncells) * (1 - inner) # XXX refine
        return udir_cell, inner, d

    def _calculateCellFeatures(self, details=None):
        """Calculate features from cell data.
        Args:
            details (None or SimpleNamespace): If given, this is filled with some
                detailed data useful in off-line analyis.
        """
        # get event data
        event = self._run.event
        assert event.has_cells
        cells_df = event.cells_df

        # get description of detector modules
        detectors_df = self._algo.spec.geospec.detectors_df

        # generate an id (umod_id) for modules that is unique within the detector
        # calculate it for modules in detectors_df
        n_module_id = detectors_df['module_id'].max().astype(np.int32) + 1
        n_layer_id = detectors_df['layer_id'].max().astype(np.int32) + 1
        umod_id_by_module = (detectors_df['module_id'].values.astype(np.int32)
                   + n_module_id * (detectors_df['layer_id'].values.astype(np.int32)
                   + n_layer_id * detectors_df['volume_id'].values.astype(np.int32)))
        max_umod_id = umod_id_by_module.max()

        # calculate the unique module id (umod_id) for the hits
        umod_id_by_hit = np.zeros(1 + event.max_hit_id, dtype=np.int32)
        umod_id_by_hit[event.hits_df['hit_id'].values] = (
            event.hits_df['module_id'].values.astype(np.int32)
            + n_module_id * (event.hits_df['layer_id'].values.astype(np.int32)
            + n_layer_id * event.hits_df['volume_id'].values.astype(np.int32)))
        assert umod_id_by_hit.max() <= max_umod_id

        # generate maps from umod_id and hit_id into the detectors_df module index
        module_index_by_umod_id = np.zeros(1 + max_umod_id, dtype=np.int32)
        module_index_by_umod_id[umod_id_by_module] = np.arange(len(umod_id_by_module))
        module_index_by_hit = module_index_by_umod_id[umod_id_by_hit]

        # get the rotation matrix for each hit_id
        colnames = tuple('rot_%s%s' % (dst,src) for dst in ('x', 'y', 'z') for src in ('u', 'v', 'w'))
        rot_matrix_by_module = detectors_df.as_matrix(columns=colnames)
        rot_matrix_by_hit = rot_matrix_by_module[module_index_by_hit]
        rot_matrix_by_hit = rot_matrix_by_hit.reshape((-1, 3, 3))

        # get u,v-pitch and module half-thickness for each hit_id
        pitch_u_by_hit = detectors_df['pitch_u'].values[module_index_by_hit]
        pitch_v_by_hit = detectors_df['pitch_v'].values[module_index_by_hit]
        module_t_by_hit = detectors_df['module_t'].values[module_index_by_hit]

        # get cell data arrays
        hit_id = cells_df['hit_id'].values
        ch0 = cells_df['ch0'].values
        ch1 = cells_df['ch1'].values

        # calculate min/max cell indices per hit and their differences
        ch0_min_by_hit = np.full(1 + event.max_hit_id,  np.inf, dtype=np.float32)
        ch0_max_by_hit = np.full(1 + event.max_hit_id, -np.inf, dtype=np.float32)
        ch1_min_by_hit = np.full(1 + event.max_hit_id,  np.inf, dtype=np.float32)
        ch1_max_by_hit = np.full(1 + event.max_hit_id, -np.inf, dtype=np.float32)
        np.minimum.at(ch0_min_by_hit, hit_id, ch0)
        np.maximum.at(ch0_max_by_hit, hit_id, ch0)
        np.minimum.at(ch1_min_by_hit, hit_id, ch1)
        np.maximum.at(ch1_max_by_hit, hit_id, ch1)
        dch0_by_hit = ch0_max_by_hit - ch0_min_by_hit
        dch1_by_hit = ch1_max_by_hit - ch1_min_by_hit

        # detect multi-cell hits by looking at the cell index differences
        # Note: For some reason, there are a few single-cell hits which have
        #       multiple entries in the cells_df dataframe (see below).
        multicell_by_hit = (dch0_by_hit > 0) | (dch1_by_hit > 0)
        multicell_by_hit[0] = False

        # count cells per hit
        ncells_by_hit = np.zeros(1 + event.max_hit_id, dtype=np.int16)
        np.add.at(ncells_by_hit, hit_id, np.ones(len(hit_id), dtype=ncells_by_hit.dtype))

        # detect some strange cases where a single-cell hit has multiple
        # entries in cells_df
        is_strange = (multicell_by_hit != (ncells_by_hit > 1))
        self._algo.log("strange hits with unexpected number of pixels: ", np.where(is_strange)[0])

        # cell-center coordinates in the u,v-plane
        cu = ch0.astype(np.float64) * pitch_u_by_hit[hit_id]
        cv = ch1.astype(np.float64) * pitch_v_by_hit[hit_id]

        # calculated weighted means of cell centers per hit
        value = cells_df['value'].values
        value_sum_by_hit = np.zeros(1 + event.max_hit_id, dtype=np.float64)
        value_sum_by_hit[0] = 1.0 # avoid divide by zero
        np.add.at(value_sum_by_hit, hit_id, value)
        cu_weighted = value * cu
        cv_weighted = value * cv
        cu_wmean_by_hit = np.zeros(1 + event.max_hit_id, dtype=np.float64)
        cv_wmean_by_hit = np.zeros(1 + event.max_hit_id, dtype=np.float64)
        np.add.at(cu_wmean_by_hit, hit_id, cu_weighted)
        np.add.at(cv_wmean_by_hit, hit_id, cv_weighted)
        cu_wmean_by_hit /= value_sum_by_hit
        cv_wmean_by_hit /= value_sum_by_hit

        # reduce cell center coordinates by the per-hit weighted means
        cu_red = cu - cu_wmean_by_hit[hit_id]
        cv_red = cv - cv_wmean_by_hit[hit_id]

        # do weighted linear least-squares regression to determine the
        # direction of the particle in the u,v-plane

        # first we calculate some (non-normalized) weighted second moments
        wcucv = value * cu_red * cv_red
        wcucu = value * np.square(cu_red)
        wcvcv = value * np.square(cv_red)
        sum_wcucv_by_hit = np.zeros(1 + event.max_hit_id, dtype=np.float64)
        sum_wcucu_by_hit = np.zeros(1 + event.max_hit_id, dtype=np.float64)
        sum_wcvcv_by_hit = np.zeros(1 + event.max_hit_id, dtype=np.float64)
        np.add.at(sum_wcucv_by_hit, hit_id, wcucv)
        np.add.at(sum_wcucu_by_hit, hit_id, wcucu)
        np.add.at(sum_wcvcv_by_hit, hit_id, wcvcv)

        # for the linear regression, determine which coordinate is better
        # suited as the independent one (per hit)
        # (for single-cell hits, we use neither coordinate)
        indep_u = multicell_by_hit & (sum_wcucu_by_hit > sum_wcvcv_by_hit)
        indep_v = multicell_by_hit & ~indep_u

        # calculate dv/du
        dv_over_du_by_hit = np.full(1 + event.max_hit_id, np.inf, dtype=np.float64)
        dv_over_du_by_hit[indep_u]  = sum_wcucv_by_hit[indep_u] / sum_wcucu_by_hit[indep_u]

        # calculate du/dv
        du_over_dv_by_hit = np.zeros(1 + event.max_hit_id, dtype=np.float64)
        du_over_dv_by_hit[indep_v] = sum_wcucv_by_hit[indep_v] / sum_wcvcv_by_hit[indep_v]

        # combine into dv/du by calculating (du/dv)^(-1) where it makes sense
        du_over_dv_zero = (du_over_dv_by_hit == 0)
        use_du_over_dv = (indep_v & ~du_over_dv_zero)
        dv_over_du_by_hit[use_du_over_dv] = 1 / du_over_dv_by_hit[use_du_over_dv]

        # set a defined slope for single-pixel hits
        dv_over_du_by_hit[~multicell_by_hit] = 0

        # mask for finite dv/du
        finite_dv_over_du = np.isfinite(dv_over_du_by_hit)

        # unit vector in u,v-plane along the estimated trajectory
        uv_by_hit = np.ones(1 + event.max_hit_id, dtype=np.float64)
        uu_by_hit = 1 / np.sqrt(1 + np.square(dv_over_du_by_hit))
        uv_by_hit[finite_dv_over_du] = dv_over_du_by_hit[finite_dv_over_du] * uu_by_hit[finite_dv_over_du]
        assert np.all(uu_by_hit[~finite_dv_over_du] == 0)
        assert np.all(uv_by_hit[~finite_dv_over_du] == 1)

        # half-extent of a cell in the estimated direction
        hepx_by_hit = 0.5 * (np.abs(uu_by_hit) * pitch_u_by_hit + np.abs(uv_by_hit) * pitch_v_by_hit)

        # calculate u,v-displacement of each activated pixel along the
        # estimated u,v-direction of the trajectory
        d_udir2 = cu_red * uu_by_hit[hit_id] + cv_red * uv_by_hit[hit_id]

        # estimate half-length of track projected to the u,v-plane
        max_value_by_hit = np.full(1 + event.max_hit_id, -np.inf, dtype=np.float64)
        np.maximum.at(max_value_by_hit, hit_id, value)
        max_value = max_value_by_hit[hit_id]
        hepx = hepx_by_hit[hit_id]
        partial_pixel_factor = 1.0
        partial_pixel_intercept = -0.5
        d_udir2_forward  = d_udir2 + (partial_pixel_factor * (value / max_value) + partial_pixel_intercept) * hepx
        d_udir2_backward = d_udir2 - (partial_pixel_factor * (value / max_value) + partial_pixel_intercept) * hepx
        d_udir2_min_by_hit = np.full(1 + event.max_hit_id,  np.inf, dtype=np.float64)
        d_udir2_max_by_hit = np.full(1 + event.max_hit_id, -np.inf, dtype=np.float64)
        np.minimum.at(d_udir2_min_by_hit, hit_id, d_udir2_backward)
        np.maximum.at(d_udir2_max_by_hit, hit_id, d_udir2_forward)
        hl2_by_hit = (d_udir2_max_by_hit - d_udir2_min_by_hit) / 2
        hl2_by_hit[0] = 0 # dummy entry to avoid numpy errors

        # for single-cell hits, set half-length in u,v-plane to zero
        hl2_by_hit[~multicell_by_hit] = 0

        # estimate half-length of track in 3 dimensions
        hl_by_hit = np.sqrt(np.square(hl2_by_hit) + np.square(module_t_by_hit))

        # calculate unit vectors in estimated 3-d directions
        dir_u_by_hit = hl2_by_hit * uu_by_hit
        dir_v_by_hit = hl2_by_hit * uv_by_hit
        dir_w_by_hit = module_t_by_hit
        udir_uvw_up_by_hit = np.stack([dir_u_by_hit, dir_v_by_hit, dir_w_by_hit], axis=1) / hl_by_hit[:,np.newaxis]
        udir_uvw_down_by_hit = udir_uvw_up_by_hit.copy()
        udir_uvw_down_by_hit[:,2] *= -1

        # rotate directions to the x,y,z coordinate system
        udir_xyz_up_by_hit   = np.einsum('...ij,...j->...i', rot_matrix_by_hit, udir_uvw_up_by_hit  )
        udir_xyz_down_by_hit = np.einsum('...ij,...j->...i', rot_matrix_by_hit, udir_uvw_down_by_hit)

        # store some details useful for off-line analysis
        if details is not None:
            details.hepx_by_hit = hepx_by_hit
            details.hl2_by_hit = hl2_by_hit
            details.hl_by_hit = hl_by_hit
            details.d_udir2_min_by_hit = d_udir2_min_by_hit
            details.d_udir2_max_by_hit = d_udir2_max_by_hit
            details.pitch_u_by_hit = pitch_u_by_hit
            details.pitch_v_by_hit = pitch_v_by_hit
            details.module_t_by_hit = module_t_by_hit
            details.rot_matrix_by_hit = rot_matrix_by_hit
            details.cu_red = cu_red
            details.cv_red = cv_red

        return udir_xyz_up_by_hit, udir_xyz_down_by_hit, ncells_by_hit
