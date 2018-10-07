"""Keeping track of the list of candidate tracks.

This is mostly boring database-y stuff which is there to provide
a reasonably convenient and efficient storage service to the rest of
the solution code.

The only function of any algorithmical interest here is the `submit`
method (and its callee `zeroUsedHits`) which translates a candidates
list into a submission dataframe. These functions, which are practically
the only not-fully-vectorized ones in this file, apply several conditions
for including or excluding a track candidate in/from the submission.

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

def zeroUsedHits(hit_matrix, used,
                 min_nhits=3, max_nloss=None, max_loss_fraction=1.0, max_nrows=None,
                 reserve_skipped=False):
    """Set repeatedly used hit_ids in the given matrix to zero (except for the
    respective first occurrence) but only consider rows which meet some given
    criteria.
    Args:
        hit_matrix (array (N,M)): matrix of hit_ids (may contain zeroes)
            This matrix is modified by this function. All unused rows and all
            repeated hit_ids in used rows are set to zero.
        used (bool array(1 + max_hit_id,)): Element used[i] is True if
            hit_id == i has been used already.
            This matrix is updated by this function.
            Note: This function will end up setting used[0] = True for internal reasons.
        min_nhits (int): Only consider rows for inclusion which have at least min_nhits
            first occurences of hit_ids.
        max_nloss (int): Only consider rows which contain no more than max_nloss non-zero hit_ids
            which have already been used before (i.e. which either occurred in used rows above or
            which already had used[i]==True upon the call of this function).
        max_loss_fraction (float): Only consider rows in which the fraction of
            already-used non-zero hit_ids out of all non-zero hit_ids does not
            exceed the given fraction. (See max_nloss for definition of "already-used".)
        reserve_skipped (bool): If True, consider all hits of a skipped row as used in the rows
            below.
    """
    # default arguments to simplify the code below
    if max_nloss is None:
        max_nloss = hit_matrix.shape[1]
    if max_nrows is None:
        max_nrows = hit_matrix.shape[0]
    # mark hit_id 0 (which is used for empty slots) as used to simplify the rest of this code
    used[0] = True
    # iterate over the rows to choose which rows and hit_ids to use
    # Note: This cannot be easily vectorized due to the dependence of the conditions applied to
    #       one row on the hit_ids in the previous rows.
    nrows_taken = 0
    for i_row in range(hit_matrix.shape[0]):
        row = hit_matrix[i_row,:]
        nhits = np.sum(row != 0)
        assert nhits > 0
        mask_first = ~used[row]
        nfirst = np.sum(mask_first)
        nloss = nhits - nfirst
        loss_fraction = nloss / nhits
        if nfirst < min_nhits or nloss > max_nloss or loss_fraction > max_loss_fraction or nrows_taken >= max_nrows:
            # discard the entire row
            if reserve_skipped:
                used[row] = True
            del row # to avoid copying it on-demand
            hit_matrix[i_row,:] = 0
        else:
            # we use this row, but set redundant hit_ids to zero
            used[row] = True
            del row # to avoid copying it on-demand
            hit_matrix[i_row,:] *= mask_first
            nrows_taken += 1

class Candidates:
    """Keeps track of the candidates list.
    Args:
        event (data.Event): the event for which to track candidates.
        nmax_per_crossing (positive int): maximum number of hits per layer crossing.
        fit_columns (None or list of str): If given, this list defines the names of
            the fit parameters to store per candidate and layer crossing.
            If None, a default list will be used.
    Attributes:
        .n (integer): Number of candidate tracks.
        .nmax_per_crossing (integer): nmax_per_crossing passed to the constructor.
        .open (boolean array (N,)): True, if the track is still open for extension. (XXX remove?)
        .ncross (np.intX[*] array (N,)): Number of layer crossings in each candidate.
        .df (pd.DataFrame): companion data for the candidates, aligned with the candidates list.
    [*]...data type of .ncross is determined by ._ncross_dtype.
    """
    def __init__(self, event, nmax_per_crossing=4, fit_columns=None):
        if fit_columns is None:
            fit_columns = ['hel_xm', 'hel_ym', 'hel_r', 'hel_pitch', 'hel_pitch_ls', 'hel_ploss']
        self.open = None
        self._event = event
        self.nmax_per_crossing = nmax_per_crossing
        self._fit_columns = fit_columns
        self._nfit = len(fit_columns)
        self._fit_columns_map = dict(zip(fit_columns, range(len(fit_columns))))
        self._fit_empty = np.nan
        self._fit_dtype      = np.float32 # data type for fit parameters
        self._ncross_dtype   = np.int8    # data type for number of crossings
        self._track_id_dtype = np.int32   # data type for track_ids in submission
        self._idx_dtype      = np.int32   # data type for flat and row indices
        self._col_dtype      = np.int32   # data type for column indices

    def copy(self):
        """Return a deep copy of this candidates list."""
        the_copy = Candidates(self._event, nmax_per_crossing=self.nmax_per_crossing, fit_columns=self._fit_columns)
        if self.open is not None:
            the_copy.open = self.open.copy()
            the_copy.ncross = self.ncross.copy()
            the_copy._candidates = self._candidates.copy()
            the_copy._fit = self._fit.copy()
            the_copy.df = self.df.copy()
        return the_copy

    def initialize(self, hit_matrix, data_df):
        """Set the candidate list to contain exactly the given hits.
        Args:
            hit_matrix (np.int32 array (N,M)): hit_ids of the candidates. One row for
                each of the N candidates. M must be a multiple of self.nmax_per_crossing.
            data_df (pd.DataFrame, aligned with rows of hit_matrix): companion data for
                the candidates
        """
        assert self.open is None
        self._candidates = hit_matrix
        n, width = self._candidates.shape
        max_ncross = width // self.nmax_per_crossing
        assert width == max_ncross * self.nmax_per_crossing
        assert len(data_df) == n
        self.open = np.full(n, True)
        self.ncross = np.count_nonzero(self._candidates[:,::self.nmax_per_crossing], axis=1).astype(self._ncross_dtype)
        self._fit = np.full((n, self._nfit * max_ncross), self._fit_empty, dtype=self._fit_dtype)
        self.df = data_df.reset_index(drop=True)

    def addSeeds(self, hit_ids, data_df):
        """Add the given arrays of hits, one for each layer crossing,
        to the candidates list.
        Args:
            hit_ids (list of np.int32 array (Ntriples,)):
                hit_ids of the hits comprising the candidates.
            data_df (pd.DataFrame, index aligned with hit_ids[...]):
                dataframe with companion data for the newly added
                candidates.
        """
        ncross_seeds = len(hit_ids)
        assert ncross_seeds >= 1
        nseeds = len(hit_ids[0])
        assert len(data_df) == nseeds
        if self.open is None:
            nnew = nseeds
            self.open = np.full(nseeds, True)
            self.ncross = np.full(nseeds, ncross_seeds, dtype=self._ncross_dtype)
            self._candidates = np.zeros((nseeds, self.nmax_per_crossing * ncross_seeds), dtype=np.int32)
            self._fit = np.full((nseeds, self._nfit * ncross_seeds), self._fit_empty, dtype=self._fit_dtype)
            self.df = data_df # Note: reset_index creates a copy below
        else:
            old_shape = self._candidates.shape
            nold = old_shape[0]
            nnew = nold + nseeds
            new_max_crossings = max(old_shape[1] // self.nmax_per_crossing, ncross_seeds)
            new_candidates = np.zeros((nnew, self.nmax_per_crossing * new_max_crossings), dtype=np.int32)
            new_candidates[0:nold,0:old_shape[1]] = self._candidates
            new_fit = np.full((nnew, self._nfit * new_max_crossings), self._fit_empty, dtype=self._fit_dtype)
            new_fit[0:nold,0:self._fit.shape[1]] = self._fit
            new_ncross = np.zeros(nnew, self._ncross_dtype)
            new_ncross[0:nold] = self.ncross
            new_ncross[nold:] = ncross_seeds
            new_open = np.full(nnew, True)
            new_open[0:nold] = self.open
            self.open = new_open
            self.ncross = new_ncross
            self._candidates = new_candidates
            self._fit = new_fit
            self.df = pd.concat([self.df, data_df], axis=0)
        self.df = self.df.reset_index(drop=True)
        assert len(self.df) == nnew
        # fill in hit_ids for the new seeds
        for i, hit_id in enumerate(hit_ids):
            assert len(hit_id) == nseeds
            self._candidates[(nnew - nseeds):,self.nmax_per_crossing * i] = hit_id

    @property
    def n(self):
        """The number of candidates."""
        assert self.open is not None
        return len(self.open)

    def has(self, which, allow_negative=True):
        """Return a mask specifying whether each candidate has hits for the requested
        layer crossings.
        Args:
            which: list of (int or tuple): elements specify:
                tuple (crossing, pair): check if the given pair index has a hit for
                    this crossing
                int (crossing): equivalent to tuple (crossing, 0)
            allow_negative (bool): if True, allow negative crossing indices in order
                to refer to crossings counting from the end of the respective candidate.
                If False, consider negative crossing indices out-of-bound (will result
                in the returned "has_em" value being False for this candidates).
        Returns:
            has_em (np.bool array(N,)): True for candidates which have ALL the
                requested hits
        """
        n, width = self._candidates.shape
        max_ncross = width // self.nmax_per_crossing
        has_em = np.ones(n, dtype=np.bool)
        for w in which:
            if type(w) is tuple:
                crossing, pair = w
            else:
                crossing, pair = w, 0
            # XXX reuse code for the following calculation
            if allow_negative:
                crossing_col = np.where(crossing >= 0, crossing, self.ncross.astype(self._col_dtype) + crossing)
            else:
                crossing_col = crossing
            in_bounds = (crossing_col >= 0) & (crossing_col < max_ncross)
            col = self.nmax_per_crossing * crossing_col + pair
            row = np.arange(n, dtype=self._idx_dtype)
            # end of column calculation
            has_em &= in_bounds
            if np.isscalar(in_bounds):
                if in_bounds:
                    has_em &= (self._candidates[row,col] != 0)
            else:
                has_em[in_bounds] &= (self._candidates[row[in_bounds],col[in_bounds]] != 0)
        return has_em

    def hitIds(self, crossing, pair=0, mask=None):
        """Get hit_ids of the given crossing for all tracks.
        Args:
            crossing (int, or int array(N,)): index of the layer crossing to look at.
                non-negative indices count from the beginning of the
                candidate track (0=first layer crossing), negative
                values count from the end (-1=last layer crossing).
            pair (int in range(nmax_per_crossing), or array (N,) of them):
                which of the hits in each layer crossing to take.
            mask (np.bool array (N,)): optional, if given, return only values
                for which mask is True
        Returns:
            hit_id (array (N[*],)): hit_ids of the hits.
        [*]...if mask is used, this N reduced to np.sum(mask, axis=0) here.
        """
        assert self.open is not None
        n, width = self._candidates.shape
        max_ncross = width // self.nmax_per_crossing
        crossing_col = np.where(crossing >= 0, crossing, self.ncross.astype(self._idx_dtype) + crossing)
        out_of_bounds = (crossing_col < 0) | (crossing_col >= max_ncross)
        col = self.nmax_per_crossing * crossing_col + pair
        indices = np.arange(0, width * n, width, dtype=self._idx_dtype) + col
        indices[out_of_bounds] = 0
        hit_ids = self._candidates.take(indices)
        hit_ids[out_of_bounds] = 0
        if mask is not None:
            hit_ids = hit_ids.compress(mask)
        return hit_ids

    def hitIdsFlat(self, n):
        """Get an array with the hit_ids of the first n (or last -n) hits per candidate.
        Args:
            n (integer): If positive, get the first n hits. If negative, get the last -n hits.
        Returns:
            hit_id (array (N, abs(n)): the hit_ids.
        """
        assert n != 0
        hit_rows, hit_cols = np.nonzero(self._candidates)
        row_start = (hit_rows != np.roll(hit_rows, 1))
        row_start[0] = True
        running_index = np.arange(len(row_start))
        row_start_index = np.maximum.accumulate(np.where(row_start, running_index, 0))
        index_in_row = running_index - row_start_index
        if n > 0:
            in_range = (index_in_row < n)
        else:
            index_in_row -= self.nHits()[hit_rows] + n
            in_range = (index_in_row >= 0)
        hit_ids = np.zeros((self.n, abs(n)), dtype=np.int32)
        hit_rows = hit_rows[in_range]
        hit_cols = hit_cols[in_range]
        index_in_row = index_in_row[in_range]
        hit_ids[hit_rows, index_in_row] = self._candidates[hit_rows, hit_cols]
        return hit_ids

    def setHitIds(self, crossing, hit_id, pair=0, mask=None):
        """Set hit_ids of the given crossing for all tracks.
        Args:
            crossing (int, or int array(N,)): index of the layer crossing to modify.
                non-negative indices count from the beginning of the
                candidate track (0=first layer crossing), negative
                values count from the end (-1=last layer crossing).
            hit_id (array (N,)): hit_ids of the hits (0 if no hit)
            pair (int in range(nmax_per_crossing), or array (N,) of them):
                which of the hits in each layer crossing to set.
            mask (np.bool array (N,)): optional, if given, modify only values
                for which mask is True
        """
        assert self.open is not None
        n, width = self._candidates.shape
        max_ncross = width // self.nmax_per_crossing
        crossing_col = np.where(crossing >= 0, crossing, self.ncross.astype(self._col_dtype) + crossing)
        in_bounds = (crossing_col >= 0) & (crossing_col < max_ncross)
        col = self.nmax_per_crossing * crossing_col + pair
        row = np.arange(n, dtype=self._idx_dtype)
        modify_mask = in_bounds
        if mask is not None:
            modify_mask = modify_mask & mask
        self._candidates[row[modify_mask], col[modify_mask]] = hit_id[modify_mask]

    def hitCoordinates(self, crossing, pair=0, mask=None):
        """Get hit coordinates of the given crossing for all tracks.
        Args:
            crossing (int, or int array (N,)): index of the layer crossing to look at.
                non-negative indices count from the beginning of the
                candidate track (0=first layer crossing), negative
                values count from the end (-1=last layer crossing).
            pair (int in range(nmax_per_crossing), or array (N,) of them):
                which of the hits in each layer crossing to take.
            mask (np.bool array (N,)): optional, if given, return only values
                for which mask is True
        Returns:
            x, y, z (array (N[*],)): coordinates of the hits.
        [*]...if mask is used, this N reduced to np.sum(mask, axis=0) here.
        """
        assert self.open is not None
        return self._event.hitCoordinatesById(self.hitIds(crossing, pair, mask=mask))

    def nHits(self):
        """Get the number of hits in each candidate track.
        Returns:
            nhits (array (N,)): number of hits per track
        """
        assert self.open is not None # XXX cleaner name for check
        nhits = np.count_nonzero(self._candidates, axis=1)
        return nhits

    def hitParticles(self):
        """Return an array of particle ids corresponding to the hits in the candidate list.
        Precondition:
            self._event.has_truth
        Returns:
            particle_ids (array of same shape as candidates list): particle_ids for the
                hits in the candidates list (organized as in self._candidates).
                particle_id == -1 is used for empty candidate list elements.
        """
        assert self._event.has_truth
        particle_map = self._event.hitToParticleMap()
        return particle_map[self._candidates]

    def getFit(self, crossing, column, mask=None):
        """Get fit parameters of the given crossing for all tracks.
        Args:
            crossing (int, or int array(N,)): index of the layer crossing to look at.
                non-negative indices count from the beginning of the
                candidate track (0=first layer crossing), negative
                values count from the end (-1=last layer crossing).
            column (str): name of fit parameter to retrieve.
            mask (np.bool array (N,)): optional, if given, return only values
                for which mask is True.
        Returns:
            fit_param (array (N[*],)): fit parameter values.
        [*]...if mask is used, this N is reduced to np.sum(mask, axis=0) here.
        """
        assert self.open is not None
        n, width = self._fit.shape
        max_ncross = width // self._nfit
        crossing_col = np.where(crossing >= 0, crossing, self.ncross.astype(self._idx_dtype) + crossing)
        out_of_bounds = (crossing_col < 0) | (crossing_col >= max_ncross)
        col = self._nfit * crossing_col + self._fit_columns_map[column]
        indices = np.arange(0, width * n, width, dtype=self._idx_dtype) + col
        indices[out_of_bounds] = 0
        fit_param = self._fit.take(indices)
        fit_param[out_of_bounds] = self._fit_empty
        if mask is not None:
            fit_param = fit_param.compress(mask)
        return fit_param

    def setFit(self, crossing, column, fit_param, mask=None):
        """Set fit parameters of the given crossing for all tracks.
        Args:
            crossing (int, or int array(N,)): index of the layer crossing to modify.
                non-negative indices count from the beginning of the
                candidate track (0=first layer crossing), negative
                values count from the end (-1=last layer crossing).
            column (str): name of fit parameter to set.
            fit_param (array (N[*],)): fit parameter values to set.
            mask (np.bool array (N,)): optional, if given, modify only values
                for which mask is True
        [*]...if mask is used, this N is reduced to np.sum(mask, axis=0) here.
        """
        assert self.open is not None
        n, width = self._fit.shape
        max_ncross = width // self._nfit
        crossing_col = np.where(crossing >= 0, crossing, self.ncross.astype(self._col_dtype) + crossing)
        in_bounds = (crossing_col >= 0) & (crossing_col < max_ncross)
        col = self._nfit * crossing_col + self._fit_columns_map[column]
        row = np.arange(n, dtype=self._idx_dtype)
        modify_mask = in_bounds
        use_mask    = in_bounds
        if mask is not None:
            modify_mask = modify_mask & mask
            use_mask = modify_mask.compress(mask)
        self._fit[row[modify_mask], col[modify_mask]] = fit_param[use_mask]

    def update(self, close_mask=False, keep_mask=True, extend_index=[], extend_hit_ids=[]):
        """Update the candidates list after an algorithm step.
        Args:
            close_mask (bool or bool array (N,)): Selects which tracks to close.
            keep_mask (bool or bool array (N,)): Selects which tracks to keep.
            extend_index (np.int32 array (M,)): indices in [0,Nopen), choosing which
                tracks to extend by another hit (or two).
            extend_hit_ids (list of np.int32 array (M,)): hit_ids to extend
                the track with.
        where
            N is the size of the candidates list before the update,
            M is an arbitrary non-negative integer (but must be the same for the
                three arrays given).
        XXX return candidate_ids of the extended tracks
        XXX limit number of layer crossings when extending
        """
        assert self.open is not None
        nextend = len(extend_index)
        assert len(extend_hit_ids) <= self.nmax_per_crossing
        for extend_hit_id in extend_hit_ids:
            assert len(extend_hit_id) == nextend
        if np.isscalar(keep_mask):
            keep_mask = np.full(len(self.open), keep_mask, dtype=np.bool)
        else:
            assert len(keep_mask) == len(self.open)
        self.open[close_mask] = False
        if nextend > 0:
            nold = np.sum(keep_mask)
            nnew = nold + nextend
            old_shape = self._candidates.shape
            extend_ncross = self.ncross[extend_index]
            new_max_crossings = max(old_shape[1] // self.nmax_per_crossing, np.amax(extend_ncross, axis=0) + 1)
            new_candidates = np.zeros((nnew, self.nmax_per_crossing * new_max_crossings), dtype=np.int32)
            # XXX compress candidates and throw away full array before?
            new_candidates[0:nold,0:old_shape[1]] = self._candidates.compress(keep_mask, axis=0)
            new_candidates[nold:,0:old_shape[1]] = self._candidates[extend_index,:]
            new_fit = np.full((nnew, self._nfit * new_max_crossings), self._fit_empty, dtype=self._fit_dtype)
            # XXX compress fit and throw away full array before?
            new_fit[0:nold,0:self._fit.shape[1]] = self._fit.compress(keep_mask, axis=0)
            new_fit[nold:,0:self._fit.shape[1]] = self._fit[extend_index,:]
            new_indices = np.arange(nold, nnew, 1, dtype=self._idx_dtype)
            assert new_indices.shape == extend_ncross.shape
            for i, extend_hit_id in enumerate(extend_hit_ids):
                new_candidates[new_indices,self.nmax_per_crossing * extend_ncross.astype(self._col_dtype) + i] = extend_hit_id
            new_ncross = np.zeros(nnew, self._ncross_dtype)
            new_ncross[0:nold] = self.ncross.compress(keep_mask)
            new_ncross[nold:] = extend_ncross + 1
            new_open = np.full(nnew, True)
            new_open[0:nold] = self.open.compress(keep_mask)

            self.open = new_open
            self.ncross = new_ncross
            self._candidates = new_candidates
            self._fit = new_fit
            keep_indices = np.nonzero(keep_mask)[0]
            backref_indices = np.concatenate([keep_indices, extend_index], axis=0)
            self.df = self.df.iloc[backref_indices].reset_index(drop=True)
        else:
            indices, = np.where(keep_mask)
            self.permute(indices)

    def findUnique(self, crossings):
        """For a given list of layer crossings, find candidates which assign
        a unique choice of hit_ids to these crossings. If several candidates
        have the same hit_id set for the given crossings, the first one of
        each group in candidate list order is taken.
        Args:
            crossings: list of layer crossing indices for which to demand
                uniqueness of hits
        Returns:
            mask (array (N,) of np.bool): Mask selecting the first unique
                candidates.
        """
        cols = [col for i in crossings for col in range(self.nmax_per_crossing * i, self.nmax_per_crossing * (i+1))]
        keys = np.sort(self._candidates[:,cols], axis=1)
        unique, indices = np.unique(keys, axis=0, return_index=True)
        mask = np.zeros(self.n, dtype=np.bool)
        mask[indices] = True
        return mask

    def permute(self, indices):
        """Order candidates list as specified by the given indices array.
        Args:
            indices (np.int32 array(M,)): Indices into the existing
                candidates list. (Typically, indices could be the result
                of an np.argsort on candidate features.)
                If M is smaller than the number of candidates, the candidates
                list is shorted to the M specified ones.
        """
        assert self.open is not None
        self.open = self.open[indices]
        self.ncross = self.ncross[indices]
        self._candidates = self._candidates[indices,:]
        self._fit = self._fit[indices,:]
        self.df = self.df.iloc[indices].reset_index(drop=True)

    def submit(self, fill=True, used=None, min_track_id=1,
               min_nhits=3, max_nloss=None, max_loss_fraction=1.0, max_nrows=None, reserve_skipped=False):
        """Return a submission dataframe for this candidates list.
        Args:
            fill (bool): If true, fill the submission up with a dummy track so that
                all hit_ids of the event are used.
            used (bool array(1 + event.max_hit_id,)): if element [i] is True, assume
                that hit_id == i has already been used and does not need to be
                put in the dummy track for filling the submission.
            min_track_id (integer): the minimum track_id to use for this candidate
                list.
                track_ids assigned are min_track_id + (row index of candidate).
            min_nhits, max_nloss, max_loss_fraction, max_nrows, reserve_skipped:
                Parameters forwarded to zeroUsedHits. See there.
        Returns:
            pd.DataFrame with columns 'event_id', 'hit_id', 'track_id'.
        """
        if self.open is not None:
            if used is None:
                used = np.zeros(1 + self._event.max_hit_id, dtype=np.bool)
            else:
                used = used.copy() # copy, because zeroUsedHits modifies the one passed to it
            # XXX should remember masked_candidates for offline analysis
            masked_candidates = self._candidates.copy()
            zeroUsedHits(masked_candidates, used,
                         min_nhits=min_nhits, max_nloss=max_nloss, max_loss_fraction=max_loss_fraction,
                         max_nrows=max_nrows, reserve_skipped=reserve_skipped)
            sorted_hit_id, indices = np.unique(masked_candidates, return_index=True)

            # remove the dummy hit_id zero
            if sorted_hit_id.size and sorted_hit_id[0] == 0:
                sorted_hit_id = sorted_hit_id[1:]
                indices = indices[1:]

            # assign track ids (they do not need to be contiguous, so
            # we just use an offset plus the row index)
            track_id = (self._track_id_dtype(min_track_id)
                        + np.floor_divide(indices, self._candidates.shape[1]).astype(self._track_id_dtype))
        else:
            sorted_hit_id = np.array([], dtype=np.int32)
            track_id      = np.array([], dtype=self._track_id_dtype)

        if fill:
            # fill up submission for hit_ids we did not use
            all_hit_id = self._event.hits_df['hit_id'].values
            max_hit_id = self._event.max_hit_id
            missing = np.full(max_hit_id + 1, False)
            missing[all_hit_id] = True
            missing[sorted_hit_id] = False
            if used is not None:
                missing &= ~used
            missing_hit_id = np.where(missing)[0]
        else:
            missing_hit_id = np.array([], dtype=np.int32)

        # build submission DataFrame
        submission_df = pd.DataFrame(data=OrderedDict([
            ('event_id', np.int16(self._event.event_id)),
            ('hit_id'  , np.concatenate([sorted_hit_id, missing_hit_id                                           ])),
            ('track_id', np.concatenate([track_id     , np.zeros(len(missing_hit_id), dtype=self._track_id_dtype)])),
        ]))
        return submission_df

    def analysisDataframe(self):
        """Return a dataframe with info for offline analysis.
        The dataframe rows are aligned with the list of candidates.
        """
        ana_df = pd.DataFrame(data={'open': self.open, 'ncross': self.ncross, 'nhits': self.nHits()})
        max_ncross = np.amax(self.ncross, axis=0)
        for i_cross in range(max_ncross):
            for i_pair in range(self.nmax_per_crossing):
                hit_ids = self.hitIds(i_cross, pair=i_pair)
                ana_df['%d_hit%d' % (i_cross, i_pair)] = hit_ids
        fit_df = pd.DataFrame(data=self._fit[:,:(self._nfit * max_ncross.astype(self._col_dtype))],
            columns=['%d_%s' % (i_cross, name) for i_cross in range(max_ncross) for name in self._fit_columns])
        ana_df = pd.concat([ana_df, fit_df, self.df], axis=1)
        return ana_df

    def dump(self, filename, mask=None):
        """Dump a readable description of the candidates list to a file.
        Args:
            filename (str): path of the file to dump to.
            mask (np.bool array (N,)): optional, if given, show only the candidates
                for which mask is true (with their indices).
        """
        with open(filename, "w") as file:
            file.write(self.__str__(mask=mask))

    def __str__(self, mask=None,
                show_coords=False, show_r2=False,
                show_pid=False, show_fit=False, show_df=False,
                break_cross=False,
                hit_in_cyl=None, hit_layer_id=None):
        """Get a readable string describing the candidates list.
        Args:
            mask (np.bool array (N,)): optional, if given, show only the candidates
                for which mask is true (with their indices).
            show_coords (bool): if True, include hit coordinates.
            show_r2 (bool): if True, include derived hit coordinate r2.
            show_pid (bool): if True, include particle_ids for hits, if available.
            show_fit (bool): if True, include fit data.
            show_df (bool): if True, include companion data.
            break_cross (bool): if True, break the line after each layer crossing.
            hit_in_cyl (None or bool array(1 + max_hit_id,)): if given, indicate
                whether each hit is in a cylinder (True) or a cap (False).
            hit_layer_id (None or int array (1 + max_hit_id,)): if given, indicate
                in which cylinder (cyl_id) or cap (cap_id) each hit resides.
        """
        if show_coords or show_r2:
            hit_coords = self._event.hitCoordinatesById(np.arange(1 + self._event.max_hit_id, dtype=np.int32))
        else:
            hit_coords = None
        if show_pid and self._event.has_truth:
            hit_pids = np.full(1 + self._event.max_hit_id, -1, dtype=np.int64)
            hit_pids[self._event.truth_df['hit_id'].values] = self._event.truth_df['particle_id'].values
        else:
            hit_pids = None
        n = self.n
        nopen = np.sum(self.open, axis=0)
        text = ("candidates: " + str(n) + " (" + str(nopen) + " open / " + str(n - nopen)
                + " closed), shape " + str(self._candidates.shape) + "\n")
        for i in range(n):
            if mask is not None:
                if not mask[i]: continue
                text += "[%5d]" % i
            row = self._candidates[i,:]
            text += "    %2d" % self.ncross[i] + " %6s" % (('closed', 'open')[self.open[i]])
            for j in range(self.ncross[i]):
                if break_cross and j > 0:
                    text += "\n" + " " * (7 + 6 + 7)
                hit_ids = [hit_id for hit_id in row[self.nmax_per_crossing * j:self.nmax_per_crossing * (j+1)] if hit_id > 0]
                hit_strs = []
                for hit_id in hit_ids:
                    hit_text = str(hit_id)
                    if hit_in_cyl is not None:
                        hit_text += (" cap", " cyl")[hit_in_cyl[hit_id]]
                    if hit_layer_id is not None:
                        hit_text += ":" + str(hit_layer_id[hit_id])
                    if show_coords:
                        hit_text += "(" + ",".join(["%.2f" % coord[hit_id] for coord in hit_coords]) + ")"
                    if show_r2:
                        hit_text += "(r2=%.1f)" % np.sqrt(np.sum(
                            [np.square(coord[hit_id]) for coord in hit_coords[:2]], axis=0))
                    if hit_pids is not None:
                        hit_text += "[" + str(hit_pids[hit_id]) + "]"
                    hit_strs.append(hit_text)
                if len(hit_ids) > 1:
                    text += " (" + ",".join(hit_strs) + ")"
                elif len(hit_ids) == 1:
                    text += " " + hit_strs[0]
                else:
                    text += " (none)"
                if show_fit:
                    text += "{"
                    for col, colname in enumerate(self._fit_columns):
                        if col > 0:
                            text += ","
                        text += colname + ": %f" % self._fit[i, self._nfit * j + col]
                    text += "}"
            text += "\n"
            if show_df:
                text += "        "
                text += ", ".join([col + ": " + str(self.df.loc[i,col]) for col in self.df.columns])
                text += "\n"

        return text
