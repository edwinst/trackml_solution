"""Provision of test and training data in a streamlined form.

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
import re
import numpy as np
import pandas as pd

from trackml.dataset import load_event

class Event:
    """This class provides data for one event (possibly including ground truth)
    in such a way that it can be consumed conveniently and efficiently by
    the rest of the code.
    """
    def __init__(self, prefix, path=None, with_truth=False, with_cells=False):
        """Set up a representation of an event. (This does not yet load any data from disk.)
        Args:
            prefix (str): filename prefix to find the event files
            path (str): directory path in which to look for event files
            with_truth (bool or 'auto'): if True, load ground truth when
                this event is opened.
                If 'auto', load ground truth if the ground truth files
                are available when the event is opened.
            with_cells (bool): if True, load cell data when this event is opened.
        """
        if path is not None:
            prefix = os.path.join(path, prefix)
        match = re.match('.*event(0*\d+)', prefix)
        assert match is not None # we must be able to extract the event_id
        self.event_id = int(match.group(1))
        self._prefix = prefix
        if with_truth == 'auto':
            truth_filename = self._prefix + '-truth.csv'
            with_truth = os.path.isfile(truth_filename)
        self._with_truth = with_truth
        self._with_cells = with_cells
        self._isopen = False
        self._hit_id_to_particle_id = None

    @property
    def has_cells(self):
        """Return True if cells data is available for the event."""
        return self._with_cells

    @property
    def has_truth(self):
        """Return True if ground truth is available for the event."""
        return self._with_truth

    def open(self, use_true_coords=False):
        """Load event data from disk."""
        parts = ['hits']
        if self._with_cells:
            parts.append('cells')
        if self._with_truth:
            parts.extend(['particles', 'truth'])
        data = load_event(self._prefix, parts=parts)
        hits = data[0]
        if self._with_cells:
            self.cells_df = data[1]
        else:
            self.cells_df = None
        self.hits_df = hits
        self._validate()
        min_hit_id = hits['hit_id'].min()
        max_hit_id = hits['hit_id'].max()
        self.max_hit_id = max_hit_id
        # assume that hit_ids are exactly the integers 1,...,#hits
        # this way we can speed up the coordinate lookup quite a bit
        assert min_hit_id == 1 and max_hit_id == len(hits)
        if self._with_truth:
            particles, truth = data[-2:]
            self._particles_columns_orig = particles.columns
            self._truth_columns_orig = truth.columns
            self.particles_df = particles
            self.truth_df = truth
        else:
            assert not use_true_coords
            self.particles_df = None
            self.truth_df = None
        self.hits_coords_by_id = []
        for i, col in enumerate(('x', 'y', 'z')):
            # IMPORTANT: We convert coordinates to float64. Otherwise we run
            # into numerical precision problems in our helix arithmetic.
            # Note: index 0 must map to np.nan.
            coord = np.full(1 + max_hit_id, np.nan, dtype=np.float64)
            if use_true_coords:
                true_col = 't' + col
                coord[self.truth_df['hit_id'].values] = self.truth_df[true_col].values
            else:
                coord[hits['hit_id'].values] = hits[col].values
            self.hits_coords_by_id.append(coord)
        self._hits_module_id_by_id = np.full(1 + max_hit_id, -1, dtype=np.int16)
        assert np.all(hits['module_id'] <= np.iinfo(self._hits_module_id_by_id.dtype).max)
        self._hits_module_id_by_id[hits['hit_id'].values] = hits['module_id'].values
        self._isopen = True

    def close(self):
        """End access to this event's data and free memory."""
        self.hits_df = None
        self.cells_df = None
        self.hits_coords_by_id = None
        self.truth_df = None
        self.particles_df = None
        self._isopen = False
        self._hit_id_to_particle_id = None

    def summary(self):
        """Return a string giving a human-readable summary of this event."""
        text = "event " + str(self.event_id) + ": "
        if self._isopen:
            text += "open, #hits=" + str(len(self.hits_df))
            if self._with_cells:
                text += ", #cells=" + str(len(self.cells_df))
            if self._with_truth:
                text += ", #particles=" + str(len(self.particles_df))
                text += ", #truth=" + str(len(self.truth_df))
            else:
                 text += ", (no truth)"
        else:
            text += "closed"
        return text

    def hitCoordinatesById(self, hit_id):
        """Get coordinate arrays for the hits with the given hit_ids.
        Args:
            hit_id (pd.Series or np.int32 array(N,)): hit_ids for which to give coordinates.
                hit_id[i] == 0 is allowed and results in x[i], y[i], z[i] being np.nan.
        Returns:
            x, y, z (np.float64 array (N,)): coordinates of the given hits.
        """
        assert self._isopen
        return tuple(coord[hit_id] for coord in self.hits_coords_by_id)

    def hitModuleIdById(self, hit_id):
        """Get module_id for the hits with the given hit_ids.
        Args:
            hit_id (pd.Series or np.int32 array(N,)): hit_ids for which to give module_ids.
                hit_id[i] == 0 is allowed and results in module_id == -1.
        Returns:
            module_id (np.int16 array (N,)): module_ids of the given hits.
        """
        assert self._isopen
        return self._hits_module_id_by_id[hit_id]

    def hitToParticleMap(self):
        """Return an array mapping hit_id (used to index the array) to particle_id.
        Precondition:
            event.has_truth
        Returns:
            hit_id_to_particle_id (array (#hits + 1,)): element [i] is the particle_id
                of the particle which caused hit with hit_id == i (may be zero for noise
                hits). element[0] is -1.
        """
        assert self.has_truth
        if self._hit_id_to_particle_id is None:
            self._hit_id_to_particle_id = np.empty(1 + len(self.hits_df), dtype=np.int64)
            self._hit_id_to_particle_id[0] = -1
            # XXX per event, unique particle_ids could be compressed to 16 bits
            self._hit_id_to_particle_id[self.truth_df['hit_id'].values] = self.truth_df['particle_id'].values
        return self._hit_id_to_particle_id

    def particleNHits(self, pids):
        """Get number of true hits for the given particle ids.
        Precondition:
            event.has_truth
        Args:
            pids (array (N,)): particle ids for which to look up number of hits.
        Returns:
            nhits (integer array (N,)): number of true hits for each given particle id.
        """
        assert self.has_truth
        pids_df = pd.DataFrame(data={'particle_id': pids})
        nhits = pids_df.merge(self.particles_df, on='particle_id', how='left')['nhits'].values
        return nhits

    def _validate(self):
        """Do some sanity checks on the event data to assert some assumptions we make."""
        # We use 0 as a marker for "no hit", so make sure it does not occur
        assert self.hits_df['hit_id'].min() > 0
