trackml Solution #5 (Edwin Steiner)
===================================

The software in this directory consitutes a solution to the
kaggle competition ["TrackML Particle Tracking Challenge"](https://www.kaggle.com/c/trackml-particle-identification).
This solution scored the fifth place in the final public and
private leaderboards. The final score reached was 0.86395. On my
notebook (with Intel Core i7-3667U @ 2GHz) the submission took about
3 minutes per event to calculate with a single thread, making this solution
possibly the fastest of the high-scoring ones in the trackml accuracy phase.

Note: In addition to the information in this README file and the
comments in the code, some discussion of the solution can be found
in the kaggle forum threads like [this one](https://www.kaggle.com/c/trackml-particle-identification/discussion/63249).

## Solution Concept

This solution implements a combinatoric geometric algorithm which first builds
a candidate list of track seeds by picking combinations of two to three hits
in adjacent detector layers and then alternately truncates this list to the
most likely correct track candidates and extends each candidate track by
finding the hit(s) closest to an extrapolated idealized trajectory fitted
to the hits found so far.

The extension step first predicts intersections points of idealized helix
trajectories with an idealized detector geometry made of perfect cylinders
and planar caps. The intersection points are then corrected using a model
derived from the training data. For each intersection point, the k closest
hits are considered as potential extensions of the respective track
candidate. Heuristics then decide which hit(s) to use (if any) to extend
each candidate. Once the extension is decided, further "paired" hits
are searched, i.e. further hits that the particle may have produced in
the same detector layer crossing.

Between extension steps, the candidate tracks are ranked using a track
quality metric (called "value" in the code) which, among other things,
evaluates the smoothness of each track candidate by backward-fitting
helix trajectories to it. The candidates list is then truncated to the
best N candidates, where the number N varies from one extension step
to the next. It starts out as a number scaling quadratically with the total
number of hits in the problem and approaches a linearly scaled threshold.

After a certain maximum number of extension steps have been completed,
the best track candidates are committed to the submission dataframe.
At this point a particular hit may be used by several track candidates.
The commit algorithm resolves such cases by processing the track candidates
with the highest track value first and removing used hits from later,
lower-valued tracks. Candidates which lose too many hits in this way
are completely skipped by the commit step (except in the very last
iteration) because such cases often arise when a partially correct
track "switches over" to another partial track corresponding to a
different particle.

After committing the best tracks in this way, the candidate list is reset
and the whole algorithm repeats with only the still unused hits 
being considered. This is repeated until the supply of hits has been exhausted
or a certain maximum number of these "commit rounds" has been done.

The heuristic parts of the algorithm are controlled by a rather large number
of hyper-parameters. The parameter files used to calculate the final kaggle
submission are included in this software distribution. During development
I made heavy use of [hyperopt](http://hyperopt.github.io/hyperopt/)[1] to
optimize these hyper-parameters.

Supervised learning is only realized in the most primitive sense:
Training data is used to calculate several statistics about hits and
trajectories which are summarized in interpolated functions for each
detector layer. These function in turn are used to inform various
heuristics used on the test data.

## Implementation

This solution is implemented in pure Python (3) and makes massive use
of vectorization as supported by the numpy and pandas libraries.

The neighborhood searches for track seeds and extension hits are
implemented using the very fast nearest-neighbor algorithms provided
by the sklearn.neighbors library. There is one nearest-neighbor data
structure maintained per detector layer. For cylinder layers, elliptical
neighborhoods are implemented by rescaling the hit coordinates
before fitting the neighborhood data structures.

The algorithm uses some models derived from training data. These models
are very simple and are represented by per-layer interpolated functions
using RegularGridIterpolator from scipy.interpolate.

The main function is located in `main.py` which calls into
`trackml_solution/supervised.py` for the supervised learning steps
and into `trackml_solution/algorithm.py` for calculating a submission.

See the "Invocation" section for how to invoke the programs from the
command line.

## Dependencies

Main dependencies of the code are:

 - numpy
 - pandas
 - scipy
 - sklearn
 - xgboost (see note below)
 - trackml library

In order to install the [trackml library](https://github.com/LAL/trackml-library)
you can clone it to a local directory and run the following code:

    import pip
    pip.main(['install', 'file://TRACKML_LIBRARY_DIR'])

If you want to run hyperopt, you need to set up a python2 environment for
example like this:

    conda create --name py27 python=2.7 numpy pandas scipy
    conda activate py27
    pip install hyperopt
    pip install networkx==1.11     # to avoid error TypeError: 'generator' object has no attribute '__getitem__'

Note about xgboost: This library was used experimentally during
development of the solution. The final result does not make use of
any xgboost models. It should be straight-forward to remove the
dependency on xgboost from `supervised.py`.

### conda environments

The file `environment-main.yml` specifies the python3 conda environment used to
run `main.py` and generate the solution.

The file `environment-hyperopt.yml` specifies a python2 conda environment
that was created in order to run hyperopt.

## Invocation

A few batch files included in this repository demonstrate how to run the
solution code. The batch files are supposed to be run in the following
sequence:

1. `learn_layer_functions.bat`

   This step learns statistics about the density of hits on the detector
   layers and the average measurement errors of the hit positions from the
   training data.

2. `learn_displacements.bat`

   This step learns the systematic deviations of the particle tracks from
   idealized helical trajectories from the training data.

3. `learn_pairs.bat`

   This step uses the training data to determine good thresholds for finding
   "paired" hits. By "paired hits" we mean the groups of one to four hits
   which are caused by a single particle track in the modules of a single
   detector layer crossed by the track.

4. `generate_submission.bat` SUBMISSION-NAME

    This step calculates a submission from test data.

## License

This software is released as free software under a simple "2-clause BSD license".
See the file LICENSE in this directory for the detailed conditions.

## References

[1]...Bergstra, J., Yamins, D., Cox, D. D. (2013) "Making a Science of Model Search: Hyperparameter Optimization in Hundreds of Dimensions for Vision Architectures."
To appear in Proc. of the 30th International Conference on Machine Learning (ICML 2013).
