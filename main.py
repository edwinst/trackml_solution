"""Main program for invoking the trackml solution."""

import sys
import os
import argparse
import pprint
import numpy as np
from collections import OrderedDict

from trackml_solution.geometry import DetectorSpec
from trackml_solution.data import Event
from trackml_solution.algorithm import Algorithm
from trackml_solution.supervised import Supervisor, LayerFunctions

default_path_to_data = 'c:/Users/edwin/nobackup/kaggle-trackml-data'
default_path_to_train_relative = 'train_100_events'

parser = argparse.ArgumentParser(description='Solve kaggle trackml-particle-identification.')
parser.add_argument('--data-dir', help='path to data files (dir which contains detectors.csv)',
                                  default=default_path_to_data)
parser.add_argument('--event-dir', help='path to event CSV files (dir which contains event*.csv)')
parser.add_argument('--params-file', help='load hyper-parameters from the given file (later ones have priority)',
                                     action='append')
parser.add_argument('-p', help='set a parameter to a given value (becomes effective in order, after param files)',
                          action='append', metavar="PARAM=VALUE", default=[])
parser.add_argument('--results-file', help='write result dictionary (for hyperopt) to the given file')
parser.add_argument('--submission-file', help='write submission CSV to the given file')
parser.add_argument('--analysis', help='store data for off-line analysis', action='store_true', default=False)
parser.add_argument('--score-intermediate', action='store_true', default=False)
parser.add_argument('--log', help='set maxmium log level', type=int, default=1)
parser.add_argument('--beg', help='event_id of first event to load', type=int, default=1000)
parser.add_argument('--end', help='event_id of last event to load (default: load only one)', type=int, default=None)
parser.add_argument('--events', help='comma-separated list of events to load', type=str, default=None)
parser.add_argument('--with-cells', help='use cells data', action='store_true', default=False)
parser.add_argument('--layer-functions', help='prefix for loading layer functions', action='append')
parser.add_argument('--learn-layer-functions', action='store_true', default=False)
parser.add_argument('--learn-tracks', action='store_true', default=False)
parser.add_argument('--load-models', type=str, default=None, help='prefix for loading models')
parser.add_argument('--save-models', type=str, default=None, help='prefix for saving models')
parser.add_argument('--save-displacements', type=str, default=None, help='prefix for saving displacement layer functions')
parser.add_argument('--save-pairs', type=str, default=None, help='prefix for saving pairing layer functions')
parser.add_argument('--save-analytics', type=str, default=None, help='path for saving CSV for off-line analysis')

args = parser.parse_args()

if args.event_dir is None:
    args.event_dir = os.path.join(args.data_dir, default_path_to_train_relative)

# load detector specification
spec = DetectorSpec.from_detectors_csv(os.path.join(args.data_dir, 'detectors.csv'))

beg = args.beg
end = args.end
if end is None:
    end = beg
event_id_sel = range(beg, end+1)

if args.events is not None:
    event_id_sel = list(map(int, args.events.split(',')))

events = [Event('event%09d' % i, path=args.event_dir, with_truth='auto', with_cells=args.with_cells) for i in event_id_sel]

# load hyper-parameters from given files and command line options
params = OrderedDict()
if args.params_file is not None:
    for filename in args.params_file:
        with open(filename) as file:
            read_params = eval(file.read())
            params.update(read_params)
for p in args.p:
    kv = p.split('=')
    if len(kv) != 2:
        raise RuntimeError("Could not parse -p PARAM=VALUE argument '" + p + "'")
    params[kv[0]] = eval(kv[1])

layer_functions = None
if args.layer_functions is not None:
    layer_functions = LayerFunctions.from_csv(args.layer_functions)

alg = Algorithm(spec, max_log_indent=args.log, params=params, layer_functions=layer_functions)
supervisor = Supervisor(spec, params=params)

if args.load_models is not None:
    supervisor.loadModels(prefix=args.load_models)

if args.learn_layer_functions:
    layer_functions = supervisor.learnLayerFunctions(events_train=events)
    layer_functions.to_csv('layer_functions')
    sys.exit(0)

if args.learn_tracks:
    supervisor.learnTracks(alg, events_train=events,
        save_models_prefix=args.save_models,
        save_displacements_prefix=args.save_displacements,
        save_pairs_prefix=args.save_pairs,
        save_analytics_path=args.save_analytics)
    sys.exit(0)

scores = alg.findTracks(supervisor=supervisor,
                        events_test=events,
                        analysis=args.analysis,
                        score_intermediate=args.score_intermediate, score_final=True,
                        submission_filename=args.submission_file)

print("score mean %.4f (std %.4f)" % (np.mean(scores), np.std(scores)))

if args.results_file is not None:
    with open(args.results_file, 'w') as file:
        results = {
            'status'        : 'ok', # hyperopt.STATUS_OK (not imported to minimize dependencies)
            'loss'          : -np.mean(scores),
            'loss_variance' : np.std(scores), # XXX should this be squared std? Could not find documentation on it.
            'scores'        : scores,
        }
        pprint.pprint(results, stream=file)
