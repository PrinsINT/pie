
import os
import random
import json
from typing import Any

import yaml
from json_minify import json_minify
import scipy.stats as stats

from pie import utils
from pie.settings import settings_from_file, check_settings, merge_task_defaults
from pie.settings import Settings


# available distributions
class truncnorm:
    def __init__(self, mu, std, lower=0, upper=1):
        a, b = (lower - mu) / std, (upper - mu) / std
        self.norm = stats.truncnorm(a, b, mu, std)

    def rvs(self):
        return float(self.norm.rvs())


class normint:
    def __init__(self, mu, std, lower, upper):
        self.norm = truncnorm(mu, std, lower, upper)

    def rvs(self):
        return int(round(self.norm.rvs())) // 2 * 2


class choice:
    def __init__(self, items):
        self.items = items

    def rvs(self):
        return random.choice(self.items)


def parse_opt(obj, opt_key):
    """
    Parses the opt file into a (possibly deep) dictionary where the leaves are 
    ready-to-use distributions
    """
    opt = {}

    for param, v in obj.items():
        if isinstance(v, list):
            opt[param] = [parse_opt(v_item, opt_key) for v_item in v]
        elif isinstance(v, dict):
            if opt_key in v:
                if v[opt_key] == 'norm':
                    opt[param] = stats.norm(**v['params'])
                elif v[opt_key] == 'truncnorm':
                    opt[param] = truncnorm(**v['params'])
                elif v[opt_key] == 'normint':
                    opt[param] = normint(**v['params'])
                elif v[opt_key] == 'choice':
                    opt[param] = choice(v['params'])
                else:
                    raise ValueError("Unknown distribution: ", v[opt_key])
            else:
                opt[param] = parse_opt(v, opt_key)
        else:
            opt[param] = v

    return opt


def read_opt(path, opt_key='opt'):
    """
    Reads and parses the opt file (as per parse_opt)
    """
    with open(path) as f:
        obj = json.loads(json_minify(f.read()))

    return parse_opt(obj, opt_key)


def sample_from_config(opt):
    """
    Applies the distributions specified in the opt.json file
    """
    output = {}

    for param, dist in opt.items():
        if isinstance(dist, dict):
            output[param] = sample_from_config(dist)
        elif isinstance(dist, list):
            output[param] = [sample_from_config(d) for d in dist]
        elif isinstance(dist, (str, float, int, bool)):
            output[param] = dist  # no sampling
        else:
            output[param] = dist.rvs()

    return output


def run_optimize(train_fn, settings, opt, n_iter, **kwargs):
    """
    Run random search over given `settings` resampling parameters as
    specified by `opt` for `n_iter` using `train_fn` function.

    - train_fn: a function that takes settings and any other possible kwargs
        and runs a training procedure
    - settings: a Settings object fully determining a training run
    - opt: a sampling file specifying parameters to resample each run,
        including a distribution to sample from. The contents are read from
        a json file with the following structure.
        { "lr": {
            "opt": "truncnorm",
            "params": {
                "mu": 0.0025, "std": 0.002, "lower": 0.0001, "upper": 1
                }
            }
        }
        "opt" specifies the distribution, and "params" the required parameters
        for that distribution:
            - "truncnorm": truncated normal
               - params: mu, std, lower, upper
            - "choice": uniform over given options
               - params: list of options
            - "normint": same as "truncnorm" but output is round up to an integer

        Other distributions can be implemented in the future.

    - n_iter: int, number of iterations to run
    """
    HALVING = 3
    ROUNDS = 3
    INIT_POP = ROUNDS * HALVING
    model_pool: list[tuple[Any, float, str]] = []
    for _ in range(INIT_POP):
        tmp = sample_from_config(opt)
        model_pool.append((tmp, 0.0, ""))
    generation = 0

    while len(model_pool) > 0:
        for i, model in enumerate(model_pool):
            resources = HALVING ** generation
            print()
            print(f"::: Model pool of size: {len(model_pool)}, generation: {generation}, resources per model: {resources} :::")
            print()
            print(f"::: Starting optimization run {i+1} of generation {generation} :::")
            print()
            sampled = model[0]
            merged = Settings(utils.recursive_merge(dict(settings), sampled, overwrite=True))
            merged.epochs = resources
            if model[2] != "":
                merged.existing_model = model[2]
            print("::: Sampled settings :::")
            print(yaml.dump(dict(merged)))
            model_path, scoring = train_fn(check_settings(merge_task_defaults(merged)), **kwargs)
            model_pool[i] = (model[0], scoring[0]['all']['accuracy'], model_path)
        
        print(f"::: Best score in generation {generation}: {model_pool[0][1]} :::")

        if len(model_pool) == 1:
            break
        model_pool = sorted(model_pool, key=lambda x: x[1], reverse=True)
        to_be_removed = model_pool[len(model_pool) // HALVING:]
        print(f"Removing {len(to_be_removed)} models")
        for model in to_be_removed:
            os.remove(model[2])
        model_pool = model_pool[:len(model_pool) // HALVING]
        generation += 1


if __name__ == '__main__':
    from pie.settings import settings_from_file
    settings = settings_from_file("./transformer-lemma.json")
    opt = read_opt("opt-transformer.json")
    for _ in range(10):
        sampled = sample_from_config(opt)
        d = Settings(utils.recursive_merge(dict(settings), sampled, overwrite=True))
        for k in opt:
            print(k, d[k])
            print()
