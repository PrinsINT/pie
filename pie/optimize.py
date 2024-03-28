import math
import os
import random
import json
import time
from typing import Any

import yaml
from json_minify import json_minify
import scipy.stats as stats
import math
from bayes_opt import BayesianOptimization, UtilityFunction

utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)

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
                if v[opt_key] == "norm":
                    opt[param] = stats.norm(**v["params"])
                elif v[opt_key] == "truncnorm":
                    opt[param] = truncnorm(**v["params"])
                elif v[opt_key] == "normint":
                    opt[param] = normint(**v["params"])
                elif v[opt_key] == "choice":
                    opt[param] = choice(v["params"])
                else:
                    raise ValueError("Unknown distribution: ", v[opt_key])
            else:
                opt[param] = parse_opt(v, opt_key)
        else:
            opt[param] = v

    return opt


def read_opt(path, opt_key="opt"):
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
    bayesian_hyperband(train_fn, settings, opt, kwargs)


# https://arxiv.org/pdf/1801.01596.pdf
# https://medium.com/@sabrinaherbst/automl-with-successive-halfing-and-hyperband-e130a05dded9
def bayesian_hyperband(train_fn, settings, opt, kwargs, R=9, eta=3):
    s_max = math.floor(math.log(R, eta))
    B = (s_max + 1) * R
    result = (0, {})
    bayes = [
        BayesianOptimization(
            f=None,
            pbounds=get_bounds(),
            verbose=2,
            random_state=1,
        )
        for _ in range(s_max + 1)
    ]

    for s in reversed(range(s_max + 1)):
        n = math.ceil((B / R) * ((eta**s) / (s + 1)))
        r = R * (eta ** (-s))
        print(f"+++ Starting bracket {s} with {n} configurations and {r} resources +++")

        # successive halving
        for i in range(s + 1):
            n_i = math.floor(n * (eta ** (-i)))
            r_i = r * (eta**i)
            L: list[float] = []

            if i == 0:
                T: list[dict] = []
                for t in range(n_i):
                    config = bayes[i].suggest(utility)
                    print(f"+++ suggested: {to_pie_config(config)} +++")
                    print(
                        f"+++ Run {t+1} of {n_i} with {r_i} resources in bracket {s}, halving {i+1}/{s+1} +++"
                    )
                    loss = single_run(config, r_i, train_fn, settings, opt, kwargs)
                    L.append(loss)
                    T.append(config)
                    bayes[i].register(params=config, target=loss)
            else:
                for t_i, t in enumerate(T):
                    print(
                        f"+++ Run {t_i+1} of {n_i} with {r_i} resources in bracket {s}, halving {i+1}/{s+1} +++"
                    )
                    loss = single_run(t, r_i, train_fn, settings, opt, kwargs)
                    L.append(loss)
                    bayes[i].register(params=t, target=loss)
            L, T = top_k(T, L, math.floor(n_i / eta))
        if L[0] > result[0]:
            result = (L[0], T[0])
            print(f"+++ New best found in bracket {s}: {result} +++")
    return result


def top_k(T, L, n) -> tuple[list[float], list[dict]]:
    if n >= len(T) or n == 0:
        return L, T
    print(f"+++ Keeping best {n} configurations +++")
    sorted_pairs = [
        (l, t) for l, t in sorted(zip(L, T), reverse=True, key=lambda x: x[0])
    ][:n]
    return [l for l, _ in sorted_pairs], [t for _, t in sorted_pairs]


def single_run(t, r_i, train_fn, settings, opt, kwargs) -> float:
    try:
        pie_config = to_pie_config(t)
        merged = Settings(
            utils.recursive_merge(dict(settings), pie_config, overwrite=True)
        )
        merged.epochs = int(r_i)
        merged.modelpath = ""
        _, scoring = train_fn(check_settings(merge_task_defaults(merged)), **kwargs)
        final_score = scoring[0]["all"]["accuracy"]
        return final_score
    except Exception as e:
        print(f"+++ Exception in single_run: {e} +++")
        return 0


def to_pie_config(t):
    pie_config = {}
    # set pie_config.default to zero
    pie_config["init_rnn"] = multi_choice(
        t["init_rnn"], ["xavier_uniform", "orthogonal"]
    )
    pie_config["cell"] = multi_choice(t["cell"], ["LSTM", "GRU"])
    pie_config["scorer"] = multi_choice(t["scorer"], ["general", "dot", "bahdanau"])
    pie_config["hidden_size"] = pie_int(t["hidden_size"])
    pie_config["num_layers"] = pie_int(t["num_layers"])
    pie_config["linear_layers"] = pie_int(t["linear_layers"])
    pie_config["cemb_layers"] = pie_int(t["cemb_layers"])
    return pie_config


def multi_choice(value, options):
    return options[math.floor(value)]


def pie_int(value):
    return math.floor(value)

def pie_bool(value):
    return bool(math.floor(value))

def get_bounds():
    return {  # Note, for bool, choice and int, add 1 to the upper bound
        "init_rnn": (0, 2),  # choice
        "cell": (0, 2),  # choice
        "scorer": (0, 3),  # choice
        "hidden_size": (10, 500),
        "num_layers": (1, 4),  # int
        "linear_layers": (1, 4),  # int
        "cemb_layers": (1, 5),  # int
    }


def successive_halving(train_fn, settings, opt, kwargs):
    HALVING = 3
    ROUNDS = 4
    INIT_POP = HALVING**ROUNDS
    THRESHOLD_SCORE = 0.1
    START_TIME = int(time.time())
    model_pool: list[tuple[Any, float, str, int]] = []
    for i in range(INIT_POP):
        tmp = sample_from_config(opt)
        model_pool.append((tmp, 0, "", i))
    generation = 0

    while len(model_pool) > 0:
        for i, model in enumerate(model_pool):
            resources = HALVING**generation
            print()
            print(
                f"::: Model pool of size: {len(model_pool)}, generation: {generation+1}, resources per model: {resources} :::"
            )
            print()
            print(f"::: Starting run {i+1} of generation {generation+1} :::")
            print()
            sampled = model[0]
            merged = Settings(
                utils.recursive_merge(dict(settings), sampled, overwrite=True)
            )
            merged.epochs = resources
            if model[2] != "":
                merged.existing_model = model[2]
            print("::: Sampled settings :::")
            print(yaml.dump(dict(merged)))
            try:
                model_path, scoring = train_fn(
                    check_settings(merge_task_defaults(merged)), **kwargs
                )
                final_score = scoring[0]["all"]["accuracy"]

                # Remove model from the previous generation
                if model[2] != "":
                    if os.path.exists(model[2]):
                        os.remove(model[2])

                if final_score < THRESHOLD_SCORE:
                    print(f"::: Score too low, removing model :::")
                    # remove the model file if the score is too low
                    # to make space for new models instead of waiting for the halving
                    if os.path.exists(model_path):
                        os.remove(model_path)
                    model_pool[i] = (model[0], 0, "", model[3])
                else:
                    model_pool[i] = (model[0], final_score, model_path, model[3])

            except Exception as e:
                print(
                    f"::: Exception in run {i+1} of generation {generation+1}, continuing anyway :::\n{e}"
                )

        # sort by score
        model_pool = sorted(model_pool, key=lambda x: x[1], reverse=True)

        # report on model pool in json file
        json_path = os.path.join(
            settings.modelpath, f"models-gen{generation+1}-{START_TIME}.json"
        )
        with open(json_path, "a+") as f:
            json.dump(model_pool, f)

        if len(model_pool) == 1:
            break

        print(f"::: Best score in generation {generation+1}: {model_pool[0][1]} :::")
        generation += 1
        one_third = HALVING ** (ROUNDS - generation)
        if one_third > len(model_pool):
            one_third = len(model_pool)
        # remove worst half
        to_be_removed = model_pool[one_third:]
        # keep best half
        model_pool = model_pool[:one_third]
        # additionally remove any models from the model_pool whose score is near zero
        to_be_removed += [model for model in model_pool if model[1] <= THRESHOLD_SCORE]
        model_pool = [model for model in model_pool if model[1] > THRESHOLD_SCORE]
        # actually remove the models files
        print(f"Removing {len(to_be_removed)} models")
        for model in to_be_removed:
            if (
                model[2] != ""
            ):  # due to an exception, the model might not have been saved
                if os.path.exists(model[2]):
                    os.remove(model[2])


if __name__ == "__main__":
    from pie.settings import settings_from_file

    settings = settings_from_file("./transformer-lemma.json")
    opt = read_opt("opt-transformer.json")
    for _ in range(10):
        sampled = sample_from_config(opt)
        d = Settings(utils.recursive_merge(dict(settings), sampled, overwrite=True))
        for k in opt:
            print(k, d[k])
            print()
