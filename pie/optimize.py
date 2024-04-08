import math
import os
import random
import json
import time
from typing import Any
import traceback

import yaml
from json_minify import json_minify
import scipy.stats as stats
import math
from bayes_opt import BayesianOptimization, UtilityFunction

utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)

from pie import utils
from pie.settings import settings_from_file, check_settings, merge_task_defaults
from pie.settings import Settings

# global flushing
import functools

print = functools.partial(print, flush=True)


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


### TODO
### save all configurations and scores in a json file

# https://arxiv.org/pdf/1801.01596.pdf
# https://arxiv.org/pdf/1603.06560.pdf
# https://medium.com/@sabrinaherbst/automl-with-successive-halfing-and-hyperband-e130a05dded9
def bayesian_hyperband(train_fn, settings, opt, kwargs, R=81, eta=3):
    s_max = math.floor(math.log(R, eta))
    B = (s_max + 1) * R
    result = (0, {})
    bayes = BayesianOptimization(
        f=None,
        pbounds=get_bounds(),
        verbose=2,
        random_state=1,
        allow_duplicate_points=True,
    )

    for s in reversed(range(s_max + 1)):
        n = math.ceil((B / R) * ((eta**s) / (s + 1)))
        r = R * (eta ** (-s))
        print(f"+++ Starting bracket {s} with {n} configurations and {r} resources +++")
        T: list[dict] = []
        loss_threshold = 0
        # successive halving
        for i in range(s + 1):
            n_i = math.floor(n * (eta ** (-i))) if i == 0 else len(T)
            r_i = r * (eta**i) if i == 0 else math.ceil(R / n_i)
            L: list[float] = []

            # 1st iter: we initialize the configurations
            # first bracket: random
            # other brackets: modelled by the GP
            for t in range(n_i):
                if i == 0:
                    config = bayes.suggest(utility)
                    print(f"+++ suggested: {to_pie_config(config)} +++")
                    T.append(config)
                print(f"+++ Bracket {s} (h{i+1}/{s+1}) (r{t+1}/{n_i}) (b{r_i})+++")
                loss = single_run(config, r_i, train_fn, settings, opt, kwargs)
                L.append(loss)
                if i != 0 or s != s_max:
                    print(f"+++ Registering score {L[t]} +++")
                    bayes.register(params=T[t], target=L[t])
            if i == 0 and s == s_max:
                # Only register the configs AFTER we ran them all.
                # Intermediate registration changes the GP, so it won't be random.
                for t in range(n_i):
                    print(f"+++ Registering score {L[t]} +++")
                    bayes.register(params=T[t], target=L[t])

            L, T, loss_threshold = top_k(T, L, math.floor(n_i / eta), loss_threshold)
        if L[0] > result[0]:
            result = (L[0], T[0])
            print(f"+++ New best found in bracket {s}: {result} +++")
    return result


def top_k(T, L, n, loss_threshold) -> tuple[list[float], list[dict], float]:
    if n >= len(T) or n == 0:
        return L, T, loss_threshold
    sorted_pairs = [
        (l, t) for l, t in sorted(zip(L, T), reverse=True, key=lambda x: x[0])
    ][:n]
    print(f"+++ Keeping {n} configurations +++")
    # filter out the pairs that are below the average threshold
    loss_threshold = (
        (loss_threshold + max(L)) / 2 if loss_threshold != 0 else max(L) + min(L) / 2
    )
    print(f"+++ Loss threshold: {loss_threshold} +++")
    good_pairs = [pair for pair in sorted_pairs if pair[0] >= loss_threshold]
    print(f"+++ Keeping best {len(good_pairs)} configurations +++")
    new_L = [l for l, _ in good_pairs]
    new_T = [t for _, t in good_pairs]
    avg_loss = sum(new_L) / len(new_L)
    return new_L, new_T, avg_loss


def single_run(t, r_i, train_fn, settings, opt, kwargs) -> float:
    try:
        pie_config = to_pie_config(t)
        merged = Settings(
            utils.recursive_merge(dict(settings), pie_config, overwrite=True)
        )
        merged.epochs = int(r_i)
        merged.modelpath = ""
        path, loss = train_fn(check_settings(merge_task_defaults(merged)), **kwargs)
        final_score = -loss["lemma"]  # negative because we want to maximize
        return final_score
    except Exception as e:
        print(f"+++ Exception in single_run: {e} +++\n{traceback.format_exc()}")
        return 0


def to_pie_config(t):
    pie_config = {}
    # set pie_config.default to zero
    pie_config["include_lm"] = to_bool(t["include_lm"])
    pie_config["lm_shared_softmax"] = to_bool(t["lm_shared_softmax"])

    pie_config["lm_schedule"] = {}
    pie_config["lm_schedule"]["patience"] = to_int(t["lm_patience"])
    pie_config["lm_schedule"]["factor"] = t["lm_factor"]
    pie_config["lm_schedule"]["weight"] = t["lm_weight"]

    pie_config["batch_size"] = to_int(t["batch_size"])
    pie_config["pretrain_embeddings"] = to_bool(t["pretrain_embeddings"])
    pie_config["freeze_embeddings"] = to_bool(t["freeze_embeddings"])
    pie_config["dropout"] = t["dropout"]
    pie_config["word_dropout"] = t["word_dropout"]
    pie_config["optimizer"] = to_choice(t["optimizer"], ["Adadelta","Adagrad","Adam","AdamW","Adamax","ASGD","SGD","RAdam","Rprop","RMSprop","NAdam"])
    pie_config["clip_norm"] = t["clip_norm"]
    pie_config["lr"] = t["lr"]
    pie_config["lr_factor"] = t["lr_factor"]
    pie_config["min_lr"] = t["min_lr"]
    pie_config["lr_patience"] = to_int(t["lr_patience"])
    pie_config["wemb_dim"] = to_int(t["wemb_dim"])
    pie_config["cemb_dim"] = to_int(t["cemb_dim"])
    pie_config["cemb_type"] = to_choice(t["cemb_type"], ["rnn", "cnn"])
    pie_config["custom_cemb_cell"] = to_bool(t["custom_cemb_cell"])
    pie_config["cemb_layers"] = to_int(t["cemb_layers"])
    pie_config["scorer"] = to_choice(t["scorer"], ["general", "dot", "bahdanau"])
    pie_config["linear_layers"] = to_int(t["linear_layers"])
    pie_config["hidden_size"] = to_int(t["hidden_size"])
    pie_config["num_layers"] = to_int(t["num_layers"])
    pie_config["cell"] = to_choice(t["cell"], ["LSTM", "GRU"])
    pie_config["init_rnn"] = to_choice(t["init_rnn"], ["xavier_uniform", "orthogonal"])
    return pie_config


def to_choice(value, options):
    return options[math.floor(value)]


def to_int(value):
    return math.floor(value)


def to_bool(value):
    return bool(math.floor(value))


def get_bounds():
    return {  # Note, for bool, choice and int, add 1 to the upper bound
        "include_lm": (0, 2),  # bool
        "lm_shared_softmax": (0, 2),  # bool

        "lm_patience": (1, 5),  # int
        "lm_factor": (0, 1),  # float
        "lm_weight": (0, 1),  # float

        "batch_size": (10, 100),  # int
        "pretrain_embeddings": (0, 2),  # bool
        "freeze_embeddings": (0, 2),  # bool
        "dropout": (0, 1),  # float
        "word_dropout": (0, 1),  # float
        "optimizer": (0, 11),  # choice
        "clip_norm": (0, 10),  # float
        "lr": (0, 0.1),  # float
        "lr_factor": (0, 1),  # float
        "min_lr": (0, 0.0001),  # float
        "lr_patience": (0, 5),  # int
        "wemb_dim": (50, 500),  # int
        "cemb_dim": (50, 500),  # int
        "cemb_type": (0, 2),  # choice
        "custom_cemb_cell": (0, 2),  # bool
        "cemb_layers": (1, 5),  # int
        "scorer": (0, 3),  # choice
        "linear_layers": (1, 4),  # int
        "hidden_size": (10, 500),  # int
        "num_layers": (1, 4),  # int
        "cell": (0, 2),  # choice
        "init_rnn": (0, 2),  # choice
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
