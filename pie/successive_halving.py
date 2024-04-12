import os
import time
import yaml
import json
from typing import Any

from pie import utils
from pie.settings import settings_from_file, check_settings, merge_task_defaults
from pie.settings import Settings
from pie.optimize import sample_from_config


def successive_halving(train_fn, settings, opt, kwargs):
    HALVING = 3
    ROUNDS = 3
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
