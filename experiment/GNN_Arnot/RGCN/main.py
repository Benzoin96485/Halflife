#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Imports
# std libs
# ...

# 3rd party libs
# ...

# my modules
from general_code.utils.argParse import parseArgs
from general_code.utils.log import Logger
from general_code.config.makeConfig import Config


def main():
    config = Config(parseArgs())
    logger = Logger(config.data["label_names"])
    for time_id in range(config["eval_times"]):
        seed_this_time = time_id + config.run["seed_init"]
        logger.start()
        split_data = 


if __name__ == "__main__":
    main()

