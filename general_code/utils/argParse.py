#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Imports
# std libs
import argparse

# 3rd party libs
# ...

# my modules
# ...


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        default="config.json",
        help="The path of configuration json file"
    )
    flags, unparsed = parser.parse_known_args()
    return flags