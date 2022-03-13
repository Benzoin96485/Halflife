#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Define argument parsing functions.
'''

# Imports
# std libs
import argparse

# 3rd party libs
# ...

# my modules
# ...


def parseArgs():
    """
    Parse the configurations of the experiment from the command line
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        default="config.json",
        help="The path of the configuration file"
    )
    flags, unparsed = parser.parse_known_args()
    return flags