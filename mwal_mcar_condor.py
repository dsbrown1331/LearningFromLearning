#! /usr/bin/env python


import os
import subprocess
import argparse

# TODO: Set this to be a path to the executable or script you want ran.
# If you need certain envrionment variables set for what you want to run it
# sometimes helps to write a wrapper that exports the variables and then
# runs the desired exectuable.
EXECUTABLE = 'python mcar_mwal_exp.py'

# I use TEST to make sure that the right jobs are being generated before
# launching to condor. When TEST = True, the commands are just printed out.
# It's one way to debug a condor_submit script.
TEST = True


def submitToCondor(param, seed):
    """
    Submit one job with given parameters to condor.

    Args:
        outfile: str, path to write results.
        param: float, some value to test.
        seed: int, seed for executable's RNG.

    Returns:
        None
    """

    # TODO: Make these whatever arguments your executable expects.
    arguments = "{} {}".format(seed, param)

    # Creating submit file for condor
    # See https://www.cs.utexas.edu/facilities/documentation/condor for more
    # details.
    submitFile = 'Executable = ' + EXECUTABLE + "\n"
    submitFile += 'Error = /dev/null\n'
    submitFile += 'Input = /dev/null\nOutput = /dev/null\n'
    submitFile += 'Log = /dev/null\narguments = %s\n' % arguments
    submitFile += '+Group = "GRAD"\n+Project = "AI_ROBOTICS"\n'
    submitFile += '+ProjectDescription = "Learning from demonstrations by learners"\n'
    submitFile += 'Queue'

    if TEST:
        print(EXECUTABLE, arguments)
    else:
        proc = subprocess.Popen('condor_submit', stdin=subprocess.PIPE)
        proc.stdin.write(submitFile)
        proc.stdin.close()


def main():

    parser = argparse.ArgumentParser(description='Launch jobs on Condor')
    parser.add_argument('--num_trials', type=int,
                        help='Number of trials for each condition.')

    args = parser.parse_args()
    
    seeds = [i for i in range(args.num_trials)]

    # TODO: params are the different conditions of your experiment.
    # For example, if you wanted to test different regularization levels on
    # a machine learnings problem you could choose a range of L2 coefficients
    params = [5,10,20,30,40,50]
    ct = 0

    for param in params:

        for seed in seeds:

            submitToCondor(param, seed)
            ct += 1

    print('%d jobs submitted to cluster' % ct)


if __name__ == "__main__":
    main()
