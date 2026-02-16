"""Demo Privacy-Preserving Survival Analysis"""

import sys
import logging
import argparse
import numpy as np
import pandas as pd
from mpyc.runtime import mpc
from secure_survival_analysis import model_fit
from secure_survival_analysis import concordance

np.set_printoptions(suppress=True)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--dataset', type=int, metavar='I',
                        help=('dataset 0=btrial(default) 1=waltons 2=aml 3=lung 4=dd'
                              ' 5=stanford_heart_transplants 6=kidney_transplant'))
    parser.add_argument('-a', '--alpha', type=int, metavar='A',
                        help='step size (default 1)')
    parser.add_argument('-l', '--bit-length', type=int, metavar='L',
                        help='override preset bit length for dataset')
    parser.add_argument('-m', '--method', type=int, metavar='M',
                        help='method for optimization 0=gd, 1=bfgs, 2=l-bgfs(default)')
    parser.add_argument('-n', '--samples', type=int, metavar='N',
                        help='number of samples in synthetic data (default=20)')
    parser.add_argument('-t', '--iterations', type=int, metavar='T',
                        help='maximum number of iterations (default=15)')
    parser.set_defaults(dataset=0, bit_length=40, samples=20, method=2, alpha=1, iterations=15)
    args = parser.parse_args()

    num_records = args.samples
    secfxp = mpc.SecFxp(args.bit_length)
    method = {0: 'gd', 1: 'bfgs', 2: 'l-bfgs'}[args.method]

    await mpc.start()

    # Read dataset
    logging.info(f'Reading dataset of {num_records} records')
    synthetic = pd.read_csv('synthetic_hazards10000.csv')[:num_records]
    synthetic.pop('hazards')
    # Move 'time' and 'status' to the front
    synthetic.insert(0, 'time', synthetic.pop('time'))
    synthetic.insert(1, 'status', synthetic.pop('status'))
    synthetic = synthetic.to_numpy()

    # Share dataset with other parties
    logging.info('Sharing dataset among parties')
    synthetic = mpc.input(secfxp.array(synthetic), senders=0)

    # Fit model
    beta, likelihoods, _ = await model_fit.fit_proportional_hazards_model(synthetic, method=method, alpha=args.alpha, num_iterations=args.iterations)
    beta = await mpc.output(beta)
    print('beta: ', beta)
    likelihoods = await mpc.output(likelihoods)
    print('likelihoods: ', likelihoods)

    logging.info(f'Reading dataset of {num_records} records')
    synthetic = pd.read_csv('synthetic_hazards10000.csv')[:num_records]
    # Extract 'time', 'hazards' and 'status' columns
    synthetic_times = synthetic['time'].to_numpy()
    synthetic_hazards = synthetic['hazards'].to_numpy()
    synthetic_status = synthetic['status'].to_numpy()

    # Share dataset with other parties
    logging.info('Sharing dataset among parties')
    times = mpc.input(secfxp.array(synthetic_times), senders=0)
    hazards = mpc.input(secfxp.array(synthetic_hazards), senders=0)
    status = mpc.input(secfxp.array(synthetic_status), senders=0)

    # Compute concordance score
    concordant, tied, comparable = await concordance.harrell_count_pairs(times, hazards, status)
    concordant = await mpc.output(concordant)
    tied = await mpc.output(tied)
    comparable = await mpc.output(comparable)
    # Note: this function can be run on either the secret-shared values, or public values.
    concordance_index = concordance.harrell_concordance_index(concordant, tied, comparable)

    print('########## MPyC Results ##########')
    print('Comparable pairs: ', comparable)
    print('Concordant pairs: ', concordant)
    print('Tied pairs: ', tied)
    print('Concordance index: ', concordance_index)

    await mpc.shutdown()

if __name__ == '__main__':
    mpc.run(main())
