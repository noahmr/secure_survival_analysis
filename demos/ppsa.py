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
    parser.add_argument('-s', '--sort_column', type=int, metavar='S',
                        help='index of time column, default 0, -1 for no secure sort')
    parser.add_argument('--concordance_index', action='store_true',
                        help='perform concordance test')
    parser.set_defaults(dataset=0, bit_length=40, samples=20, method=2, alpha=1, iterations=15, sort_column=0)
    args = parser.parse_args()

    num_records = args.samples
    secfxp = mpc.SecFxp(args.bit_length)
    method = {0: 'gd', 1: 'bfgs', 2: 'l-bfgs'}[args.method]

    await mpc.start()

    if num_records <= 10000:
        # Read dataset
        logging.info(f'Reading dataset of {num_records} records')
        synthetic = pd.read_csv('synthetic_hazards10000.csv')[:num_records]
        synthetic.pop('hazards')
        # Move 'time' and 'status' to the front
        synthetic.insert(0, 'time', synthetic.pop('time'))
        synthetic.insert(1, 'status', synthetic.pop('status'))
        synthetic = synthetic.to_numpy()
    else:
        from generate_data import generate_survival_sample
        beta = [0.02, 0.9, 0.6, 0.03, 0.8]

        # The values 30, 2, 50 and 10 are the parameters associated with the Weibull and
        # exponential distribution. These were chosen as to generate datasets with a
        # significant number of censored subjects, along with introducing many tied event
        # times in order to demonstrate the accuracy of the methods in handling these.
        #
        # Different values can be chosen to simulate different settings.
        synthetic, t, status = generate_survival_sample(30, 2, 50, 10, num_records, beta)
        synthetic = np.column_stack((t, status, synthetic))

    if args.sort_column == -1:
        logging.info(f'Sorting in the clear')
        synthetic = pd.DataFrame(synthetic)
        synthetic = synthetic.sort_values(by=[0])
        synthetic = synthetic.to_numpy()

    # Share dataset with other parties
    logging.info('Sharing dataset among parties')
    synthetic = mpc.input(secfxp.array(synthetic), senders=0)

    # Fit model
    beta, likelihoods, _ = await model_fit.fit_proportional_hazards_model(synthetic, method=method, alpha=args.alpha, num_iterations=args.iterations, sort_column=args.sort_column)
    beta = await mpc.output(beta)
    print('beta: ', beta)
    likelihoods = await mpc.output(likelihoods)
    print('likelihoods: ', likelihoods)

    await mpc.shutdown()

    if num_records > 10000 or not args.concordance_index:
        return 
        
    await mpc.start()
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
