"""
Author: Noah van der Meer
Description: Demo for securely computing the concordance index on a set
    of hazard predictions. Reading and parsing the records is based on
    the Pandas package

By default, this reads the 'synthetic_hazards10000.csv' dataset which
was synthetically generated for testing and validation purposes.


License: MIT License

Copyright (c) 2025, Noah van der Meer
 
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to 
deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or 
sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in 
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.

"""

# MPyC
from mpyc.runtime import mpc

# Numpy & Pandas
import pandas as pd
import numpy as np
np.set_printoptions(suppress=True)

# Python
import logging
import sys

# This repository
from secure_survival_analysis.concordance import harrell_count_pairs, harrell_concordance_index

secfxp = mpc.SecFxp(40, 15)

async def main():
    # Handle arguments
    args = sys.argv
    if len(args) == 1:
        print("Usage: test_concordance <num_values>")
        return 1
    num_values = int(args[1])

    logging.info("Starting MPyC backend")
    await mpc.start()

    # Read dataset
    logging.info("Read dataset")
    #synthetic = pd.read_csv('synthetic_hazards500.csv')[:num_values]
    synthetic = pd.read_csv('synthetic_hazards10000.csv')[:num_values]

    # Extract 'time', 'hazards' and 'status' columns
    synthetic_times = synthetic['time'].to_numpy()
    synthetic_hazards = synthetic['hazards'].to_numpy()
    synthetic_status = synthetic['status'].to_numpy()

    # Share dataset with other parties
    logging.info("Sharing dataset among parties")
    times_mpc = mpc.input(secfxp.array(synthetic_times), senders=0)
    hazards_mpc = mpc.input(secfxp.array(synthetic_hazards), senders=0)
    status_mpc = mpc.input(secfxp.array(synthetic_status), senders=0)

    # Compute concordance score
    concordant, tied, comparable = await harrell_count_pairs(times_mpc, hazards_mpc, status_mpc)

    # Print results
    concordant_open = await mpc.output(concordant)
    tied_open = await mpc.output(tied)
    comparable_open = await mpc.output(comparable)

    # Note: this function can be run on either the secret-shared values, or
    # public values.
    concordance_index = harrell_concordance_index(concordant_open, tied_open, comparable_open)

    print("########## MPyC Results ##########")
    print("Comparable pairs: ", comparable_open)
    print("Concordant pairs: ", concordant_open)
    print("Tied pairs: ", tied_open)

    print("Concordance index: ", concordance_index)

    logging.info("finished computing concordance index through MPyC")


    await mpc.shutdown()

if __name__ == '__main__':
    mpc.run(main())
