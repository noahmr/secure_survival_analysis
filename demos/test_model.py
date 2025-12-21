"""
Author: Noah van der Meer
Description: Demo for securely fitting a proportional hazards model on
    a set of records. Reading and parsing the records is based on Pandas

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
import sys
import logging

# This repository
from secure_survival_analysis import model_fit

secfxp = mpc.SecFxp(50, 25)

async def main():
    # Handle arguments
    args = sys.argv
    if len(args) == 1:
        print("Usage: test_model <num_records>")
        return 1
    num_records = int(args[1])

    logging.info("Starting MPyC backend")
    await mpc.start()

    # Read dataset
    logging.info(f"Reading dataset of {num_records} records")
    synthetic = pd.read_csv('synthetic_hazards10000.csv')[:num_records]
    synthetic.pop('hazards')
    # Move 'time' and 'status' to the front
    synthetic.insert(0, 'time', synthetic.pop('time'))
    synthetic.insert(1, 'status', synthetic.pop('status'))

    synthetic_np = synthetic.to_numpy()

    # Share dataset with other parties
    logging.info("Sharing dataset among parties")
    synthetic_shared = mpc.input(secfxp.array(synthetic_np), senders=0)
    
    # Fit model
    beta, likelihoods, _ = await model_fit.fit_proportional_hazards_model(synthetic_shared, method = 'l-bfgs', alpha = 1, num_iterations = 15)

    result = await mpc.output(beta)
    print("beta: ", result)
    result_likelihoods = await mpc.output(likelihoods)
    print("likelihoods: ", result_likelihoods)

    await mpc.shutdown()

if __name__ == '__main__':
    mpc.run(main())
