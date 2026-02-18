"""
Author: Noah van der Meer
Description: Demo for performing the secure exponentiation of fixed-
    point numbers


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

import sys
import time
import math
import logging
from mpyc.numpy import np
from mpyc.runtime import mpc
from secure_survival_analysis import np_pow


async def main():

    # Handle arguments
    args = sys.argv
    if len(args) <= 1:
        print("Usage: test_np_pow <num_values> <optional:bitlength>")
        return 1
    num_values = int(args[1])

    if len(args) >= 3:
        bitlength = int(args[2])
    else:
        bitlength = 32

    logging.info(f"Using bitlength: {bitlength}")
    secfxp = mpc.SecFxp(bitlength, bitlength // 2)

    logging.info(f"Computing exponential function for {num_values} random values")

    await mpc.start()

    # Generate random input values within (-4, 4]
    randoms = 8 * np.random.random(num_values) - 4

    # Let party 0 share these values with the other parties
    v = mpc.input(secfxp.array(randoms), senders=0)

    await mpc.barrier("sharing inputs")
    logging.info("finished sharing inputs")

    base = math.e
    logging.info(f"Base = {base}")

    start = time.time()

    # Compute powers
    ex = np_pow.np_pow(base, v)
    result = await mpc.output(ex)

    await mpc.barrier("computing results")
    logging.info("finished computing results")

    dt = time.time() - start
    logging.info(f"computing results took {dt:.3f} seconds")

    randoms_open = await mpc.output(v)

    # Only print the first 5 numbers to avoid flooding the console
    print("Printing the first 5 results:")
    print_randoms = randoms_open[:5]
    print_results = result[:5]
    print("MPyC pow(", print_randoms, ") = ", print_results)
    print("Python pow(", print_randoms, ") = ", np.pow(base, print_randoms))

    await mpc.shutdown()

if __name__ == '__main__':
    mpc.run(main())
