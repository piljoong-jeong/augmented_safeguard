import os
import sys

import augmented_safeguard as asfgd

def main():

    

    # return asfgd.app.run_default()
    return asfgd.app.run_incremental_SVD()

if __name__ == "__main__":
    main()