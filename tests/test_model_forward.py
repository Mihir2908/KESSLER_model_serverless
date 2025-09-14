import numpy as np
import unittest
import dsgp4
import pandas as pd
import matplotlib.pyplot as plt

import kessler.model
from kessler import GNSS, Radar

class UtilTestCase(unittest.TestCase):
    def test_forward_and_plot_cdms(self):
        t_tle_list = [
            '0 ELECTRON KICK STAGE R/B',
            '1 44227U 19026C   22068.79876951  .00010731  00000-0  41303-3 0  9993',
            '2 44227  40.0221 252.2030 0008096   5.2961 354.7926 15.26135826158481'
        ]
        c_tle_list = [
            '0 HARBINGER',
            '1 44229U 19026E   22068.90017356  .00004812  00000-0  20383-3 0  9992',
            '2 44229  40.0180 261.5261 0008532 356.1827   3.8908 15.23652474158314'
        ]

        # Create TLE objects
        t_tle = dsgp4.tle.TLE(t_tle_list)
        c_tle = dsgp4.tle.TLE(c_tle_list)

        # Create the Conjunction model
        model = kessler.model.Conjunction(
            t_observing_instruments=[GNSS()],
            c_observing_instruments=[Radar()],
            t_tle=t_tle,
            c_tle=c_tle
        )

        # Run the forward model and get the Pyro trace with CDMs
        print("About to call get_conjunction()")
        trace, iters = model.get_conjunction(max_iters=250)
        if trace is None:
            print(f"No conjunction found after {iters} iterations. Exiting test.")
            return  # or raise an exception, or skip the rest of the test
        cdms = trace.nodes['cdms']['infer']['cdms']
        print("About to call get_conjunction()")

        # Convert CDMs to DataFrame for plotting
        cdm_dicts = [cdm.to_dict() for cdm in cdms]
        df = pd.DataFrame(cdm_dicts)

        # Save the DataFrame to a CSV file
        df.to_csv("synthetic_cdms.csv", index=False)
        print("Synthetic CDMs saved to synthetic_cdms.csv")

        # Example: Get summary statistics
        summary = df.describe()
        print(summary)

        # Generalized: Plot all numeric columns (except TCA) vs TCA
        if 'TCA' in df.columns:
            df['TCA'] = pd.to_datetime(df['TCA'], errors='coerce')
            df = df.sort_values('TCA')
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            for col in numeric_cols:
                if col.upper() != 'TCA':
                    plt.figure(figsize=(8, 5))
                    plt.plot(df['TCA'], df[col], marker='o')
                    plt.xlabel('TCA')
                    plt.ylabel(col)
                    plt.title(f'Synthetic CDMs: {col} vs TCA')
                    plt.grid(True)
                    plt.tight_layout()
                    fname = f'synthetic_cdms_{col.lower()}_vs_tca.pdf'
                    plt.savefig(fname)
                    plt.close()
                    print(f"Plot saved as {fname}")
        else:
            print("TCA not found in CDM DataFrame columns.")

        # Return cdms for programmatic use
        return cdms

if __name__ == "__main__":
    unittest.main()