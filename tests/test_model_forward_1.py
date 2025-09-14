import numpy as np
import unittest
import dsgp4
import pandas as pd
import matplotlib.pyplot as plt
import kessler
import pyro
import os

import kessler.model
from kessler import GNSS, Radar

class UtilTestCase(unittest.TestCase):
    def test_forward_and_plot_cdms(self):

        #we seed everything for reproducibility
        pyro.set_rng_seed(10)

        #we define the observing instruments
        #GNSS first:
        gnss_cov_rtn=np.array([1e-9, 1.115849341564346, 0.059309835843067, 1e-9, 1e-9, 1e-9])**2,
        instrument_characteristics_gnss={'bias_xyz': np.array([[0., 0., 0.],
            [0., 0., 0.]]), 'covariance_rtn': gnss_cov_rtn}
        gnss = kessler.GNSS(instrument_characteristics_gnss)

        #and then radar:
        radar_cov_rtn=np.array([1.9628939405514678, 2.2307686944695706, 0.9660907831563862, 1e-9, 1e-9, 1e-9])**2
        instrument_characteristics_radar={'bias_xyz': np.array([[0., 0., 0.],
            [0., 0., 0.]]), 'covariance_rtn': radar_cov_rtn}
        radar = kessler.Radar(instrument_characteristics_radar)

        tles_path = os.path.join(os.path.dirname(__file__), "tles_sample_population.txt")
        tles=dsgp4.tle.load(tles_path)

        # Create the Conjunction model
        conjunction_model = kessler.model.ConjunctionSimplified(time0=60727.13899462018,
                                                        # max_duration_days=7.0,
                                                        # time_resolution=600000.0,
                                                        # time_upsample_factor=100,
                                                        miss_dist_threshold=5000.0,
                                                        prior_dict=None,
                                                        t_prob_new_obs=0.96,
                                                        c_prob_new_obs=0.4,
                                                        cdm_update_every_hours=8.0,
                                                        mc_samples=100,
                                                        mc_upsample_factor=100,
                                                        pc_method='MC',
                                                        collision_threshold=70,
                                                        # likelihood_t_stddev=[371.068006, 0.0999999999, 0.172560879],
                                                        # likelihood_c_stddev=[371.068006, 0.0999999999, 0.172560879],
                                                        likelihood_time_to_tca_stddev=0.7,
                                                        t_observing_instruments=[gnss],
                                                        c_observing_instruments=[radar],
                                                        tles=tles,)

        # Run the forward model and get the Pyro trace with CDMs
        print("About to call get_conjunction()")
        trace, iters = conjunction_model.get_conjunction()
        if trace is None:
            print(f"No conjunction found after {iters} iterations. Exiting test.")
            return  # or raise an exception, or skip the rest of the test
        cdms_1 = trace.nodes['cdms']['infer']['cdms']
        print("About to call get_conjunction()")

        # Convert CDMs to DataFrame for plotting
        cdm_dicts = [cdm.to_dict() for cdm in cdms_1]
        df = pd.DataFrame(cdm_dicts)

        # Save the DataFrame to a CSV file
        df.to_csv("synthetic_cdms_1.csv", index=False)
        print("Synthetic CDMs saved to synthetic_cdms_1.csv")

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
        return cdms_1

if __name__ == "__main__":
    unittest.main()