import unittest
import datetime
import matplotlib.pyplot as plt
from kessler.nn import LSTMPredictor
from kessler.event import Event, EventDataset

class TestNNTrainAndPredictCDMs(unittest.TestCase):
    def test_train_and_predict_future_cdms(self):
        # Import synthetic CDMs from test_model_forward_1
        import sys, os
        sys.path.append(os.path.dirname(__file__))
        from test_model_forward_1 import UtilTestCase as ForwardTestCase

        # Run the forward modeling test to get CDMs
        forward_test = ForwardTestCase()
        cdms_many = forward_test.test_forward_and_plot_cdms()
        # You may need to refactor test_model_forward to return cdms or save them to a file for easier access

        # For this example, let's assume you can access the CDMs as a list of dicts
        # (If not, you may need to refactor test_model_forward to expose them)
        # Here, we simulate loading them:
        import pandas as pd
    
        # For demonstration, let's load from a CSV if you saved it in test_model_forward
        if os.path.exists("synthetic_cdms_many.csv"):
            df = pd.read_csv("synthetic_cdms_many.csv")
            df = df.dropna(subset=['TCA'])
            # Remove rows where TCA is NaN, None, or empty string
            df = df[df['TCA'].notnull() & (df['TCA'] != '') & (df['TCA'] != None)]
            print(df.columns)
            #cdms_1 = df.to_dict(orient="records")
            #cdms_1 = []  # Replace with actual list of CDMs from forward_test
        else:
            self.skipTest("No synthetic CDMs available for training.")

        if 'EVENT_ID' in df.columns:
            df = df.rename(columns={'EVENT_ID': 'event_id'})

        df['TCA'] = df['TCA'].astype(str) #correction 1
        print("TCA after string conversion:", df['TCA'].unique()) #correction 2
        print("DataFrame columns:", df.columns.tolist()) # correction 3

        # Try creating a single CDM manually to see if TCA gets preserved - Correction 4
        first_row = df.iloc[0].to_dict()
        print("First row TCA from dict:", first_row.get('TCA'))

        
        # Check if the CDM creation process preserves TCA - Correction 5
        try:
            from kessler.cdm import ConjunctionDataMessage
            test_cdm = ConjunctionDataMessage()
            # See what happens when we try to set TCA
            test_cdm.set_header('TCA', first_row.get('TCA'))
            print("Test CDM TCA:", test_cdm['TCA'])
            print(type(test_cdm['TCA']))
        except Exception as e:
            print("Error creating test CDM:", e)

        # Wrap as a single Event and EventDataset
        #event_u = Event([cdm for cdm in cdms_1])
        #event_set_u = EventDataset(events=[event_u])
        
        # Fix TCA format to match CDM expectations : Correction
        def standardize_tca_format(tca_value):
            if pd.isna(tca_value) or tca_value is None:
                return None
            try:
                # Parse the datetime and reformat to standard microsecond precision
                dt = datetime.datetime.fromisoformat(str(tca_value))
                return dt.strftime('%Y-%m-%dT%H:%M:%S.%f')
            except:
                return str(tca_value)

        df['TCA'] = df['TCA'].apply(standardize_tca_format)
        df['TCA type'] = type(df['TCA'])
        print(df.head())  
        event_set_u = EventDataset.from_pandas(df, group_events_by='event_id')

        n_events = len(event_set_u)

        if n_events < 3:
            self.skipTest(f"Not enough events for training (got {n_events})")

        len_test_set_u = int(0.05 * n_events)
        if len_test_set_u <= 0:
            len_test_set_u = 1
        if len_test_set_u >= n_events:
            len_test_set_u = 1

        nn_features_u=event_set_u.common_features(only_numeric=True)
        print("Total events:", len(event_set_u))

        #len_test_set_u=int(0.05*len(event_set_u))
        events_test_u=event_set_u[-len_test_set_u:]
        events_train_and_val_u=event_set_u[:-len_test_set_u]

        # Initialize the LSTM model
        model = LSTMPredictor()

        # Train the model
        model.learn(events_train_and_val_u, epochs=2, lr=1e-3, batch_size=2, device='cpu', valid_proportion=0.15, num_workers=4, event_samples_for_stats=1000)

        # Save the trained model
        model.save("lstm_model_trained.pt")

        event_u = events_test_u[2] 
        event_len_u=len(event_u)
        event_beginning_u=event_u[0:event_len_u-1]

        # Predict future CDMs using the trained model
        predicted_event_u = model.predict_event(event_beginning_u, num_samples=1, max_length=22)
        df_pred_u = predicted_event_u.to_dataframe()
        print(df_pred_u)

        # Optionally, plot a feature
        predicted_event_u.plot_feature('OBJECT1_CT_T', file_name='predicted_OBJECT1_CT_T_vs_TCA.pdf')
        print("Plot saved as predicted_OBJECT1_CT_T_vs_TCA.pdf")

        axs=predicted_event_u.plot_features(['RELATIVE_SPEED', 'MISS_DISTANCE'], return_axs=True, linewidth=0.1, color='red', alpha=0.33, label='Prediction')
        predicted_event_u.plot_features(['RELATIVE_SPEED', 'MISS_DISTANCE'], axs=axs, label='Real', legend=True)

        plt.savefig('prediction_vs_real_features_fromTLES.pdf')
        plt.close()


        self.assertTrue(len(df_pred_u) > 0)

if __name__ == "__main__":
    unittest.main()
