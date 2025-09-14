import unittest
from kessler.nn import LSTMPredictor
from kessler.event import Event, EventDataset

class TestNNTrainAndPredictCDMs(unittest.TestCase):
    def test_train_and_predict_future_cdms(self):
        # Import synthetic CDMs from test_model_forward
        import sys, os
        sys.path.append(os.path.dirname(__file__))
        from test_model_forward import UtilTestCase as ForwardTestCase

        # Run the forward modeling test to get CDMs
        forward_test = ForwardTestCase()
        forward_test.test_forward_and_plot_cdms()
        # You may need to refactor test_model_forward to return cdms or save them to a file for easier access

        # For this example, let's assume you can access the CDMs as a list of dicts
        # (If not, you may need to refactor test_model_forward to expose them)
        # Here, we simulate loading them:
        import pandas as pd
    
        # For demonstration, let's load from a CSV if you saved it in test_model_forward
        if os.path.exists("synthetic_cdms.csv"):
            df = pd.read_csv("synthetic_cdms.csv")
            cdms = df.to_dict(orient="records")
            cdms = cdms[:100]  # Replace with actual list of CDMs from forward_test
        else:
            self.skipTest("No synthetic CDMs available for training.")

        # Wrap as a single Event and EventDataset
        event = Event([cdm for cdm in cdms])
        event_set = EventDataset(events=[event])

        # Initialize the LSTM model
        model = LSTMPredictor()

        # Train the model
        model.learn(event_set, epochs=2, lr=1e-3, batch_size=2, device='cpu', valid_proportion=0.2)

        # Save the trained model
        model.save("lstm_model_trained.pt")

        # Predict future CDMs using the trained model
        predicted_event = model.predict_event(event, num_samples=1, max_length=22)
        df_pred = predicted_event.to_dataframe()
        print(df_pred)

        # Optionally, plot a feature
        predicted_event.plot_feature('OBJECT1_CT_T', file_name='predicted_OBJECT1_CT_T_vs_TCA.pdf')
        print("Plot saved as predicted_OBJECT1_CT_T_vs_TCA.pdf")

        self.assertTrue(len(df_pred) > 0)

if __name__ == "__main__":
    unittest.main()
    