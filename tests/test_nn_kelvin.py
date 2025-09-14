import unittest
import matplotlib.pyplot as plt

class TestKelvinNN(unittest.TestCase):
    def test_kelvin_lstm(self):
        from kessler import EventDataset
        from kessler.data import kelvins_to_event_dataset
        import pandas as pd

        file_name='docs/notebooks/train_data.csv'
        events = kelvins_to_event_dataset(file_name, drop_features=['c_rcs_estimate', 't_rcs_estimate'], num_events=1000)

        nn_features=events.common_features(only_numeric=True)
        print("Total events:", len(events))

        len_test_set=int(0.05*len(events))
        events_test=events[-len_test_set:]
        events_train_and_val=events[:-len_test_set]

        from kessler.nn import LSTMPredictor
        model = LSTMPredictor(
            lstm_size=256,
            lstm_depth=2,
            dropout=0.2,
            features=nn_features)

        model.learn(events_train_and_val,
            epochs=10,
            lr=1e-3,
            batch_size=16,
            device='cpu',
            valid_proportion=0.15,
            num_workers=4,
            event_samples_for_stats=1000)

        model.save(file_name='LSTM_model_trained_kelvin')
        model.plot_loss(file_name='plot_loss.pdf')

        event=events_test[3]
        event_len=len(event)
        event_beginning=event[0:event_len-1]
        event_evolution=model.predict_event(event_beginning, num_samples=100, max_length=14)
        axs=event_evolution.plot_features(['RELATIVE_SPEED', 'MISS_DISTANCE'], return_axs=True, linewidth=0.1, color='red', alpha=0.33, label='Prediction')
        event.plot_features(['RELATIVE_SPEED', 'MISS_DISTANCE'], axs=axs, label='Real', legend=True)

        plt.savefig('prediction_vs_real_features.pdf')
        plt.close()

        # Add a simple assertion so unittest recognizes this as a test
        self.assertTrue(len(events) > 0)

if __name__ == "__main__":
    unittest.main()