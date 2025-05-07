import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from sklearn.metrics import roc_curve, confusion_matrix
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import PatchTSMixerForPrediction, Trainer, TrainingArguments
from data.dataloader import FloodForecastingDataLoader
import logging
from typing import Tuple, List

# Constants for configuration management
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 10
DEFAULT_LEARNING_RATE = 1e-4

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a Dash app
app = dash.Dash(__name__)

# Define the layout of the dashboard
app.layout = html.Div([
    html.H1('Training Dashboard'),
    dcc.Graph(id='val-loss-graph'),
    dcc.Graph(id='roc-curve-graph'),
    dcc.Graph(id='confusion-matrix-graph'),
    dcc.Interval(id='interval-component', interval=5000, n_intervals=0)  # Update every 5 seconds
])

# Global variables to store the data
val_loss_data = []
roc_curve_data = []
confusion_matrix_data = []


def postprocess_predictions(predictions: torch.Tensor) -> torch.Tensor:
    """
    Post-process the model predictions to obtain flood predictions.

    Args:
        predictions (torch.Tensor): Raw model predictions.

    Returns:
        torch.Tensor: Processed flood predictions.
    """
    probabilities = F.softmax(predictions, dim=-1)
    flood_predictions = torch.argmax(probabilities, dim=-1)
    return flood_predictions


class ModelTrainer:
    def __init__(self, flood_data_loader: FloodForecastingDataLoader):
        self.data_loader = flood_data_loader
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    def prepare_data(self) -> Tuple[torch.utils.data.TensorDataset, torch.utils.data.TensorDataset]:
        """
        Prepare the training and validation datasets.

        Returns:
            Tuple[torch.utils.data.TensorDataset, torch.utils.data.TensorDataset]: Training and validation datasets.
        """
        data_matrix = self.data_loader.prepare_data_matrix()
        train_data = data_matrix[:-self.data_loader.forecast_horizon]
        val_data = data_matrix[-self.data_loader.forecast_horizon:]

        train_dataset = torch.utils.data.TensorDataset(torch.tensor(train_data, dtype=torch.float32))
        val_dataset = torch.utils.data.TensorDataset(torch.tensor(val_data, dtype=torch.float32))

        return train_dataset, val_dataset

    def load_pretrained_model(self) -> PatchTSMixerForPrediction:
        """
        Load the pre-trained model.

        Returns:
            PatchTSMixerForPrediction: Pre-trained model.
        """
        model = PatchTSMixerForPrediction.from_pretrained("ibm-granite/granite-timeseries-patchtsmixer")
        model.to(self.device)
        return model

    def validate_and_evaluate(self, val_dataset: torch.utils.data.TensorDataset, model: PatchTSMixerForPrediction) -> None:
        """
        Validate and evaluate the model.

        Args:
            val_dataset (torch.utils.data.TensorDataset): Validation dataset.
            model (PatchTSMixerForPrediction): Model to evaluate.
        """
        val_dataloader = DataLoader(val_dataset, batch_size=DEFAULT_BATCH_SIZE)
        val_predictions = Trainer.predict(val_dataset, metric_key_prefix="val")
        flood_predictions = postprocess_predictions(val_predictions.predictions)

        val_labels = val_dataset.tensors[0][:, -1].long()

        # Update ROC curve data
        fpr, tpr, _ = roc_curve(val_labels.numpy(), flood_predictions.numpy())
        roc_curve_data.append((fpr, tpr))

        # Update confusion matrix data
        conf_matrix = confusion_matrix(val_labels.numpy(), flood_predictions.numpy())
        confusion_matrix_data.append(conf_matrix)

    def train(self, n_epochs: int, n_batch_size: int, train_learning_rate: float) -> None:
        """
        Train the model.

        Args:
            n_epochs (int): Number of training epochs.
            n_batch_size (int): Batch size for training.
            train_learning_rate (float): Learning rate for training.
        """
        train_dataset, val_dataset = self.prepare_data()

        train_dataloader = DataLoader(train_dataset, batch_size=n_batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=n_batch_size)

        model = self.load_pretrained_model()

        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=n_epochs,
            per_device_train_batch_size=n_batch_size,
            per_device_eval_batch_size=n_batch_size,
            learning_rate=train_learning_rate,
            evaluation_strategy="epoch",
            logging_dir="./logs",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=None,
            compute_metrics=self.validate_and_evaluate,
        )

        trainer.train(resume_from_checkpoint=False)

        trainer.save_model("./trained_model")


# Callback to update the validation loss graph
@app.callback(Output('val-loss-graph', 'figure'),
              [Input('interval-component', 'n_intervals')])
def update_val_loss_graph(n):
    trace = go.Scatter(y=val_loss_data, mode='lines', name='Validation Loss')
    layout = go.Layout(title='Validation Loss', xaxis=dict(title='Epoch'), yaxis=dict(title='Loss'))
    return go.Figure(data=[trace], layout=layout)


# Callback to update the ROC curve graph
@app.callback(Output('roc-curve-graph', 'figure'),
              [Input('interval-component', 'n_intervals')])
def update_roc_curve_graph(n):
    if len(roc_curve_data) > 0:
        fpr, tpr = roc_curve_data[-1]
        trace = go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve')
        layout = go.Layout(title='Receiver Operating Characteristic', xaxis=dict(title='False Positive Rate'),
                           yaxis=dict(title='True Positive Rate'))
        return go.Figure(data=[trace], layout=layout)
    else:
        return go.Figure()


# Callback to update the confusion matrix graph
@app.callback(Output('confusion-matrix-graph', 'figure'),
              [Input('interval-component', 'n_intervals')])
def update_confusion_matrix_graph(n):
    if len(confusion_matrix_data) > 0:
        conf_matrix = confusion_matrix_data[-1]
        trace = go.Heatmap(z=conf_matrix, x=['No Flood', 'Flood'], y=['No Flood', 'Flood'],
                           colorscale=[[0, 'red'], [1, 'green']])
        layout = go.Layout(title='Confusion Matrix')
        return go.Figure(data=[trace], layout=layout)
    else:
        return go.Figure()


def run_dashboard(geojson_path: str, start_date: str, end_date: str, forecast_horizon: int,
                  epochs: int, batch_size: int, learning_rate: float) -> None:
    """
    Run the dashboard and start the model training.

    Args:
        geojson_path (str): Path to the GeoJSON file.
        start_date (str): Start date for the data.
        end_date (str): End date for the data.
        forecast_horizon (int): Forecast horizon in hours.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        learning_rate (float): Learning rate for training.
    """
    data_loader = FloodForecastingDataLoader(geojson_path, start_date, end_date, forecast_horizon)
    train_instance = ModelTrainer(data_loader)

    # Start the model training in a separate thread
    import threading
    train_thread = threading.Thread(target=train_instance.train, args=(epochs, batch_size, learning_rate))
    train_thread.start()

    # Run the Dash app
    app.run_server(debug=True)


if __name__ == "__main__":
    # Set the parameters for the dashboard and model training
    geojson_path = "path/to/geojson"
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    forecast_horizon = 48
    epochs = 10
    batch_size = 64
    learning_rate = 1e-4

    run_dashboard(geojson_path, start_date, end_date, forecast_horizon, epochs, batch_size, learning_rate)