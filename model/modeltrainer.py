import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import PatchTSMixerForPrediction, Trainer, TrainingArguments
import plotly.graph_objs as go
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, f1_score
from data.dataloader import FloodForecastingDataLoader
import logging
import click
from typing import Tuple, List

# Constants for configuration management
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 10
DEFAULT_LEARNING_RATE = 1e-4

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


class ROCPlotGenerator:
    @staticmethod
    def generate_roc_curve(y_true: torch.Tensor, y_score: torch.Tensor, output_path: str) -> None:
        """
        Generate and save the ROC curve.

        Args:
            y_true (torch.Tensor): Ground truth labels.
            y_score (torch.Tensor): Predicted scores.
            output_path (str): Path to save the ROC curve plot.
        """
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_trace = go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve')
        roc_layout = go.Layout(title='Receiver Operating Characteristic', xaxis=dict(title='False Positive Rate'),
                               yaxis=dict(title='True Positive Rate'))
        roc_fig = go.Figure(data=[roc_trace], layout=roc_layout)
        roc_fig.write_image(output_path)
        logger.info(f"ROC curve saved to {output_path}")

    @staticmethod
    def generate_confusion_matrix(y_true: torch.Tensor, y_pred: torch.Tensor, output_path: str) -> None:
        """
        Generate and save the confusion matrix plot.

        Args:
            y_true (torch.Tensor): Ground truth labels.
            y_pred (torch.Tensor): Predicted labels.
            output_path (str): Path to save the confusion matrix plot.
        """
        conf_matrix = confusion_matrix(y_true, y_pred)
        conf_trace = go.Heatmap(z=conf_matrix, x=['No Flood', 'Flood'], y=['No Flood', 'Flood'])
        conf_layout = go.Layout(title='Confusion Matrix')
        conf_fig = go.Figure(data=[conf_trace], layout=conf_layout)
        conf_fig.write_image(output_path)
        logger.info(f"Confusion matrix saved to {output_path}")


class ModelTrainer:
    def __init__(self, flood_data_loader: FloodForecastingDataLoader):
        self.data_loader = flood_data_loader
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    def prepare_data(self) -> Tuple[TensorDataset, TensorDataset]:
        """
        Prepare the training and validation datasets.

        Returns:
            Tuple[TensorDataset, TensorDataset]: Training and validation datasets.
        """
        data_matrix = self.data_loader.prepare_data_matrix()
        train_data = data_matrix[:-self.data_loader.forecast_horizon]
        val_data = data_matrix[-self.data_loader.forecast_horizon:]

        train_dataset = TensorDataset(torch.tensor(train_data, dtype=torch.float32))
        val_dataset = TensorDataset(torch.tensor(val_data, dtype=torch.float32))

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

    def validate_and_evaluate(self, val_dataset: TensorDataset, model: PatchTSMixerForPrediction) -> float:
        """
        Validate and evaluate the model.

        Args:
            val_dataset (TensorDataset): Validation dataset.
            model (PatchTSMixerForPrediction): Model to evaluate.

        Returns:
            float: F1-score of the model.
        """
        val_dataloader = DataLoader(val_dataset, batch_size=DEFAULT_BATCH_SIZE)
        val_predictions = Trainer.predict(val_dataset, metric_key_prefix="val")
        flood_predictions = postprocess_predictions(val_predictions.predictions)

        val_labels = val_dataset.tensors[0][:, -1].long()
        f1_binary_score = f1_score(val_labels.numpy(), flood_predictions.numpy())

        ROCPlotGenerator.generate_roc_curve(val_labels, flood_predictions, 'roc_curve.svg')
        ROCPlotGenerator.generate_confusion_matrix(val_labels, flood_predictions, 'confusion_matrix.svg')

        return f1_binary_score

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
        )

        trainer.train()

        # Evaluate the model
        binary_f1_score = self.validate_and_evaluate(val_dataset, model)
        logger.info(f"F1-score: {binary_f1_score}")

        trainer.save_model("./trained_model")


@click.command()
@click.option('--geojson_path', type=str, required=True, help='Path to the GeoJSON file.')
@click.option('--start_date', type=str, required=True, help='Start date for the data.')
@click.option('--end_date', type=str, required=True, help='End date for the data.')
@click.option('--forecast_horizon', type=int, default=48, help='Forecast horizon in hours.')
@click.option('--epochs', type=int, default=DEFAULT_EPOCHS, help='Number of training epochs.')
@click.option('--batch_size', type=int, default=DEFAULT_BATCH_SIZE, help='Batch size for training.')
@click.option('--learning_rate', type=float, default=DEFAULT_LEARNING_RATE, help='Learning rate for training.')
def main(geojson_path: str, start_date: str, end_date: str, forecast_horizon: int, epochs: int, batch_size: int, learning_rate: float) -> None:
    """
    Main function to run the model training.

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
    train_instance.train(epochs, batch_size, learning_rate)


if __name__ == "__main__":
    main()