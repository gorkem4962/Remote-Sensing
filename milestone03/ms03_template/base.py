import lightning as L
import torch



from torchmetrics import MetricCollection, Accuracy, F1Score, Precision, Recall, AveragePrecision
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
import sys

class BaseModel(L.LightningModule):
    def __init__(self, args, datamodule, network):
        super().__init__()
        self.args = args

        self.model = network
        self.save_hyperparameters('args')
        
        self.datamodule = datamodule
        self.criterion = self.init_criterion()

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.metrics = self.init_metrics()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """Handle training step."""
        x, t = batch  # Unpack batch (x: inputs, t: targets)
        y = self.forward(x)  # Forward pass
        loss = self.criterion(y, t)  # Compute loss
        probabilities = torch.softmax(y, dim=1)  # Class probabilities

        # Create output dictionary
        output = {
            "labels": t,
            "probabilities": probabilities,
            "loss": loss
        }

        # Log loss for training
       
        # Append output to the list
        self.training_step_outputs.append(output)
        return output

            
           
        

    def validation_step(self, batch, batch_idx):
        """Handle training step."""
        x, t = batch  # Unpack batch (x: inputs, t: targets)
        y = self.forward(x)  # Forward pass
        loss = self.criterion(y, t)  # Compute loss
        probabilities = torch.softmax(y, dim=1)  # Class probabilities

        # Create output dictionary
        output = {
            "labels": t,
            "probabilities": probabilities,
            "loss": loss
        }

        # Log loss for training
       
        # Append output to the list
        self.validation_step_outputs.append(output)
        return output


    def test_step(self, batch, batch_idx):
        """Handle training step."""
        x, t = batch  # Unpack batch (x: inputs, t: targets)
        y = self.forward(x)  # Forward pass
        loss = self.criterion(y, t)  # Compute loss
        probabilities = torch.softmax(y, dim=1)  # Class probabilities

        # Create output dictionary
        output = {
            "labels": t,
            "probabilities": probabilities,
            "loss": loss
        }

        # Log loss for training
       

        # Append output to the list
        self.test_step_outputs.append(output)
        return output
    
    def aggregate_and_log(self, outputs, stage):
        # Aggregate losses and logits
        all_losses = [o["loss"].item() for o in outputs]
        all_logits = torch.cat([o["logits"] for o in outputs], dim=0)
        all_labels = torch.cat([o["labels"] for o in outputs], dim=0)

        # Compute metrics
        avg_loss = sum(all_losses) / len(all_losses)
        metrics = self.metrics(all_logits, all_labels)

        # Log aggregated metrics
        self.log(f"{stage}_loss", avg_loss, on_epoch=True)
        for name, value in metrics.items():
            self.log(f"{stage}_{name}", value, on_epoch=True)



    def on_train_epoch_end(self):
        self.aggregate_and_log(self.training_step_outputs, "train")
        self.training_step_outputs = []  # Clear outputs after aggregation

    def on_validation_epoch_end(self):
        self.aggregate_and_log(self.validation_step_outputs, "val")
        self.validation_step_outputs = []  # Clear outputs after aggregation

    def on_test_epoch_end(self):
        self.aggregate_and_log(self.test_step_outputs, "test")
        self.test_step_outputs = []  # Clear outputs after aggregation
    ########################
    # CRITERION & OPTIMIZER
    ########################

    def configure_optimizers(self):
    # Initialize AdamW optimizer with specified lr and weight_decay
        
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.args.learning_rate,         # Provided learning rate
            weight_decay=self.args.weight_decay # Provided weight decay
        )

        # Define the OneCycleLR scheduler
        steps_per_epoch = self.args.steps_per_epoch
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.args.max_lr,               # Maximum learning rate
            epochs=self.args.epochs,               # Total number of epochs
            steps_per_epoch=steps_per_epoch,       # Number of steps per epoch
            pct_start=self.args.pct_start          # Fraction of cycle for initial phase
        )

        # Return optimizer and lr_scheduler
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',  # Apply the scheduler at each training step
                'frequency': 1       # Frequency of calling the scheduler
            }
        }
    def init_criterion(self):
         return torch.nn.CrossEntropyLoss()
        

    #################
    # LOGGING MODULE
    #################
    def init_metrics(self):
        if self.args.task == "slc":
            return MetricCollection({
                "accuracy_micro": Accuracy(task='multiclass', num_classes=self.args.num_classes, average="micro"),
                "accuracy_macro": Accuracy(task='multiclass', num_classes=self.args.num_classes, average="macro"),
                "accuracy_per_class": Accuracy(task='multiclass', num_classes=self.args.num_classes, average="none"),
                "f1_micro": F1Score(task='multiclass', num_classes=self.args.num_classes, average='micro'),
                "f1_macro": F1Score(task='multiclass', num_classes=self.args.num_classes, average='macro'),
                "f1_per_class": F1Score(task='multiclass', num_classes=self.args.num_classes, average='none'),
                "precision_micro": Precision(task='multiclass', num_classes=self.args.num_classes, average="micro"),
                "precision_macro": Precision(task='multiclass', num_classes=self.args.num_classes, average="macro"),
                "precision_per_class": Precision(task='multiclass', num_classes=self.args.num_classes, average="none"),
                "recall_micro": Recall(task='multiclass', num_classes=self.args.num_classes, average="micro"),
                "recall_macro": Recall(task='multiclass', num_classes=self.args.num_classes, average="macro"),
                "recall_per_class": Recall(task='multiclass', num_classes=self.args.num_classes, average="none"),
            })
        elif self.args.task == "mlc":
            return MetricCollection({
                "average_precision_micro": AveragePrecision(task='multilabel', num_labels=self.args.num_classes, average="micro"),
                "average_precision_macro": AveragePrecision(task='multilabel', num_labels=self.args.num_classes, average="macro"),
                "average_precision_per_class": AveragePrecision(task='multilabel', num_labels=self.args.num_classes, average="none"),
                "f1_micro": F1Score(task='multilabel', num_labels=self.args.num_classes, average="micro"),
                "f1_macro": F1Score(task='multilabel', num_labels=self.args.num_classes, average="macro"),
                "f1_per_class": F1Score(task='multilabel', num_labels=self.args.num_classes, average="none"),
            })
    def update_metrics(self, predictions, targets, split="train"):
        """
        Update the metrics based on the current predictions and targets.

        Args:
            predictions: Model's outputs.
            targets: Ground truth labels.
            split: Dataset split, e.g., 'train', 'val', 'test'.
        """
        metrics = self.metrics(predictions, targets)
        for metric_name, value in metrics.items():
            self.log(f"{split}_{metric_name}", value)
    
    def log(self, name, value):
        """
        Logs the metric (you can adapt it based on your logging setup).
        Here, we just print for demonstration.
        """
        print(f"{name}: {value:.4f}")


    ####################
    # DATA RELATED HOOKS
    ####################

    def train_dataloader(self):
        return self.datamodule.train_dataloader()

    def val_dataloader(self):
        return self.datamodule.val_dataloader()

    def test_dataloader(self):
        return self.datamodule.test_dataloader()
