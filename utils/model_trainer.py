from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


class TrainingManager:
    
    def __init__(self, model: nn.Module,
                 trainloader,
                 testloader,
                 optimizer,
                 criterion,
                 epochs,
                 scheduler = None,
                 trainingsession_path: Path = None,
                 optimizer_params = dict(),
                 scheduler_params = dict(),
                 device = 'cuda'
                 ) -> None:
        
        self._model = model
        self._trainloader = trainloader
        self._testloader = testloader
        self._optimizer = optimizer(model.parameters(), **optimizer_params)
        self._criterion = criterion
        self._epochs = epochs
        self.device = device
        
        if scheduler:
            self._scheduler = scheduler(self._optimizer, **scheduler_params)
        else:
            self._scheduler = None
        
        
        if trainingsession_path:
            trainingsession_path = Path(trainingsession_path)
            self._logdir = trainingsession_path / 'tensorboard_logs'
            self._logdir.mkdir(exist_ok=True, parents=True)
            self._writer = SummaryWriter(log_dir=self._logdir)
            self._model_path = trainingsession_path / 'checkpoints'
            self._model_path.mkdir(exist_ok=True, parents=True)
        else:
            self._logdir = None
            self._model_path = None
    
    
    def _train_one_epoch(self, epoch):
        
        self._model.train()  
        running_loss = 0.0
        running_acc = 0.0
        for i, (inputs, labels) in tqdm(enumerate(self._trainloader), total=len(self._trainloader)):
            
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self._optimizer.zero_grad()
            outputs, regularization_losses = self._model(inputs, labels)
            loss = self._criterion(outputs, labels)
            
            if regularization_losses is not None:
                    loss += regularization_losses
                       
            loss.backward()
            self._optimizer.step()
            running_loss += loss.item()
            
            pred_labels = torch.argmax(outputs, dim=1).cpu().numpy()
            accuracy = accuracy_score(pred_labels, labels.cpu().numpy())
            running_acc += accuracy
            
            if self._logdir and (i%100 == 0):
                self._writer.add_scalar('Loss/train', loss.item(), epoch * len(self._trainloader) + i)
                self._writer.add_scalar('Accuracy/train', accuracy, epoch * len(self._trainloader) + i)
                
                
                    
                
            del inputs
            del outputs
            del labels
           
        if self._scheduler:
            self._scheduler.step()
            current_lr = self._optimizer.param_groups[0]['lr']
            self._writer.add_scalar('LR', current_lr, epoch)
        

                 
        epoch_loss, epoch_accuracy = running_loss / len(self._trainloader), running_acc / len(self._trainloader)
        return epoch_loss, epoch_accuracy
    
    def _evaluate(self):
        self._model.eval()
        
        val_predictions = []
        val_targets = []
        with torch.no_grad():
            for inputs, labels in self._testloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs, _ = self._model(inputs)
                val_predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
        val_accuracy = accuracy_score(val_targets, val_predictions)
        
        return val_accuracy
        
    
    def train_model(self):
        
        # Iterate over epochs
        
        best_accuracy = 0
        for epoch in range(self._epochs):
            
            train_loss, train_accuracy = self._train_one_epoch(epoch)            
            val_accuracy = self._evaluate()
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                torch.save(self._model.state_dict(), self._model_path / f"best.pth")
                
            if self._logdir:
                self._writer.add_scalar('Accuracy/val', val_accuracy, epoch)
                self._writer.add_scalar('Loss/epoch', train_loss, epoch)    
                self._writer.add_scalar('Accuracy/epoch', train_accuracy, epoch)
            
                # for param_group in self._optimizer.param_groups:
                #     param_group['weight_decay'] = param_group['weight_decay']*0.98
                
            
            print(f"Epoch {epoch+1}/{self._epochs}, Loss: {self._epochs}, Validation Accuracy: {val_accuracy}") #, Learning Rate: {self._scheduler.get_last_lr()[0]}")
            print(f"Epoch {epoch+1}/{self._epochs}, Loss: {self._epochs}, Training Accuracy: {train_accuracy}")#, Learning Rate: {self._scheduler.get_last_lr()[0]}")
                
        # Save the model
        if self._model_path:
            torch.save(self._model.state_dict(), self._model_path / f"final.pth")
            print(f"final Model saved checkpoint at {self._model_path}")
        
        print('Training complete')
        # Close TensorBoard writer
        if self._logdir:
            self._writer.close()
            
        return max(self._evaluate(), best_accuracy)
