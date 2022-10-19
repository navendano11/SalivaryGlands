import pytorch_lightning as pl
import torch
from torch.utils.tensorboard import SummaryWriter
from metrics_evaluation import *
import numpy as np
import matplotlib.pyplot as plt

class Segmentation(pl.LightningModule):
    def __init__(self,model, lr=1e-4,loss=dice_loss,  metric = dice_score,  side = 'right', net2 = False):
        super().__init__()
        
        assert side in ['right', 'left'] # Verify that the side to segment value is correct

        self.backbone = model 
        self.loss = dice_loss
        self.lr = lr
        self.side = side
        self.net2 = net2
        
        
        if self.side == 'right':
            self.seg_side = 'mask_rsg'
        else:
            self.seg_side = 'mask_lsg'                
        
        self.metric = dice_score
        self.writer = SummaryWriter()
        
        self.train_preds = []
        self.train_gts = []
        self.val_preds = []
        self.val_gts = []
        self.test_preds = []
        self.test_gts = []

        self.train_loss = []
        self.val_loss = []
        self.test_loss = []
        self.train_metric = []
        self.val_metric = []
        self.test_metric= []
        
        
        
    def forward(self, x):
        y = self.backbone(x)

        return y
            
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        
        x, y = batch['ct'], batch[self.seg_side]
        x = x.cuda(0)
        y_pred = self.backbone.forward(x)
                
        if not torch.is_tensor(y_pred):
            y_pred = y_pred[0]
        
        loss = self.loss(y_pred, y)
        metric = self.metric(y_pred, y)
        
        self.train_loss.append(loss.item())
        self.train_metric.append(metric.item())

        y_pred_plot = np.array(y_pred.cpu().detach(), dtype=float)
        y_plot = y.cpu().numpy()
        
        mask_y = np.where(y_plot==1,y_plot,np.nan)
        mask_y_pred = np.where(y_pred_plot>=0.5,1,np.nan)

        self.train_preds.append(y_pred.cpu().detach())  
        self.train_gts.append(mask_y_pred)
        
        fig, ax = plt.subplots(nrows=1, ncols=4,figsize=[20, 7])
        ax[0].imshow(x[0, 0, :, :].cpu(),cmap="gray")
        ax[0].set_title('Image')
        ax[1].imshow(y_plot[0, 0, :, :], cmap="gray")
        ax[1].set_title('GT Segm')
        ax[2].imshow(y_pred_plot[0, 0, :, :], cmap="gray")
        ax[2].set_title('Pred Segm')
        ax[3].imshow(x[0, 0, :, :].cpu(),cmap='gray')
        ax[3].imshow(mask_y[0, 0, :, :],cmap='jet', alpha=1)
        ax[3].imshow(mask_y_pred[0, 0, :, :], alpha=0.5, cmap='autumn')
        ax[3].set_title('Overlay')
        self.writer.add_figure("Train/"+str(batch_idx), fig, self.current_epoch)
        plt.close()
        
        return {"loss": loss, "metric": metric, "label": y, "pred": y_pred}
      
    def training_epoch_end(self,output):

        
        all_preds, all_labels = [], []
        
        loss = 0
        metric = 0
        for o in output:
            loss = loss + o['loss']
            metric = metric + o['metric']
            probs = list(o['pred'].cpu().detach().numpy()) # predicted values
            labels = list(o['label'].cpu().numpy())
            all_preds.extend(probs)
            all_labels.extend(labels)
           
        
        loss = loss / len(output)
        metric = metric / len(output)
        self.writer.add_scalar('Epoch_loss/training', loss, self.current_epoch)
        self.writer.add_scalar('Epoch_metric/training', metric, self.current_epoch)
    
    def validation_step(self, batch, batch_idx):
        # validation_step defines the validation loop.
        x, y = batch['ct'], batch[self.seg_side]
        y_pred = self.backbone.forward(x)
       
        x = x.cuda(0)
        if not torch.is_tensor(y_pred):
            y_pred = y_pred[0]
        
        loss = self.loss(y_pred, y)
        metric = self.metric(y_pred, y)
        
                
        
        self.val_loss.append(loss.item())
        self.val_metric.append(metric.item())
        
        y_pred_plot = np.array(y_pred.cpu().detach(), dtype=float)
        y_plot = y.cpu().numpy()
        
       
        mask_y = np.where(y_plot==1,y_plot,np.nan)
        mask_y_pred = np.where(y_pred_plot>=0.5,1,np.nan)
        
        fig, ax = plt.subplots(nrows=1, ncols=4,figsize=[20, 7])
        ax[0].imshow(x[0, 0, :, :].cpu(),cmap="gray")
        ax[0].set_title('Image')
        ax[1].imshow(y_plot[0, 0, :, :], cmap="gray")
        ax[1].set_title('GT Segm')
        ax[2].imshow(y_pred_plot[0, 0, :, :], cmap="gray")
        ax[2].set_title('Pred Segm')
        ax[3].imshow(x[0, 0, :, :].cpu(),cmap='gray')
        ax[3].imshow(mask_y[0, 0, :, :],cmap='jet', alpha=1)
        ax[3].imshow(mask_y_pred[0, 0, :, :], alpha=0.5, cmap='autumn')
        ax[3].set_title('Overlay')
        self.writer.add_figure("Validation/"+str(batch_idx), fig, self.current_epoch)
        plt.close()
        
        
        
        return {"loss": loss, "metric": metric, "label": y, "pred": y_pred}
    
    def validation_epoch_end(self,output):
        
        all_preds, all_labels = [], []
        
        loss = 0
        metric = 0
        for o in output:
            loss = loss + o['loss']
            metric = metric + o['metric']
            probs = list(o['pred'].cpu().detach().numpy()) # predicted values
            labels = list(o['label'].cpu().numpy())
            all_preds.extend(probs)
            all_labels.extend(labels)
        loss = loss / len(output)
        metric = metric / len(output)
        self.log('val_dice', metric)
        self.log("val_loss", loss)
        self.writer.add_scalar('Epoch_loss/validation', loss, self.current_epoch)
        self.writer.add_scalar('Epoch_metric/validation', metric, self.current_epoch)
        
        # print('Epoch_loss/validation', loss, self.current_epoch)
        print('Epoch_metric/validation', metric, self.current_epoch)

    def test_step(self, batch, batch_idx):
        # test_step defines the test loop.
        x, y = batch['ct'], batch[self.seg_side]
        y_pred = self.backbone.forward(x)
        
        if not torch.is_tensor(y_pred):
            y_pred = y_pred[0]
        
        loss = self.loss(y_pred, y)
        metric = self.metric(y_pred, y)
        
        y_pred_plot = np.array(y_pred.cpu().detach(), dtype=float)
        y_plot = y.cpu().numpy()
        
        mask_y = np.where(y_plot==1,y_plot,np.nan)
        mask_y_pred = np.where(y_pred_plot>=0.5,1,np.nan)
        
        fig, ax = plt.subplots(nrows=1, ncols=4,figsize=[20, 7])
        ax[0].imshow(x[0, 0, :, :].cpu(),cmap="gray")
        ax[0].set_title('Image')
        ax[1].imshow(y_plot[0, 0, :, :], cmap="gray")
        ax[1].set_title('GT Segm')
        ax[2].imshow(y_pred_plot[0, 0, :, :], cmap="gray")
        ax[2].set_title('Pred Segm')
        ax[3].imshow(x[0, 0, :, :].cpu(),cmap='gray')
        ax[3].imshow(mask_y[0, 0, :, :],cmap='jet', alpha=1)
        ax[3].imshow(mask_y_pred[0, 0, :, :], alpha=0.5, cmap='autumn')
        ax[3].set_title('Overlay')
        self.writer.add_figure("Test/"+str(batch_idx), fig, self.current_epoch)
        plt.close()
        
        self.test_loss.append(loss.item())
        
        return {"loss": loss, "metric": metric,"label": y, "pred": y_pred}       
    
    def test_epoch_end(self, output):

        all_preds, all_labels = [], []
        
        loss = 0
        metric = 0
        for o in output:
            loss = loss + o['loss']
            metric = metric + o['metric']
            probs = list(o['pred'].cpu().detach().numpy()) # predicted values
            labels = list(o['label'].cpu().numpy())
            all_preds.extend(probs)
            all_labels.extend(labels)
            
        loss = loss / len(output)
        metric = metric / len(output)
        self.log('Test dice', metric)
        self.log('Loss', loss)
        self.writer.add_scalar('Epoch_loss/test', loss, self.current_epoch)
        self.writer.add_scalar('Epoch_metric/test', metric, self.current_epoch)