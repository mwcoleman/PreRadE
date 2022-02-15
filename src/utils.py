import csv, base64, time
import torch, torchmetrics
import numpy as np
import pytorch_lightning as pl
import wandb

def load_tsv(fname, topk=None):
    """Load object features from tsv file.

    :param fname: The path to the tsv file.
    :param topk: Only load features for top K images (lines) in the tsv file.
        Will load all the features if topk is either -1 or None.
    :return: A dict of image object features where each feature is a dict.
    """
    import sys
    csv.field_size_limit(sys.maxsize)
    start_time = time.time()
    print(f"\nStarting to load pre-extracted Faster-RCNN detected objects from {fname}...")
    with open(fname, 'r') as f:
        reader = csv.DictReader(f, ["img_id", "img_h", "img_w", 
                        "num_boxes", "boxes", "features"], delimiter="\t")
        
        data = {}
        for _, item in enumerate(reader):
            new_item = {}
            num_boxes = int(item['num_boxes'])
            for key in ['img_h', 'img_w', 'num_boxes']:
                new_item[key] = int(item[key])
            # slice from 2: to remove b' (csv.writer wraps all vals in str())
            new_item['features'] = np.frombuffer(base64.b64decode(item['features'][2:]), dtype=np.float32).reshape(num_boxes,-1).copy()
            new_item['boxes'] = np.frombuffer(base64.b64decode(item['boxes'][2:]), dtype=np.float32).reshape(num_boxes,4).copy()
            
            data[item['img_id']] = new_item
            if topk is not None and len(data) == topk:
                break
    elapsed_time = time.time() - start_time
    print(f"Loaded {len(data)} image features from {fname} in {elapsed_time:.2f} seconds.\n\n")
    return data


class MetricsCallback(pl.Callback):
    """PL Callback to Log auroc & TP,FP,TN,FP stats 
       using accumulated predictions & labels

       (Will probably break on distributed GPU training..)
    """
    def __init__(self,n_classes=13):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.auroc = torchmetrics.AUROC(num_classes=n_classes,
                                        average=None)
        self.statscores = torchmetrics.StatScores(reduce='macro',
                                                  num_classes=n_classes,
                                                    )
        self.auroc.to(self.device)
        self.statscores.to(self.device)

        self.result_auc = torch.zeros((n_classes,), device=self.device)
        self.n_classes = n_classes
    
    def on_validation_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs, batch, batch_idx,_) -> None:
        
        # Accumulate preds/labels across batches
        if batch_idx==0:
            self.val_preds = outputs['preds']
            self.val_labels = batch['label']
        else:
            self.val_preds = torch.vstack((self.val_preds,outputs['preds']))
            self.val_labels = torch.vstack((self.val_labels, batch['label'])) 

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        # Skip labels that don't have both instances (0,1); no chance of all 1's
        mask = torch.sum(self.val_labels, dim=0) > 0
        self.auroc.num_classes = torch.sum(mask)

        # Compute & update AUC for the others; carry over old vals (0) 
        self.result_auc[mask] = self.auroc(self.val_preds[:,mask], self.val_labels[:,mask].type(torch.int)).to(self.device)

        # Compute stat scores (TP,...) over epoch
        # StatScores returns tensor of shape (num_classes, 5)
        # When macro is used. last dim is [TP,FP,TN,FN,TP+FN]
        statscores = self.statscores(self.val_preds, self.val_labels.type(torch.int)).type(torch.float)
        
        for name,score,stats in zip(pl_module.labelset, self.result_auc, torch.tensor_split(statscores,self.n_classes,dim=0)):
            stats = stats.squeeze(0)

            results = {'VAL_AUC_'+name:score,
                       'SS_'+name+'_TP':stats[0],
                       'SS_'+name+'_FP':stats[1],
                       'SS_'+name+'_TN':stats[2],
                       'SS_'+name+'_FN':stats[3],
                       'SS_'+name+'_SUP':stats[4]}
            self.log_dict(results)

    def on_test_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs, batch, batch_idx,_) -> None:
        
        # Accumulate preds/labels across batches
        if batch_idx==0:
            self.test_preds = outputs['preds']
            self.test_labels = batch['label']
        else:
            self.test_preds = torch.vstack((self.test_preds, outputs['preds']))
            self.test_labels = torch.vstack((self.test_labels, batch['label'])) 

    def on_test_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        # TODO: Check if unecessary now due larger & balanced dsets
        # Skip labels that don't have both instances (0,1); no chance of all 1's
        mask = torch.sum(self.test_labels, dim=0) > 0
        self.auroc.num_classes = torch.sum(mask)

        # Compute & update AUC for the others; carry over old vals (0) 
        # auroc returns (num_classes,)
        self.result_auc[mask] = self.auroc(self.test_preds[:,mask], self.test_labels[:,mask].type(torch.int)).to(self.device)

        # Compute stat scores (TP,...) over epoch
        # StatScores returns tensor of shape (num_classes, 5)
        # When macro is used. last dim is [TP,FP,TN,FN,TP+FN]
        statscores = self.statscores(self.test_preds, self.test_labels.type(torch.int)).type(torch.float)
        num_total= torch.sum(statscores,dim=0)[4]

        supports = statscores.cpu().numpy()[:,4]
        prevs = supports/num_total.cpu().numpy()

        AUC_avg = self.result_auc.mean()
        wAUC_avg = np.sum((prevs*self.result_auc.cpu().numpy()))

        # Create a wandb table
        columns=['Label','AUC','# Cases','% Prev', 'TP', 'TN', 'FP', 'FN']
        table_data = []
        for name,score,stats in zip(pl_module.labelset, self.result_auc, torch.tensor_split(statscores,self.n_classes,dim=0)):
            stats = stats.squeeze(0)
            s = {k:v for k,v in zip(['TP','FP','TN','FN','SUP'],stats)}
            prev = np.round((100*s['SUP']/num_total).cpu().numpy(),decimals=2)
            table_data.append([name,score,s['SUP'],prev ,s['TP'],s['FP'],s['TN'],s['FN']])

        self.log_dict({'Avg_AUC':AUC_avg, 'wAvg_AUC':wAUC_avg})
        pl_module.logger.log_table(key="tr_table", columns=columns, data=table_data)

        # Also create a table logging the support of Val dataset for ease
        val_statscores = self.statscores(self.val_preds, self.val_labels.type(torch.int)).type(torch.float)
        val_supports = val_statscores.cpu().numpy()[:,4]
        val_prevs = supports/num_total.cpu().numpy()

        pl_module.logger.log_table(key='val_support',
                                   columns=['# Cases','Prev'],
                                   data=[val_supports,val_prevs])

        # Log a precision-recall chart
        # wandb.log({"pr": wandb.plot.pr_curve(self.test_labels, self.test_preds)})
