import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataloader import custom_dataset

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
from utill import set_seed
from module import HAN
import wandb
from torcheval.metrics import BinaryConfusionMatrix


def ddp_setup(rank, world_size):
    """ 
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12347"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        dev_data: DataLoader, 
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler, 
        gpu_id: int,
        check_every: int,
        train_class_weight,
        dev_class_weight, 
        usewb: bool,
        save_name: str,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.dev_data = dev_data
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.check_every = check_every
        self.model = DDP(model, device_ids=[gpu_id])
        self.train_class_weight = train_class_weight.to(gpu_id)
        self.dev_class_weight = dev_class_weight.to(gpu_id)
        self.best_loss = 10000
        self.best_epoch = 0
        self.usewb = usewb
        self.save_name = save_name


    
    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.cross_entropy(output, targets, weight=self.train_class_weight, \
            reduction='mean')
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return loss

    def _run_epoch(self, epoch):
        self.model.train()
        b_sz = len(next(iter(self.train_data))[0])
        loss = []
        self.train_data.sampler.set_epoch(epoch)
        for source, targets in self.train_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            loss.append(self._run_batch(source, targets))
            
        loss = sum(loss) / len(loss)
        
        wandb.log({'train/loss': loss,
                   }, step=epoch) if self.usewb and self.gpu_id == 0 else None
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)} | loss: {loss}")

    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = f'{self.save_name}.pt'
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")



    def _eval_epoch(self, epoch):
        self.model.eval()
        mat = BinaryConfusionMatrix()
        with torch.no_grad():
            loss = []
            for source, targets in self.dev_data:
                source = source.to(self.gpu_id)
                targets = targets.to(self.gpu_id)
                output = self.model(source)
                pred = F.softmax(output, dim=1).argmax(dim=1)
                loss.append(F.cross_entropy(output, targets, weight=self.dev_class_weight, \
                    reduction='mean'))
                mat.update(pred, targets)
        
        loss = sum(loss) / len(loss)
        calc_metric(mat, loss, epoch, self.usewb)
            
        return loss

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.check_every == 0: 
                loss = self._eval_epoch(epoch)
                if loss < self.best_loss:
                    self.best_loss = loss 
                    self.best_epoch = epoch
                    self._save_checkpoint(epoch)
        wandb.log({
            'dev/best_loss': self.best_loss,
            'dev/best_epoch': self.best_epoch
            }) if self.usewb and self.gpu_id==0 else None


def load_train_objs(flags, l):

    model = HAN(flags)
    if flags.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=flags.learning_rate, \
            weight_decay = flags.weight_decay)
    elif flags.optimizer == 'momentumSGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=flags.learning_rate, \
            weight_decay = flags.weight_decay, momentum=0.9)
    elif flags.optimizer == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=flags.learning_rate, \
            weight_decay = flags.weight_decay)
        
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=flags.learning_rate, \
        epochs=flags.train_epochs, steps_per_epoch=l)
    return model, optimizer, scheduler

def calc_metric(mat, loss, epoch=0, usewb=False, mode='dev'):#positive = down, negative = up
    mat = mat.compute()
    tp, fn, fp, tn = mat[0, 0], mat[0, 1], mat[1, 0], mat[1, 1]
    acc = (tp + tn) / (tp + fn + fp + tn)
    rec = tp / (tp + fn)
    pre = tp / (tp + fp)
    f1 = 2 * tp / (2 * tp + fp + fn)
    mcc = (tp * tn - fp * fn) / torch.sqrt(((tp + fp) * (tp + fn) \
        * (tn + fp) * (tn + fn)))
    print(f'{mode}_loss: {loss}')
    print(f'(TP, FN, FP, TN): {tp}, {fn}, {fp}, {tn}')
    print(f'(acc, rec, pre, f1, mcc): {acc}, {rec}, {pre}, {f1}, {mcc}')
    wandb.log({f'{mode}/loss': loss, 
                f'{mode}/acc': acc,
                f'{mode}/rec': rec,
                f'{mode}/pre': pre,
                f'{mode}/f1': f1,
                f'{mode}/mcc': mcc, 
                }, step=epoch) if usewb else None
    
    

def test(flags, save_name):
    test_data = custom_dataset(flags.test_x_path, flags.test_y_path, flags.days, \
        flags.max_num_tweets_len, flags.max_num_tokens_len)
    test_loader = DataLoader(dataset=test_data, batch_size=flags.batch_size, 
                              num_workers=flags.num_workers, shuffle=False, \
                                  pin_memory=True, drop_last=True)
    
    model = HAN(flags).to(device=flags.device)
    model.load_state_dict(torch.load(f'{save_name}.pt'), strict=False)
    model.eval()
    mat = BinaryConfusionMatrix()
    with torch.no_grad():
        loss = []
        for source, targets in test_loader:
            source = source.to(flags.device)
            targets = targets.to(flags.device)
            output = model(source)
            pred = F.softmax(output, dim=1).argmax(dim=1)
            loss.append(F.cross_entropy(output, targets, weight=(test_data.class_weights).to(flags.device), \
                reduction='mean'))
            mat.update(pred, targets)
    
    loss = sum(loss)/len(loss)
    calc_metric(mat=mat, loss=loss, usewb=flags.usewb, mode='test')
    
    

def main(rank: int, world_size: int, flags):
    save_name = init_wandb(flags) if flags.usewb and rank==0 else 'tmp'
    set_seed(flags.seed)
    ddp_setup(rank, world_size)
    train_data = custom_dataset(flags.train_x_path, flags.train_y_path, flags.days, \
        flags.max_num_tweets_len, flags.max_num_tokens_len)
    dev_data = custom_dataset(flags.dev_x_path, flags.dev_y_path, flags.days, \
        flags.max_num_tweets_len, flags.max_num_tokens_len)
    train_loader = DataLoader(dataset=train_data, batch_size=flags.batch_size, 
                              num_workers=flags.num_workers, shuffle=False, \
                                  pin_memory=True, drop_last=True, \
                                      sampler=DistributedSampler(train_data))
    dev_loader = DataLoader(dataset=dev_data, batch_size=flags.batch_size, 
                              num_workers=flags.num_workers, shuffle=False, \
                                  pin_memory=True, drop_last=True)
     
    model, optimizer, scheduler = load_train_objs(flags, len(train_loader))
    trainer = Trainer(model, train_loader, dev_loader, optimizer, scheduler, rank, \
        flags.check_interval, train_data.class_weights, dev_data.class_weights, flags.usewb, save_name)
    trainer.train(flags.train_epochs)
    if flags.usewb and rank==0: test(flags, save_name) 
    destroy_process_group()

def init_wandb(flags):
    import wandb
    from wandb_setting import ini
    import random
    import string
    
    exp_name = ''.join(random.choices(
        string.ascii_uppercase + string.digits, k=8))
    ini()
    wandb.init(name=exp_name, config=flags.__dict__)
    wandb.save(f'{flags.save_dir}/*', flags.save_dir)
    return exp_name

if __name__ == "__main__":
    import config 
    flags = config.args
    
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, flags), nprocs=world_size)
    