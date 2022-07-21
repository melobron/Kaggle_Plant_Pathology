import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import random
from torch.utils.tensorboard import SummaryWriter
import json

from tqdm.notebook import tqdm
from dataset import Plant
from utils import *


class TrainPlant:
    def __init__(self, args):
        # Arguments
        self.args = args

        # Device
        self.gpu_num = args.gpu_num
        self.device = torch.device('cuda:{}'.format(self.gpu_num) if torch.cuda.is_available() else 'cpu')

        # Random Seeds
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

        # Models
        self.model = create_model(args, pretrained=True).to(self.device)

        # Training Parameters
        self.n_epochs = args.n_epochs
        self.batch_size = args.batch_size
        self.threshold = args.threshold

        # Loss
        self.criterion = torch.nn.MultiLabelSoftMarginLoss()

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)

        # Scheduler
        # self.scheduler =

        # Transform
        train_transform = A.Compose(get_transforms(args))
        val_transform = A.Compose([
            A.Resize(args.patch_size, args.patch_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

        # Dataset
        self.train_dataset = Plant(args=self.args, mode='train', transform=train_transform)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=True)

        self.valid_dataset = Plant(args=self.args, mode='valid', transform=val_transform)
        self.valid_dataloader = DataLoader(self.valid_dataset, batch_size=args.batch_size, shuffle=True)

        # Metric Monitors
        self.train_monitor = MetricMonitor()
        self.valid_monitor = MetricMonitor()

        # Directories
        self.exp_dir = make_exp_dir('./experiments/')['new_dir']
        self.exp_num = make_exp_dir('./experiments/')['new_dir_num']
        self.checkpoint_dir = os.path.join(self.exp_dir, 'checkpoints')
        self.result_path = os.path.join(self.exp_dir, 'results')

        # Tensorboard
        self.summary = SummaryWriter('runs/exp{}'.format(self.exp_num))

    def prepare(self):
        # Save Paths
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)

        # Save Argument file
        param_file = os.path.join(self.exp_dir, 'params.json')
        with open(param_file, mode='w') as f:
            json.dump(self.args.__dict__, f, indent=4)

    def train(self):
        print(self.device)

        self.prepare()

        # Losses
        loss_train, train_acc, train_f1 = 0, 0, 0
        loss_valid, valid_acc, valid_f1 = 0, 0, 0

        for epoch in range(1, self.n_epochs + 1):

            # Training
            self.model.train()

            train_stream = tqdm(self.train_dataloader)
            for batch, (img, target) in enumerate(train_stream, start=1):
                img, target = img.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()

                pred_prob = self.model(img)
                loss = self.criterion(pred_prob, target)

                loss.backward()
                self.optimizer.step()

                loss_train += loss
                acc, f1 = get_metrics(to_numpy(pred_prob), to_numpy(target), self.threshold)

                train_acc += acc
                train_f1 += f1

                print(
                    '[Epoch {}][{}/{}] | Training | loss:{:.3f} acc:{:.3f}, f1:{:.3f}'.format(
                        epoch, (batch + 1) * self.batch_size, len(self.train_dataset),
                        loss.item(), acc, f1
                    ))

            # Save Metrics
            self.train_monitor.update('loss', loss_train/self.batch_size)
            self.train_monitor.update('acc', train_acc/self.batch_size)
            self.train_monitor.update('f1', train_f1/self.batch_size)

            # Validation
            self.model.eval()

            valid_stream = tqdm(self.valid_dataloader)
            for batch, (img, target) in enumerate(valid_stream, start=1):
                img, target = img.to(self.device), target.to(self.device)

                with torch.no_grad():
                    pred_prob = self.model(img)
                    loss = self.criterion(pred_prob, target)

                loss_valid += loss
                acc, f1 = get_metrics(to_numpy(pred_prob), to_numpy(target), self.threshold)

                valid_acc += acc
                valid_f1 += f1

                print(
                    '[Epoch {}][{}/{}] | Validation | loss:{:.3f} acc:{:.3f}, f1:{:.3f}'.format(
                        epoch, (batch + 1) * self.batch_size, len(self.valid_dataset),
                        loss.item(), acc, f1
                    ))

            # Save Metrics
            self.valid_monitor.update('loss', loss_valid/self.batch_size)
            self.valid_monitor.update('acc', valid_acc/self.batch_size)
            self.valid_monitor.update('f1', valid_f1/self.batch_size)

            # Checkpoints
            if epoch % 25 == 0 or epoch == self.n_epochs:
                torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, 'model_{}epochs.pth'.format(epoch)))
