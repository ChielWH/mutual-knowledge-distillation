import torch
from trainer import Trainer
from config import get_config
from utils import prepare_dirs, save_config
from data_loader import get_test_loader, get_train_loader
from model_factories.resnet_factory import create_model

config, unparsed = get_config()

# alter config for specific test
torch.manual_seed(config.random_seed)
kwargs = {}
if not config.disable_cuda and torch.cuda.is_available():
    use_gpu = True
    torch.cuda.manual_seed_all(config.random_seed)
    kwargs = {'num_workers': config.num_workers,
              'pin_memory': config.pin_memory}
else:
    use_gpu = False

# teachers = [create_model('32') for _ in range(
#     len(config.model_names))]

teachers = []

test_data_loader = get_test_loader(data_dir=config.data_dir,
                                   batch_size=config.batch_size,
                                   cuda=use_gpu,
                                   teachers=teachers,
                                   model_num=len(config.model_names),
                                   **kwargs)

if config.is_train:
    train_data_loader = get_train_loader(data_dir=config.data_dir,
                                         batch_size=config.batch_size,
                                         random_seed=config.random_seed,
                                         shuffle=config.shuffle,
                                         model_num=len(config.model_names),
                                         teachers=teachers,
                                         cuda=use_gpu,
                                         **kwargs)

    data_loader = (train_data_loader, test_data_loader)
else:
    data_loader = test_data_loader


trainer = Trainer(config, data_loader)
trainer.train()
