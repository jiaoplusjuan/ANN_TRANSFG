import logging
from PIL import Image
import os

from jittor import transform
from jittor.dataset import DataLoader, RandomSampler, SequentialSampler

from .dataset import CUB, CarsDataset, dogs, NABirds,INat2017
from .autoaugment import AutoAugImageNetPolicy

logger = logging.getLogger(__name__)


def get_loader(args):
    # if args.local_rank not in [-1, 0]:
    #     torch.distributed.barrier()

    if args.dataset == 'CUB_200_2011':
        train_transform=transform.Compose([transform.Resize((600, 600), Image.BILINEAR),
                                    transform.RandomCrop((448, 448)),
                                    transform.RandomHorizontalFlip(),
                                    transform.ToTensor(),
                                    transform.ImageNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        test_transform=transform.Compose([transform.Resize((600, 600), Image.BILINEAR),
                                    transform.CenterCrop((448, 448)),
                                    transform.ToTensor(),
                                    transform.ImageNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        trainset = CUB(root=args.data_root, is_train=True, transform=train_transform)
        testset = CUB(root=args.data_root, is_train=False, transform = test_transform)
    elif args.dataset == 'car':
        trainset = CarsDataset(os.path.join(args.data_root,'devkit/cars_train_annos.mat'),
                            os.path.join(args.data_root,'cars_train'),
                            os.path.join(args.data_root,'devkit/cars_meta.mat'),
                            # cleaned=os.path.join(data_dir,'cleaned.dat'),
                            transform=transform.Compose([
                                    transform.Resize((600, 600), Image.BILINEAR),
                                    transform.RandomCrop((448, 448)),
                                    transform.RandomHorizontalFlip(),
                                    AutoAugImageNetPolicy(),
                                    transform.ToTensor(),
                                    transform.ImageNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                            )
        testset = CarsDataset(os.path.join(args.data_root,'cars_test_annos_withlabels.mat'),
                            os.path.join(args.data_root,'cars_test'),
                            os.path.join(args.data_root,'devkit/cars_meta.mat'),
                            # cleaned=os.path.join(data_dir,'cleaned_test.dat'),
                            transform=transform.Compose([
                                    transform.Resize((600, 600), Image.BILINEAR),
                                    transform.CenterCrop((448, 448)),
                                    transform.ToTensor(),
                                    transform.ImageNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                            )
    elif args.dataset == 'dog':
        train_transform=transform.Compose([transform.Resize((600, 600), Image.BILINEAR),
                                    transform.RandomCrop((448, 448)),
                                    transform.RandomHorizontalFlip(),
                                    transform.ToTensor(),
                                    transform.ImageNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        test_transform=transform.Compose([transform.Resize((600, 600), Image.BILINEAR),
                                    transform.CenterCrop((448, 448)),
                                    transform.ToTensor(),
                                    transform.ImageNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        trainset = dogs(root=args.data_root,
                                train=True,
                                cropped=False,
                                transform=train_transform,
                                download=False
                                )
        testset = dogs(root=args.data_root,
                                train=False,
                                cropped=False,
                                transform=test_transform,
                                download=False
                                )
    elif args.dataset == 'nabirds':
        train_transform=transform.Compose([transform.Resize((600, 600), Image.BILINEAR),
                                        transform.RandomCrop((448, 448)),
                                        transform.RandomHorizontalFlip(),
                                        transform.ToTensor(),
                                        transform.ImageNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        test_transform=transform.Compose([transform.Resize((600, 600), Image.BILINEAR),
                                        transform.CenterCrop((448, 448)),
                                        transform.ToTensor(),
                                        transform.ImageNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        trainset = NABirds(root=args.data_root, train=True, transform=train_transform)
        testset = NABirds(root=args.data_root, train=False, transform=test_transform)
    elif args.dataset == 'INat2017':
        train_transform=transform.Compose([transform.Resize((400, 400), Image.BILINEAR),
                                    transform.RandomCrop((304, 304)),
                                    transform.RandomHorizontalFlip(),
                                    AutoAugImageNetPolicy(),
                                    transform.ToTensor(),
                                    transform.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        test_transform=transform.Compose([transform.Resize((400, 400), Image.BILINEAR),
                                    transform.CenterCrop((304, 304)),
                                    transform.ToTensor(),
                                    transform.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        trainset = INat2017(args.data_root, 'train', train_transform)
        testset = INat2017(args.data_root, 'val', test_transform)

    # if args.local_rank == 0:
    #     torch.distributed.barrier()

    train_sampler = RandomSampler(trainset)
    test_sampler = SequentialSampler(testset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              num_workers=4,
                              drop_last=True
                              )
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=args.eval_batch_size,
                             num_workers=4
                                )if testset is not None else None

    return train_loader, test_loader
