# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import random
import numpy as np
import time

from datetime import timedelta
from tqdm import tqdm

import jittor as jt
jt.flags.use_cuda = jt.has_cuda
from jittor.nn import CrossEntropyLoss, FocalLoss

from models.modeling import VisionTransformer, CONFIGS, LabelSmoothing, con_loss_mix
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.vis_utils import *
from utils.data_utils import get_loader

logger = logging.getLogger(__name__)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def reduce_mean(tensor):
    return tensor.mean()

def save_model(args, model):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, "%s_checkpoint.bin" % args.name)
    
    checkpoint = {
        'model': model_to_save.state_dict(),
    }
    jt.save(checkpoint, model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)

def setup(args):
    # Prepare model
    config = CONFIGS[args.model_type]
    config.split = args.split
    config.slide_step = args.slide_step

    if args.dataset == "CUB_200_2011":
        num_classes = 200
    elif args.dataset == "car":
        num_classes = 196
    elif args.dataset == "nabirds":
        num_classes = 555
    elif args.dataset == "dog":
        num_classes = 120
    elif args.dataset == "INat2017":
        num_classes = 5089

    model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes, smoothing_value=args.smoothing_value, loss_function=args.loss_function, mix=args.mix, clip_alpha=args.clip_alpha, psm=args.psm, con_loss=args.con_loss)

    model.load_from(np.load(args.pretrained_dir))
    if args.pretrained_model is not None:
        pretrained_model = jt.load(args.pretrained_model)['model']
        model.load_state_dict(pretrained_model)

    num_params = count_parameters(model)

    logger.info("{}".format(config))
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    return args, model

def count_parameters(model):
    for p in model.parameters():
        p.requires_grad = True
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    jt.set_global_seed(args.seed)
    jt.seed(args.seed)


def valid(args, model, test_loader, global_step):
    # Validation!
    eval_losses = AverageMeter()

    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])
    loss_fct = jt.nn.CrossEntropyLoss()
    valid_losses = []
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t for t in batch)
        
        x, y = batch
        with jt.no_grad():
            logits = model(x)
            eval_loss = loss_fct(logits, y)
            eval_loss = eval_loss.mean()
            eval_losses.update(eval_loss.item())

            preds, _ = jt.argmax(logits, dim=-1)
        if len(all_preds) == 0:
            all_preds.append(preds.numpy())
            all_label.append(y.numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.numpy(), axis=0
            )
        
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_label = all_preds[0], all_label[0]
    accuracy = simple_accuracy(all_preds, all_label)

    val_accuracy = reduce_mean(accuracy)
    val_accuracy = val_accuracy

    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Global Steps: %d" % global_step)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid Accuracy: %2.5f" % val_accuracy)

    return eval_losses.avg, val_accuracy

def fmix_data(x, y, alpha=1.0, decay_power=3, max_soft=0.0, reformulate=False):
    lam, mask = sample_mask(alpha, x.size(), decay_power, max_soft, reformulate)
    index = jt.randperm(x.size(0))

    mixed_x = mask * x + (1 - mask) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def sample_mask(alpha, size, decay_power, max_soft, reformulate):
    lam = np.random.beta(alpha, alpha)
    mask = make_low_freq_image(size, decay_power)
    mask = binarise_mask(mask, lam, size, max_soft, reformulate)
    return lam, mask

def binarise_mask(mask, lam, size, max_soft, reformulate):
    mask = mask - mask.min()
    mask = mask / mask.max()
    mask = (mask > np.percentile(mask, (1 - lam) * 100)).float()
    if reformulate:
        mask = mask * lam
    return mask

def make_low_freq_image(size, decay_power):
    freqs = np.fft.fftfreq(size[2], d=1.0)
    freqs = np.sqrt(freqs[:, None] ** 2 + freqs[None, :] ** 2)
    spectrum = np.random.normal(size=(size[0], size[1], size[2], size[3]))
    spectrum = spectrum / (freqs ** decay_power + 1e-8)
    spectrum = np.fft.ifft2(spectrum).real
    return spectrum

def mixup_data(x, y, args, alpha=1.0):
    if args.mix == "cutmix":
        
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1.0

        batch_size = x.size(0)
        index = jt.randperm(batch_size)

        H, W = x.size(2), x.size(3)
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int32(W * cut_rat)
        cut_h = np.int32(H * cut_rat)

        cut_w = max(1, cut_w)
        cut_h = max(1, cut_h)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bx1 = np.clip(cx - cut_w // 2, 0, W)
        bx2 = np.clip(cx + cut_w // 2, 0, W)
        by1 = np.clip(cy - cut_h // 2, 0, H)
        by2 = np.clip(cy + cut_h // 2, 0, H)

        # 混合输入数据x
        mixed_x = x.clone()
        print(bx1, by1, bx2, by2, W, H)
        mixed_x[:, :, by1:by2, bx1:bx2] = x[index, :, by1:by2, bx1:bx2]

        y_a, y_b = y, y[index]

        for i in range(mixed_x.shape[0]):
            image = mixed_x[i]
            image = image.numpy()
            
            # Denormalize if necessary (e.g., if data is in range [-1, 1])
            # image = (image + 1) / 2  # Uncomment if needed
            from PIL import Image
            # Scale to 0-255 and convert to uint8
            image = (image * 255).astype(np.uint8)
            
            # Check if channels are in the first dimension and transpose if necessary
            if image.shape[0] == 3:
                image = np.transpose(image, (1, 2, 0))
            
            # Ensure the image has the correct number of channels
            if image.shape[-1] not in [1, 3]:
                raise ValueError("Image has an unexpected number of channels.")
            
            # Create an Image object and save it
            img = Image.fromarray(image)
            img.save(os.path.join(f'image/image_{i}.jpg'))

        return mixed_x, y_a, y_b, lam
    else:
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
            if lam < 0.5:
                lam = 1 - lam
        else:
            lam = 1
        batch_size = x.size()[0]
        index = jt.randperm(batch_size)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]

        for i in range(mixed_x.shape[0]):
            image = mixed_x[i]
            image = image.numpy()
            
            # Denormalize if necessary (e.g., if data is in range [-1, 1])
            # image = (image + 1) / 2  # Uncomment if needed
            from PIL import Image
            # Scale to 0-255 and convert to uint8
            image = (image * 255).astype(np.uint8)
            
            # Check if channels are in the first dimension and transpose if necessary
            if image.shape[0] == 3:
                image = np.transpose(image, (1, 2, 0))
            
            # Ensure the image has the correct number of channels
            if image.shape[-1] not in [1, 3]:
                raise ValueError("Image has an unexpected number of channels.")
            
            # Create an Image object and save it
            img = Image.fromarray(image)
            img.save(os.path.join(f'image/image_{i}.jpg'))
        return mixed_x, y_a, y_b, lam

def train(args, model):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # Prepare dataset
    train_loader, test_loader = get_loader(args)

    # Prepare optimizer and scheduler
    optimizer = jt.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=0.9,
                                weight_decay=args.weight_decay)
    # optimizer.zero_grad()
    t_total = args.num_steps
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", args.num_steps)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    losses = AverageMeter()
    global_step, best_acc = 0, 0
    start_time = time.time()

    # 初始化数据收集器
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    tmp_losses = []

    while True:
        model.train()
        epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True,
                              disable=args.local_rank not in [-1, 0])
        all_preds, all_label = [], []
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t for t in batch)
            x, y = batch
            if args.mix != "none":
                # 应用 Mixup
                inputs, targets_a, targets_b, lam = mixup_data(x, y, args, alpha=args.mix_alpha)

                loss, logits = model(inputs, targets_a, targets_b, lam)

                loss = loss.mean()

            else:
                loss, logits = model(x, y)
                loss = loss.mean()


            tmp_losses.append(loss)

            preds, _ = jt.argmax(logits, dim=-1)

            if len(all_preds) == 0:
                all_preds.append(preds.numpy())
                all_label.append(y.numpy())
            else:
                all_preds[0] = np.append(
                    all_preds[0], preds.numpy(), axis=0
                )
                all_label[0] = np.append(
                    all_label[0], y.numpy(), axis=0
                )

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
                optimizer.backward(loss)
            else:
                optimizer.backward(loss)

            if (step + 1) % args.gradient_accumulation_steps == 0:
                losses.update(loss.item()*args.gradient_accumulation_steps)
                optimizer.clip_grad_norm(args.max_norm, args.max_grad_norm_type)
                scheduler.step()
                optimizer.step()
                global_step += 1

                epoch_iterator.set_description(
                    "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, losses.val)
                )

                if global_step % args.eval_every == 0:
                    with jt.no_grad():
                        val_loss , accuracy = valid(args, model, test_loader, global_step)
                        val_losses.append(val_loss) 
                        # val_losses.append(val_loss)
                        val_accuracies.append(accuracy)

                    logger.info("validation accuracy :%f", accuracy)
                    if args.local_rank in [-1, 0]:
                        if best_acc < accuracy:
                            save_model(args, model)
                            best_acc = accuracy
                        logger.info("best accuracy so far: %f" % best_acc)
                    
                    pred, label = all_preds[0], all_label[0]
                    accuracy = simple_accuracy(pred, label)
                    
                    train_accuracy = reduce_mean(accuracy)
                    train_accuracy = train_accuracy
                    train_accuracies.append(train_accuracy)
                    train_losses.append(reduce_mean(jt.array(tmp_losses)))
                    tmp_losses = []

                    logger.info("train accuracy so far: %f" % train_accuracy)
                    all_preds, all_label = [], []

                    model.train()

                if global_step % t_total == 0:
                    break

                
        losses.reset()
        if global_step % t_total == 0:
            break

    logger.info("Best Accuracy: \t%f" % best_acc)
    logger.info("End Training!")
    end_time = time.time()
    logger.info("Total Training Time: \t%f" % ((end_time - start_time) / 3600))

    show(train_losses, val_losses, train_accuracies, val_accuracies, args)

def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", required=False,
                        help="Name of this run. Used for monitoring.", default="debug")
    parser.add_argument("--dataset", choices=["CUB_200_2011", "car", "dog", "nabirds", "INat2017"], default="CUB_200_2011",
                        help="Which dataset.")
    parser.add_argument('--data_root', type=str, default='/home/aiuser/workspace/2024ANN/dataset')
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument("--pretrained_dir", type=str, default="/home/aiuser/workspace/2024ANN/model/ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--pretrained_model", type=str, default=None,
                        help="load pretrained model")
    parser.add_argument("--output_dir", default="./output", type=str,
                        help="The output directory where checkpoints will be written.")
    parser.add_argument("--img_size", default=448, type=int,
                        help="Resolution size")
    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=500, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")

    parser.add_argument("--learning_rate", default=3e-2, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_steps", default=10000, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm_type", default=2.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")

    parser.add_argument('--smoothing_value', type=float, default=0.0,
                        help="Label smoothing value\n")
    
    parser.add_argument('--clip_alpha', type=float , default=0.4)
    parser.add_argument('--psm', type=str, default='use')
    parser.add_argument('--con_loss', type=str, default="use")
    parser.add_argument('--split', type=str, default='non-overlap',
                        help="Split method")
    parser.add_argument('--slide_step', type=int, default=12,
                        help="Slide step for overlap split")
    
    parser.add_argument('--max_norm', type=float, default=1.0,
                        help="TODO")

    parser.add_argument('--mix', type=str, default="none")
    
    parser.add_argument('--mix_alpha', type=float, default=1.0,
                        help="TODO")
    
    parser.add_argument('--loss_function', type=str, default="CrossEntropyLoss")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename=args.name+'.log',  # 指定日志文件名
        filemode='w'  # 写入模式，'w' 表示覆盖写入，'a' 表示追加写入
    )

    args.data_root = '{}/{}'.format(args.data_root, args.dataset)
    # Setup CUDA, GPU
    args.n_gpu = jt.get_device_count()

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s,  16-bits training: %s" %
                   (args.local_rank, jt.has_cuda, args.n_gpu, False))

    # Set seed
    set_seed(args)

    # Model & Tokenizer Setup
    args, model = setup(args)
    print(model)
    # Training
    train(args, model)

if __name__ == "__main__":
    main()
