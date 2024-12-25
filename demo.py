import gradio as gr
import torchvision.transforms as transforms
from PIL import Image
from models.modeling import VisionTransformer, CONFIGS
import logging
import argparse
import os
import random
import numpy as np
import jittor as jt
from jittor import transform


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

    model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes,                                                   smoothing_value=args.smoothing_value)

    model.load_from(np.load(args.pretrained_dir))
    if args.pretrained_model is not None:
        pretrained_model = jt.load(args.pretrained_model)['model']
        model.load_state_dict(pretrained_model)
    return args, model

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    jt.set_global_seed(args.seed)

def prepare(label):
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
    # parser.add_argument('--fp16', action='store_true',
    #                     help="Whether to use 16-bit float precision instead of 32-bit")
    # parser.add_argument('--fp16_opt_level', type=str, default='O2',
    #                     help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
    #                          "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")

    parser.add_argument('--smoothing_value', type=float, default=0.0,
                        help="Label smoothing value\n")

    parser.add_argument('--split', type=str, default='non-overlap',
                        help="Split method")
    parser.add_argument('--slide_step', type=int, default=12,
                        help="Slide step for overlap split")
    
    parser.add_argument('--max_norm', type=float, default=1.0,
                        help="TODO")

    args = parser.parse_args()
    args.dataset = label
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename=args.name+'.log',  # 指定日志文件名
        filemode='w'  # 写入模式，'w' 表示覆盖写入，'a' 表示追加写入
    )

    # if args.fp16 and args.smoothing_value != 0:
    #     raise NotImplementedError("label smoothing not supported for fp16 training now")
    args.data_root = '{}/{}'.format(args.data_root, args.dataset)
    # Setup CUDA, GPU
    args.n_gpu = jt.get_device_count()

    # Set seed
    set_seed(args)

    # Model & Tokenizer Setup
    args, model = setup(args)

    return model



# 定义图像预处理函数
def preprocess_image(image, label):
    if label == "CUB_200_2011":
        test_transform=transform.Compose([transform.Resize((600, 600), Image.BILINEAR),
                                        transform.CenterCrop((448, 448)),
                                        transform.ToTensor(),
                                        transform.ImageNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return test_transform(image)


def read_species_file(file_path):
    species_dict = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                index = int(parts[0])
                species_name = parts[1]
                species_dict[index] = species_name
    return species_dict

# 示例使用
file_path = '/home/aiuser/workspace/2024ANN/TransFG_jittor/demo/cub.txt'
cub_species_dict = read_species_file(file_path)

# # 打印字典
# for index, species_name in species_dict.items():
#     print(f"{index}: {species_name}")

# 定义预测函数
def predict(image, label):
    model = prepare(label)
    pretrained_model = jt.load("/home/aiuser/workspace/2024ANN/TransFG_jittor/output/test_checkpoint.bin")['model']
    model.load_state_dict(pretrained_model)
    model.eval()

    image = preprocess_image(image, label)
    image = np.expand_dims(image, axis=0)

    image = jt.array(image)

    with jt.no_grad():
        output = model(image)

    print(output)
    preds, _ = jt.argmax(output, 1)
    print(preds)
    return cub_species_dict[int(preds) + 1]

# 创建 Gradio 接口
iface = gr.Interface(
    fn=predict,
    inputs=[gr.Image(type="pil"), gr.Textbox(label="Label")],  # 添加两个输入组件
    outputs="label",
    live=True
)

# 启动 Gradio 接口
iface.launch(share=True)