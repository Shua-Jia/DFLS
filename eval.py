import argparse
import os
import torch
from Data  import data_preprocess
from Network.FVSRs import FVSRs_en,FVSRs_de
from tqdm import tqdm
import time
import torchvision.utils as vutils

def run_inference(model_ckpt, output_dir,dataloder):
    model_en = FVSRs_en().cuda()
    model_de = FVSRs_de().cuda()
    if torch.cuda.is_available():
        device = f"cuda:0"
    else:
        device = torch.device("cpu")
    ckpt = torch.load(model_ckpt,map_location=device)
    model_en.load_state_dict({k.replace('module.', ''): v for k, v in ckpt['enc'].items()})
    model_de.load_state_dict({k.replace('module.', ''): v for k, v in ckpt['dec'].items()})
    del ckpt
    model_en.eval()
    model_de.eval()

    for index, batch_data in enumerate(tqdm(dataloder,desc="Evaluating")):
        input_fr = batch_data["input_frames"].cuda()
        with torch.no_grad():
            en_output = model_en(input_fr)
            de_output ,diff = model_de(en_output, input_fr)
        save_path_base = os.path.abspath(os.path.join(output_dir))
        os.makedirs(save_path_base, exist_ok=True)
        save_path = os.path.join(save_path_base, batch_data["save_name"][0])
        vutils.save_image(de_output, save_path, normalize=True)

def video_to_frame():
    pass

def infer():
    parser = argparse.ArgumentParser(description="Run inference on FVSR model")
    parser.add_argument("--data_path", type=str, default=r'input_frame', help="Path to the decoder model file")
    parser.add_argument("--model_ckpt", type=str,default=r'ckpts\FVSR.pth')
    parser.add_argument("--output_dir", type=str, default=r'Results')
    parser.add_argument('--latent', type=str, default='input', help='the input folder')
    parser.add_argument("--is_train", default=False)
    parser.add_argument("--nodes", type=int, default=1, help="number of total node")
    parser.add_argument("--batch_size", type=int, default=1, help="number of batch size")
    parser.add_argument("--works", type=int, default=0, help="number of works")
    parser.add_argument("--num_frames", type=int, default=3)
    parser.add_argument("--interval", type=int, default=1)
    parser.add_argument('--label_smoothing', type=float, default=0.001, help='Label smoothing factor')
    parser.add_argument("--fusion_in_channel", type=int, default=48)
    parser.add_argument("--patch", type=int, default=4)
    parser.add_argument("--num_blocks", type=int, default=12)

    parser.add_argument("--rate", type=int, default=4)


    args = parser.parse_args()
    current_path = os.path.abspath(__file__)
    directory = os.path.dirname(current_path)
    args.data_path = os.path.join(directory, args.data_path)
    args.ckpt = os.path.join(directory, args.model_ckpt)
    args.output_dir = os.path.join(directory, args.output_dir)
    return args


    args = infer()
    dataset = data_preprocess.data(args)
    dataloader = torch.utils.data.DataLoader(dataset)
    run_inference(args.model_ckpt, args.output_dir, dataloader)

