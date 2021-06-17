"""Usage: predict.py [-m MODEL] [-s BS] [-d DECODE] [-b BEAM] [IMAGE ...]

-h, --help    show this
-m MODEL     model file [default: ./checkpoints/crnn_synth90k.pt]
-s BS       batch size [default: 256]
-d DECODE    decode method (greedy, beam_search or prefix_beam_search) [default: beam_search]
-b BEAM   beam size [default: 10]

"""
from docopt import docopt
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms

from config import common_config as config
from dataset import WbsinImageDataset, wbsin_collate_fn
from model import CRNN
from ctc_decoder import ctc_decode
from pathlib import Path

def predict(crnn, dataloader, label2char, decode_method, beam_size):
    crnn.eval()
    pbar = tqdm(total=len(dataloader), desc="Predict")

    all_preds = []
    with torch.no_grad():
        for images, targets, target_lengths in dataloader:
            device = "cuda" if next(crnn.parameters()).is_cuda else "cpu"

            images = images.to(device)
            logits = crnn(images)
            log_probs = torch.nn.functional.log_softmax(logits, dim=2)
            preds = ctc_decode(
                log_probs,
                method=decode_method,
                beam_size=beam_size,
                label2char=label2char,
            )
            all_preds += preds

            pbar.update(1)
        pbar.close()

    return all_preds


def show_result(paths, preds):
    print("\n===== result =====")
    for path, pred in zip(paths, preds):
        text = "".join(pred)
        print(f"{path} > {text}")


def main():
    arguments = docopt(__doc__)

    images = arguments["IMAGE"]
    images = [image for image in (Path.cwd() / images[0]).glob("*.jpg")]
    reload_checkpoint = arguments["-m"]
    batch_size = int(arguments["-s"])
    decode_method = arguments["-d"]
    beam_size = int(arguments["-b"])

    img_height = config["img_height"]
    img_width = config["img_width"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    X_transforms = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((160, 1440)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    wbsin_dataset = WbsinImageDataset(
        meta_file=(Path.cwd() / "data" / "interim" / "cropped_wbsin_meta.csv"), transform=X_transforms
    )

    train_size = int(0.8 * len(wbsin_dataset))
    test_size = len(wbsin_dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(
        wbsin_dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        collate_fn=wbsin_collate_fn,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        collate_fn=wbsin_collate_fn,
    )

    num_class = len(WbsinImageDataset.LABEL2CHAR) + 1
    crnn = CRNN(
        1,
        img_height,
        img_width,
        num_class,
        map_to_seq_hidden=config["map_to_seq_hidden"],
        rnn_hidden=config["rnn_hidden"],
        leaky_relu=config["leaky_relu"],
    )
    crnn.load_state_dict(torch.load(reload_checkpoint, map_location=device))
    crnn.to(device)

    preds = predict(
        crnn,
        train_dataloader,
        WbsinImageDataset.LABEL2CHAR,
        decode_method=decode_method,
        beam_size=beam_size,
    )

    show_result(images, preds)


if __name__ == "__main__":
    main()
