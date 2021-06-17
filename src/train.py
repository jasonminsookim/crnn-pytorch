import os

import cv2
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn import CTCLoss

from dataset import extract_jpg_meta, WbsinImageDataset, wbsin_collate_fn
from model import CRNN
from evaluate import evaluate
from config import train_config as config
from torchvision import transforms
from pathlib import Path

torch.backends.cudnn.enabled = True


def train_batch(crnn, data, optimizer, criterion, device):
    crnn.train()
    images, targets, target_lengths = [d.to(device) for d in data]

    logits = crnn(images)
    log_probs = torch.nn.functional.log_softmax(logits, dim=2)

    batch_size = images.size(0)
    input_lengths = torch.LongTensor([logits.size(0)] * batch_size)
    target_lengths = torch.flatten(target_lengths)

    loss = criterion(log_probs, targets, input_lengths, target_lengths)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def main():
    epochs = config["epochs"]
    train_batch_size = config["train_batch_size"]
    eval_batch_size = config["eval_batch_size"]
    lr = config["lr"]
    show_interval = config["show_interval"]
    valid_interval = config["valid_interval"]
    save_interval = config["save_interval"]
    cpu_workers = config["cpu_workers"]
    reload_checkpoint = config["reload_checkpoint"]
    valid_max_iter = config["valid_max_iter"]

    img_width = config["img_width"]
    img_height = config["img_height"]
    data_dir = config["data_dir"]

    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    #     # Extracts metadata related to the file path, label, and image width + height.
    #    wbsin_dir = Path().cwd() / "data" / "processed" / "cropped_wbsin"
    #    wbsin_meta_df = extract_jpg_meta(img_dir=wbsin_dir, img_type="wbsin")
    #     # Saves the extracted metadata.
    # interim_path = Path.cwd() / "data" / "interim"
    #    interim_path.mkdir(parents=True, exist_ok=True)
    #    wbsin_meta_df.to_csv(interim_path / "wbsin_meta.csv", index=False)

    X_transforms = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((160, 1440)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    wbsin_dataset = WbsinImageDataset(
        meta_file=(Path.cwd() / "data" / "processed" / "processed_wbsin_meta.csv"),
        transform=X_transforms,
    )

    train_size = int(0.8 * len(wbsin_dataset))
    test_size = len(wbsin_dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(
        wbsin_dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    # Save the test_dataset

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=cpu_workers,
        collate_fn=wbsin_collate_fn,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=True,
        num_workers=cpu_workers,
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
    if reload_checkpoint:
        crnn.load_state_dict(torch.load(reload_checkpoint, map_location=device))
    crnn.to(device)

    optimizer = optim.RMSprop(crnn.parameters(), lr=lr)
    criterion = CTCLoss(reduction="sum")
    criterion.to(device)

    assert save_interval % valid_interval == 0
    i = 1
    for epoch in range(1, epochs + 1):
        print(f"epoch: {epoch}")
        tot_train_loss = 0.0
        tot_train_count = 0
        for train_data in train_dataloader:
            loss = train_batch(crnn, train_data, optimizer, criterion, device)
            train_size = train_data[0].size(0)

            tot_train_loss += loss
            tot_train_count += train_size
            if i % show_interval == 0:
                print("train_batch_loss[", i, "]: ", loss / train_size)

            if i % valid_interval == 0:
                evaluation = evaluate(
                    crnn,
                    test_dataloader,
                    criterion,
                    decode_method=config["decode_method"],
                    beam_size=config["beam_size"],
                )
                print(
                    "valid_evaluation: loss={loss}, acc={acc}, char_acc={char_acc}".format(
                        **evaluation
                    )
                )

                if i % save_interval == 0:
                    prefix = "crnn"
                    loss = evaluation["loss"]
                    save_model_path = os.path.join(
                        config["checkpoints_dir"], f"{prefix}_{i:06}_loss{loss}.pt"
                    )
                    torch.save(crnn.state_dict(), save_model_path)
                    print("save model at ", save_model_path)
            i += 1

        print("train_loss: ", tot_train_loss / tot_train_count)


if __name__ == "__main__":
    main()
