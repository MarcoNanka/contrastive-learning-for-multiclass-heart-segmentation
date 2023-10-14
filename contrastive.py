import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
import wandb
from model import Encoder
from data_loading import MMWHSContrastiveDataset
from config import parse_args
import os

os.environ['WANDB_CACHE_DIR'] = "$HOME/wandb_tmp"
os.environ['WANDB_CONFIG_DIR'] = "$HOME/wandb_tmp"
os.environ['WANDB_DIR'] = "$HOME/wandb_tmp"
os.environ['WANDB_TEMP'] = "$HOME/wandb_tmp"


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, x1, x2, labels):
        print(f"INPUT LOSS: {x1.shape, x2.shape, labels.shape, labels}")
        similarities = nn.functional.cosine_similarity(x1, x2, dim=1) / self.temperature
        print(f"similarities.shape: {similarities.shape}")

        positive_pairs = similarities[labels == 1]
        negative_pairs = similarities[labels == 0]
        print(f"positive_pairs.shape: {positive_pairs.shape}, negative_pairs.shape: {negative_pairs.shape}")

        positive_loss = -torch.log(positive_pairs).mean() if len(positive_pairs) > 0 else \
            torch.tensor(0.0, device=x1.device)
        negative_loss = -torch.log(1 - negative_pairs).mean() if len(negative_pairs) > 0 else \
            torch.tensor(0.0, device=x1.device)

        loss = positive_loss + negative_loss
        return loss


class PreTrainer:
    def __init__(self, encoder, contrastive_dataset, num_epochs, batch_size, learning_rate, patch_size):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = encoder
        self.contrastive_dataset = contrastive_dataset
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.patch_size = patch_size

    def pre_train(self):
        contrastive_dataloader = DataLoader(self.contrastive_dataset, batch_size=self.batch_size, shuffle=True)
        contrastive_loss = ContrastiveLoss()
        optimizer = optim.Adam(self.encoder.parameters(), lr=self.learning_rate)
        self.encoder.to(device=self.device, dtype=torch.float)

        for epoch in range(self.num_epochs):
            for batch in contrastive_dataloader:
                pairs, labels = batch
                x1, x2 = pairs
                x1, x2 = x1.to(device=self.device, dtype=torch.float), x2.to(device=self.device, dtype=torch.float)
                labels = labels.to(device=self.device, dtype=torch.long)
                repr1, repr2 = self.encoder(x1), self.encoder(x2)
                loss = contrastive_loss(repr1, repr2, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                wandb.log({
                    "Epoch": epoch + 1,
                    "Training Loss": loss.item()
                })

            # print(f'Epoch {epoch + 1}/{self.num_epochs}, Loss: {loss.item():.4f}')

        encoder_weights = (self.encoder.encoder_conv1.weight.data, self.encoder.encoder_conv2.weight.data,
                           self.encoder.encoder_conv3.weight.data, self.encoder.encoder_conv4.weight.data,
                           self.encoder.encoder_conv5.weight.data)
        encoder_biases = (self.encoder.encoder_conv1.bias.data, self.encoder.encoder_conv2.bias.data,
                          self.encoder.encoder_conv3.bias.data, self.encoder.encoder_conv4.bias.data,
                          self.encoder.encoder_conv5.bias.data)

        return encoder_weights, encoder_biases


def main(args):
    # DATA LOADING
    contrastive_dataset = MMWHSContrastiveDataset(folder_path=args.contrastive_folder_path, patch_size=args.patch_size,
                                                  patches_filter=args.patches_filter)

    # SET UP WEIGHTS & BIASES
    wandb.login(key="ef43996df858440ef6e65e9f7562a84ad0c407ea")
    wandb.init(
        entity="marco-n",
        project="local-contrastive-learning",
        config={
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "patch_size": args.patch_size,
            "patches_filter": args.patches_filter
        }
    )

    # CONTRASTIVE LEARNING
    encoder = Encoder()
    pre_trainer = PreTrainer(encoder=encoder, contrastive_dataset=contrastive_dataset, num_epochs=args.num_epochs,
                             batch_size=args.batch_size, learning_rate=args.learning_rate, patch_size=args.patch_size)
    encoder_weights, encoder_biases = pre_trainer.pre_train()
    torch.save({'encoder_weights': encoder_weights, 'encoder_biases': encoder_biases}, 'pretrained_encoder.pth')


if __name__ == "__main__":
    args = parse_args()
    main(args)
