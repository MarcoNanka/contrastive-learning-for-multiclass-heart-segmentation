import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
import wandb
from model import LocalEncoder, DomainEncoder
from data_loading import MMWHSLocalContrastiveDataset, MMWHSDomainContrastiveDataset
from config import parse_args
import os
import random

os.environ['WANDB_CACHE_DIR'] = "$HOME/wandb_tmp"
os.environ['WANDB_CONFIG_DIR'] = "$HOME/wandb_tmp"
os.environ['WANDB_DIR'] = "$HOME/wandb_tmp"
os.environ['WANDB_TEMP'] = "$HOME/wandb_tmp"


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=1):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, x1, x2, labels):
        similarities = nn.functional.cosine_similarity(x1, x2, dim=1) / self.temperature
        similarities = torch.clamp(similarities, min=-1, max=1)

        positive_pairs = similarities[labels == 1]
        negative_pairs = similarities[labels == 0]

        epsilon = 1e-8  # A small positive constant to avoid log(0) and log(1) issues
        positive_loss = -torch.log(positive_pairs + epsilon).mean() if len(positive_pairs) > 0 else \
            torch.tensor(0.0, device=x1.device)
        negative_loss = -torch.log(1 - negative_pairs + epsilon).mean() if len(negative_pairs) > 0 else \
            torch.tensor(0.0, device=x1.device)

        loss = positive_loss + negative_loss
        return loss


class DistanceAdjustedContrastiveLoss(nn.Module):
    def __init__(self, temperature, num_of_partitions):
        super(DistanceAdjustedContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.num_of_partitions = num_of_partitions

    def forward(self, x1, x2, distances):
        similarities = nn.functional.cosine_similarity(x1, x2, dim=1) / self.temperature
        similarities = torch.clamp(similarities, min=-1, max=1)

        weights = torch.pow(1 - distances / (self.num_of_partitions - 1), 4)  # low distance, high weight
        losses = []
        for i in range(similarities.size(0)):
            current_loss = nn.functional.mse_loss(similarities[i],
                                                  torch.full_like(similarities[i], fill_value=weights[i]))
            losses.append(current_loss)

        loss = sum(losses) / len(losses)
        return loss


class PreTrainer:
    def __init__(self, encoder, contrastive_dataset, num_epochs, batch_size, learning_rate, patch_size,
                 training_shuffle, patience, contrastive_type, is_distance_adjusted, temperature):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = encoder
        self.contrastive_dataset = contrastive_dataset
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.patch_size = patch_size
        self.training_shuffle = training_shuffle
        self.patience = patience
        self.contrastive_type = contrastive_type
        self.is_distance_adjusted = is_distance_adjusted
        self.temperature = temperature

    def pre_train(self):
        contrastive_loss = ContrastiveLoss(temperature=self.temperature) if not self.is_distance_adjusted else \
            DistanceAdjustedContrastiveLoss(temperature=self.temperature,
                                            num_of_partitions=self.contrastive_dataset.num_of_partitions)
        optimizer = optim.Adam(self.encoder.parameters(), lr=self.learning_rate)
        self.encoder.to(device=self.device, dtype=torch.float)
        num_patches = len(self.contrastive_dataset)
        num_patches_to_use = int(self.training_shuffle * num_patches)
        indices = list(range(num_patches))
        no_improvement_counter = 0
        best_loss = 5.0

        for epoch in range(self.num_epochs):
            random.shuffle(indices)
            selected_indices = indices[:num_patches_to_use]
            contrastive_dataloader = DataLoader(self.contrastive_dataset, batch_size=self.batch_size, shuffle=False,
                                                sampler=torch.utils.data.SubsetRandomSampler(selected_indices))
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

            if loss.item() < best_loss:
                best_loss = loss.item()
                selected_layers = (self.encoder.encoder_conv1, self.encoder.encoder_conv2,
                                   self.encoder.encoder_conv3, self.encoder.encoder_conv4)

                best_encoder_weights = tuple(layer.weight.data for layer in selected_layers) + (
                    self.encoder.encoder_conv5.weight.data,) if self.contrastive_type == "local" else tuple(
                    layer.weight.data for layer in selected_layers)

                best_encoder_biases = tuple(layer.bias.data for layer in selected_layers) + (
                    self.encoder.encoder_conv5.bias.data,) if self.contrastive_type == "local" else tuple(
                    layer.bias.data for layer in selected_layers)

                no_improvement_counter = 0
            else:
                no_improvement_counter += 1
            if no_improvement_counter > self.patience:
                print(f'Early stopping at epoch {epoch + 1}. Best loss: {best_loss}')
                break
            wandb.log({
                "Epoch": epoch + 1,
                "Training Loss": loss.item(),
                "Best (baseline: dice, contrastive: loss)": best_loss
            })
            print(f'Epoch {epoch + 1}/{self.num_epochs}, Loss: {loss.item():.4f}')

        return best_encoder_weights, best_encoder_biases


def main(args):
    # DATA LOADING
    is_distance_adjusted = args.contrastive_type not in ["domain", "local"]
    image_type = "CT"
    if "mr" in args.folder_path:
        image_type = "MRI"
    if args.contrastive_type == "local":
        contrastive_dataset = MMWHSLocalContrastiveDataset(folder_path=args.folder_path,
                                                           patch_size=args.patch_size,
                                                           removal_percentage=args.removal_percentage,
                                                           image_type=image_type)
    elif "domain" in args.contrastive_type:
        contrastive_dataset = MMWHSDomainContrastiveDataset(folder_path=args.folder_path,
                                                            patch_size=args.patch_size,
                                                            image_type=image_type,
                                                            is_distance_adjusted=is_distance_adjusted)
    else:
        raise ValueError(f"{args.contrastive_type} must be domain or local")

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
            "patience": args.patience,
            "training type": "CONTRASTIVE",
            "contrastive type": args.contrastive_type,
            "image_type": image_type,
            "filter": args.removal_percentage,
            "model_name": args.model_name,
            "folder_path": args.folder_path
        }
    )

    # CONTRASTIVE LEARNING
    if args.contrastive_type == "local" and os.path.isfile("pretrained_encoder/domain/" + args.encoder_file_name):
        pretrained_encoder = torch.load("pretrained_encoder/domain/" + args.encoder_file_name)
        encoder_weights, encoder_biases = pretrained_encoder['encoder_weights'], pretrained_encoder['encoder_biases']
        print("Pre-trained encoder is LOADED")
    else:
        encoder_weights, encoder_biases = None, None
        print("Pre-trained encoder does NOT EXIST")
    encoder = LocalEncoder(encoder_weights=encoder_weights, encoder_biases=encoder_biases) if \
        args.contrastive_type == "local" else DomainEncoder()
    pre_trainer = PreTrainer(encoder=encoder, contrastive_dataset=contrastive_dataset, num_epochs=args.num_epochs,
                             batch_size=args.batch_size, learning_rate=args.learning_rate, patch_size=args.patch_size,
                             training_shuffle=args.training_shuffle, patience=args.patience,
                             contrastive_type=args.contrastive_type, is_distance_adjusted=is_distance_adjusted,
                             temperature=args.temperature)
    encoder_weights, encoder_biases = pre_trainer.pre_train()
    save_path = f"pretrained_encoder/{'local' if args.contrastive_type == 'local' else 'domain'}/{args.model_name}"
    torch.save({'encoder_weights': encoder_weights, 'encoder_biases': encoder_biases}, save_path)


if __name__ == "__main__":
    args = parse_args()
    main(args)
