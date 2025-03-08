import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import json
from PIL import Image
import os
import matplotlib.pyplot as plt  
from tqdm import tqdm  # для отображения прогресса обучения

# ===== 1. Собственный текстовый энкодер =====
class CustomTextEncoder(nn.Module):
    def __init__(self, vocab_size=30522, embed_dim=256, hidden_dim=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
    
    def forward(self, tokens):
        x = self.embedding(tokens)
        _, (hidden, _) = self.lstm(x)
        return hidden[-1]  # Возвращаем последний скрытый слой

# ===== 2. UNet с FiLM для управления стилем =====
class UNet(nn.Module):
    def __init__(self, input_channels=3, style_dim=512, num_styles=100):
        super().__init__()
        # Эмбеддинг стиля
        self.style_embedding = nn.Embedding(num_styles, style_dim)
        
        # Используем ResNet18 как энкодер (без последних слоев)
        resnet = models.resnet18(pretrained=True)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])
        
        # FiLM-слой для модификации признаков по стилю
        self.film = nn.Linear(style_dim, 512)
        
        # Декодер: можно доработать, добавив skip-соединения для лучшей детализации
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, input_channels, kernel_size=3, padding=1)
        )

    def forward(self, x, style_idx):
        # Получаем эмбеддинг стиля и пропускаем через FiLM
        style_embedding = self.style_embedding(style_idx)
        x = self.encoder(x)
        gamma = self.film(style_embedding).view(-1, 512, 1, 1)
        x = x * gamma  # Применяем FiLM-модификацию
        x = self.decoder(x)
        return x

# ===== 3. Диффузионный процесс =====
class DiffusionModel(nn.Module):
    def __init__(self, unet, timesteps=1000):
        super().__init__()
        self.unet = unet
        self.timesteps = timesteps
        self.beta = torch.linspace(0.0001, 0.02, timesteps)
        self.alpha = 1.0 - self.beta
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0)
    
    def forward(self, x, style_idx, t):
        noise = torch.randn_like(x)
        alpha_t = self.alpha_cumprod.index_select(0, t).view(-1, 1, 1, 1).to(x.device)
        noisy_x = torch.sqrt(alpha_t) * x + torch.sqrt(1 - alpha_t) * noise
        return self.unet(noisy_x, style_idx)

# ===== 4. Контрастивное обучение и ImageEncoder =====
class ImageEncoder(nn.Module):
    def __init__(self, output_dim=512):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        # Изменили входной размер с 2048 на 512
        self.fc = nn.Linear(512, output_dim)
    
    def forward(self, x):
        x = self.encoder(x).view(x.size(0), -1)
        return self.fc(x)

# Улучшенная версия контрастивной потери: линейные слои проекции создаются один раз
class ContrastiveLoss(nn.Module):
    def __init__(self, image_dim, text_dim, projection_dim=512, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.img_proj = nn.Linear(image_dim, projection_dim)
        self.text_proj = nn.Linear(text_dim, projection_dim)
    
    def forward(self, image_embeddings, text_embeddings):
        image_embeddings = self.img_proj(image_embeddings)
        text_embeddings = self.text_proj(text_embeddings)
        
        image_embeddings = F.normalize(image_embeddings, dim=1)
        text_embeddings = F.normalize(text_embeddings, dim=1)
        
        logits = torch.matmul(image_embeddings, text_embeddings.T) / self.temperature
        labels = torch.arange(len(logits)).to(logits.device)
        loss = F.cross_entropy(logits, labels)
        return loss

# ===== 5. Датасет с текстовым кодированием =====
class ImageDataset(Dataset):
    def __init__(self, json_path, image_dir, text_encoder, transform=None, device="cuda"):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.image_dir = image_dir
        self.transform = transform
        self.device = device
        self.text_encoder = text_encoder.to(device)
        # Создаем словарь стилей
        self.styles = {style: i for i, style in enumerate(set(d["Style"] for d in self.data.values()))}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        key = list(self.data.keys())[idx]
        img_path = os.path.join(self.image_dir, f"{key}.png")
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        
        prompt = self.data[key]["promt"]
        tokens = torch.randint(0, 30522, (1, 20), dtype=torch.long).to(self.device)
        # Убираем лишнюю размерность, чтобы текстовый эмбеддинг имел форму (512,)
        with torch.no_grad():
            text_embedding = self.text_encoder(tokens).squeeze(0)
        
        style = self.styles[self.data[key]["Style"]]
        style_tensor = torch.tensor(style, dtype=torch.long, device=self.device)
        return img, text_embedding, style_tensor

# ===== 6. Инициализация и обучение =====
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    text_encoder = CustomTextEncoder().to(device)
    unet = UNet().to(device)
    diffusion = DiffusionModel(unet).to(device)
    image_encoder = ImageEncoder().to(device)

    contrastive_loss = ContrastiveLoss(image_dim=512, text_dim=512, projection_dim=512, temperature=0.07).to(device)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    dataset = ImageDataset("data/dataset.json", "data/images", text_encoder, transform, device)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

    optimizer = torch.optim.AdamW(
        list(diffusion.parameters()) +
        list(text_encoder.parameters()) +
        list(image_encoder.parameters()) +
        list(contrastive_loss.parameters()),
        lr=1e-4
    )

    epochs = 100
    epoch_losses = []
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        for imgs, text_embeddings, styles in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            imgs = imgs.to(device)
            text_embeddings = text_embeddings.to(device)
            styles = styles.to(device)

            optimizer.zero_grad()
            t = torch.randint(0, diffusion.timesteps, (imgs.size(0),), device=device)
            generated_imgs = diffusion(imgs, styles, t)
            img_embeds = image_encoder(generated_imgs)
            loss = contrastive_loss(img_embeds, text_embeddings)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        epoch_losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    plt.figure(figsize=(10, 5))
    plt.plot(epoch_losses, marker='o')
    plt.title("Потери по эпохам")
    plt.xlabel("Эпоха")
    plt.ylabel("Средний лосс")
    plt.grid(True)
    plt.show()

    cmd = int(input("Хотите ли сохранить модель? 0/1: "))
    if cmd == 0:
        # Пример сохранения моделей
        torch.save(text_encoder.state_dict(), "checkpoints/text_encoder.pth")
        torch.save(unet.state_dict(), "checkpoints/unet.pth")
        torch.save(diffusion.state_dict(), "checkpoints/diffusion.pth")
        torch.save(image_encoder.state_dict(), "checkpoints/image_encoder.pth")
        torch.save(contrastive_loss.state_dict(), "checkpoints/contrastive_loss.pth")
        print("Модели сохранены!")
    else:
        print("Обучение закончено")

if __name__ == '__main__':
    train()