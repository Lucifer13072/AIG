import torch
import json
from PIL import Image
import torchvision.transforms as transforms
from model import CustomTextEncoder, UNet, DiffusionModel  # Импорт твоих классов

# ===== 1. Загрузка обученной модели =====
device = "cuda" if torch.cuda.is_available() else "cpu"

text_encoder = CustomTextEncoder().to(device)
unet = UNet().to(device)
diffusion = DiffusionModel(unet).to(device)

text_encoder.load_state_dict(torch.load("checkpoints/text_encoder.pth", map_location=device))
unet.load_state_dict(torch.load("checkpoints/unet.pth", map_location=device))

text_encoder.eval()
unet.eval()
diffusion.eval()

# ===== 2. Функция для генерации изображения =====
def generate_image(prompt, style, steps=50):
    # Преобразуем промт в тензор
    tokens = torch.randint(0, 30522, (1, 200), dtype=torch.long).to(device)  # Тут должен быть твой токенизатор
    text_embedding = text_encoder(tokens)

    # Получаем индекс стиля
    with open("data/dataset.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    styles = {style_name: i for i, style_name in enumerate(set(d["Style"] for d in data.values()))}
    
    if style not in styles:
        raise ValueError(f"Неизвестный стиль '{style}'. Доступные стили: {list(styles.keys())}")

    style_tensor = torch.tensor([styles[style]], dtype=torch.long, device=device)

    # Начальный шум
    img = torch.randn((1, 3, 256, 256), device=device)

    # Прогоняем через диффузионную модель
    for t in reversed(range(steps)):
        img = diffusion(img, style_tensor, torch.tensor([t], device=device))

    # Обратное преобразование в изображение
    transform = transforms.ToPILImage()
    img = transform(img.squeeze(0).cpu().detach().clamp(0, 1))

    return img

# ===== 3. Запуск генерации =====
prompt = "Современная улица"  # Пример промта
style = "Reality"  # Пример стиля

generated_img = generate_image(prompt, style)
generated_img.save("generated_image.png")
generated_img.show()
