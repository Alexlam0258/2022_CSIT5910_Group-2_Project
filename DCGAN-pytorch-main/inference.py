import os

import torch
from torchvision.utils import save_image

from dcgan import Generator128 as Generator
from utils import denorm


C = 3
H = 128 # 250
W = 128 # 250

number_of_images = 2000

# state_dict_path = 'G_state_dict_flower_250_batch32.pt'
state_dict_path = 'G_state_dict_test_test_128_batch128.pt'

def inference(output_dir):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    generator = Generator().to(device)
    generator.load_state_dict(torch.load(state_dict_path))
    generator.eval()
    with torch.inference_mode():
        for i in range(number_of_images):
            noise = torch.randn(1, 100, 1, 1, device=device)
            fake = generator(noise).detach().cpu()

            # reshape. C W H
            fake_image = fake.reshape(fake.size(0), C, W, H)
            fake_images_path = os.path.join(f"{output_dir}", f'fake_images_{i}.png')
            print(f"{fake_images_path}, fake_image.shape={fake_image.shape}")
            save_image(denorm(fake_image), fake_images_path)

if __name__ == '__main__':
    # Create a directory if not exists
    output_dir = f"inference{W}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    inference(output_dir)
