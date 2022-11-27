from PIL import Image

def add_frame_to_gif(dataset_name, gif_frames, fake_images_path):
    with Image.open(fake_images_path) as frame:
        gif_frames.append(frame)
        gif_frames[0].save(f'{dataset_name}.gif', format='GIF',
                append_images=gif_frames[1:],
                save_all=True,
                duration=500, loop=1)


def denorm(x):
    # TANH [-1, 1]
    out = (x + 1) / 2
    return out.clamp(0, 1)