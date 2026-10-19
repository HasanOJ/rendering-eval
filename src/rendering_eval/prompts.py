from .config import INSTRUCTION


def messages_for(example, mode, image=None):
    if mode == "text":
        content = [{"type": "text", "text": example.task_text()}]
    elif mode == "image" and image is not None:
        content = [{"type": "image", "image": image}]
    else:
        raise ValueError("Expected text mode, or image mode with an in-memory image")
    content.append({"type": "text", "text": INSTRUCTION})
    return [{"role": "user", "content": content}]
