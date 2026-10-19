"""Notebook visualization. Display handles can be cleared before saving the notebook."""

from .backends import encode, image_input_metadata, load_processor, reconstruct_processor_image
from .rendering import DocumentRenderer


def preview_example(config, example, *, renderer=None, processor=None):
    from IPython.display import display
    from PIL import Image, ImageDraw

    renderer = renderer if renderer is not None else DocumentRenderer(config)
    original, render_info = renderer.render(example)
    if config.backend == "openrouter":
        try:
            handle = display(original, display_id=True)
        finally:
            original.close()
        return handle, {"language": example.language, "sample_id": example.sample_id,
                        **render_info, "image_preprocessing": "provider-managed; unavailable for inspection"}
    processor = processor if processor is not None else load_processor(config)
    inputs = encode(processor, config, example, "image", original)
    processed = reconstruct_processor_image(inputs, processor)
    processor_info = image_input_metadata(inputs, processor)
    # Use native pixels: the panel itself does not resize either input.
    panel = Image.new("RGB", (original.width + processed.width + 24, max(original.height, processed.height) + 48), "white")
    panel.paste(original, (0, 48))
    panel.paste(processed, (original.width + 24, 48))
    draw = ImageDraw.Draw(panel)
    draw.text((8, 8), f"{example.language}: rendered {original.width} x {original.height}", fill="black")
    draw.text((original.width + 32, 8), f"Processor input: {processed.width} x {processed.height}", fill="black")
    handle = display(panel, display_id=True)
    original.close()
    processed.close()
    panel.close()
    return handle, {"language": example.language, "sample_id": example.sample_id, **render_info, **processor_info}


def clear_previews(handles):
    """Replace displayed image outputs with text before saving the notebook."""
    for handle in handles:
        handle.update("Preview cleared; rerun the preview cell to display it again.")
    handles.clear()
