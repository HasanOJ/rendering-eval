"""Notebook visualization. Display handles can be cleared before saving the notebook."""

from .backends import encode, image_input_metadata, load_processor, reconstruct_processor_image
from .rendering import DocumentRenderer


def _comparison_panel(original, processed, language):
    """Add readable preview headings around unchanged image pixels."""
    from pathlib import Path
    import matplotlib
    from PIL import Image, ImageDraw, ImageFont

    fonts = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    title_font = ImageFont.truetype(str(fonts / "DejaVuSans-Bold.ttf"), 32)
    detail_font = ImageFont.truetype(str(fonts / "DejaVuSans.ttf"), 24)
    header, gutter = 112, 24
    right = original.width + gutter
    panel = Image.new("RGB", (right + processed.width, max(original.height, processed.height) + header), "white")
    draw = ImageDraw.Draw(panel)
    draw.rectangle((0, 0, panel.width, header - 1), fill="#eef2f6")
    for x, title, detail in (
        (16, "Original rendering", f"{language} · {original.width} × {original.height} px"),
        (right + 16, "After Gemma preprocessing", f"{processed.width} × {processed.height} px · reconstructed pixels"),
    ):
        draw.text((x, 14), title, font=title_font, fill="#15283b")
        draw.text((x, 60), detail, font=detail_font, fill="#32465a")
    # These headings and the shared display canvas never enter model inputs.
    panel.paste(original, (0, header))
    panel.paste(processed, (right, header))
    return panel


def preview_example(config, example, *, renderer=None, processor=None):
    from IPython.display import display

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
    panel = _comparison_panel(original, processed, example.language)
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
