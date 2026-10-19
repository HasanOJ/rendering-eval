"""Fresh in-memory document canvases using sign/pixel-renderer's Pango backend."""

from hashlib import sha256
from pathlib import Path

from .config import Config, FONT_REVISION

FAMILIES = {
    "eng_Latn": "Noto Sans", "rus_Cyrl": "Noto Sans", "arb_Arab": "Noto Sans Arabic",
    "hin_Deva": "Noto Sans Devanagari", "ben_Beng": "Noto Sans Bengali",
    "tel_Telu": "Noto Sans Telugu", "tha_Thai": "Noto Sans Thai",
    "khm_Khmr": "Noto Sans Khmer", "mya_Mymr": "Noto Sans Myanmar",
    "zho_Hans": "Noto Sans SC", "jpn_Jpan": "Noto Sans JP", "kor_Hang": "Noto Sans KR",
}


class DocumentRenderer:
    def __init__(self, config: Config):
        from font_download import FontConfig
        from font_download.fonts import FontSource
        from font_configurator.font_configurator import FontConfigurator
        from font_configurator.fontconfig_managers import FontconfigMode
        # Reuse the specified backend, but not its single-line canvas cache.
        from pixel_renderer.renderer import Pango, PangoCairo, cairo, bgra_to_rgb

        self.config = config
        self.Pango, self.PangoCairo, self.cairo = Pango, PangoCairo, cairo
        self.bgra_to_rgb = bgra_to_rgb
        families = sorted({FAMILIES[x] for x in config.languages} | {"Noto Sans", "Noto Sans Symbols2"})
        sources = []
        for family in families:
            compact = family.replace(" ", "")
            axes = "wght" if family.endswith((" SC", " JP", " KR")) else "wdth,wght"
            filename = f"{compact}%5B{axes}%5D.ttf"
            if family == "Noto Sans Symbols2":
                filename = "NotoSansSymbols2-Regular.ttf"
            sources.append(FontSource(
                url=f"https://raw.githubusercontent.com/google/fonts/{FONT_REVISION}/ofl/{compact.lower()}/{filename}",
            ))
        font_dir = Path(FontConfig(sources=sources).get_font_dir())
        FontConfigurator().setup_font(
            mode=FontconfigMode.TEMPLATE_MINIMAL, font_dir=font_dir,
            fontconfig_destination_dir=font_dir, force_reinitialize=True,
        )
        self.font_metadata = {
            "revision": FONT_REVISION,
            "files": {p.name: sha256(p.read_bytes()).hexdigest() for p in sorted(font_dir.glob("*.ttf"))},
            "pango_version": Pango.version_string(), "cairo_version": cairo.cairo_version_string(),
        }

    def render(self, example):
        import numpy as np
        from PIL import Image

        cfg, Pango, PangoCairo, cairo = self.config, self.Pango, self.PangoCairo, self.cairo
        surface = cairo.ImageSurface(cairo.FORMAT_RGB24, 1, 1)
        layout = PangoCairo.create_layout(cairo.Context(surface))
        layout.set_width((cfg.width - 2 * cfg.margin) * Pango.SCALE)
        layout.set_wrap(Pango.WrapMode.WORD_CHAR)
        layout.set_auto_dir(True)
        layout.set_line_spacing(1.5)
        font = Pango.FontDescription()
        font.set_family(f"{FAMILIES[example.language]}, Noto Sans, Noto Sans Symbols2")
        font.set_weight(Pango.Weight.NORMAL)
        font.set_absolute_size(cfg.font_size * Pango.SCALE)
        layout.set_font_description(font)
        layout.set_text(example.task_text(), -1)
        if layout.get_unknown_glyphs_count():
            raise ValueError(f"Missing glyphs for {example.language}/{example.sample_id}")
        ink, logical = layout.get_pixel_extents()
        x = cfg.margin
        y = cfg.margin - min(0, ink.y, logical.y)
        height = y + max(ink.y + ink.height, logical.y + logical.height) + cfg.margin
        if x + ink.x < 0 or x + ink.x + ink.width > cfg.width:
            raise ValueError(f"Horizontal clipping for {example.language}/{example.sample_id}")
        if y + ink.y < 0 or y + ink.y + ink.height > height:
            raise ValueError(f"Vertical clipping for {example.language}/{example.sample_id}")

        actual_fonts = set()
        iterator = layout.get_iter()
        while True:
            run = iterator.get_run()
            if run is not None:
                # Keep the nested GI boxed owners alive while reading the font pointer.
                item = run.item
                analysis = item.analysis
                actual_font = analysis.font
                description = actual_font.describe()
                actual_fonts.add(description.get_family())
            if not iterator.next_run():
                break

        # No call to the upstream _get_canvas and no application-level image cache.
        surface = cairo.ImageSurface(cairo.FORMAT_RGB24, cfg.width, height)
        context = cairo.Context(surface)
        context.set_source_rgb(1, 1, 1)
        context.paint()
        context.set_source_rgb(0, 0, 0)
        context.move_to(x, y)
        PangoCairo.show_layout(context, layout)
        surface.flush()
        pixels = np.frombuffer(surface.get_data(), dtype=np.uint8).reshape(height, surface.get_stride() // 4, 4)
        image = Image.fromarray(self.bgra_to_rgb(pixels[:, :cfg.width]))
        return image, {
            "width": image.width, "height": image.height, "font_size": cfg.font_size,
            "actual_fonts": sorted(actual_fonts), "unknown_glyphs": 0,
            "ink_bounds": [x + ink.x, y + ink.y, ink.width, ink.height],
        }
