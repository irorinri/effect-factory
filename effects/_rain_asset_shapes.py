import math

from PIL import Image, ImageDraw, ImageFilter


BUILTIN_RAIN_SPRITE_IDS = ("drop", "circle", "square", "star")
_BUILTIN_RAIN_SPRITE_PREFIX = "builtin:png_rain:"


def builtin_rain_sprite_token(sprite_id: str) -> str:
    sprite_id = str(sprite_id or "").strip().lower()
    return _BUILTIN_RAIN_SPRITE_PREFIX + sprite_id


def parse_builtin_rain_sprite_token(value: str):
    text = str(value or "").strip().lower()
    if not text.startswith(_BUILTIN_RAIN_SPRITE_PREFIX):
        return None
    sprite_id = text[len(_BUILTIN_RAIN_SPRITE_PREFIX):]
    return sprite_id if sprite_id in BUILTIN_RAIN_SPRITE_IDS else None


def _finalize_sprite(img: Image.Image, blur_radius: float) -> Image.Image:
    if blur_radius > 0.0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    try:
        bbox = img.getchannel("A").getbbox()
    except Exception:
        bbox = None
    if bbox:
        img = img.crop(bbox)
    return img


def make_builtin_rain_sprite(sprite_id: str, size: int = 96) -> Image.Image:
    sprite_id = parse_builtin_rain_sprite_token(sprite_id) or str(sprite_id or "").strip().lower()
    if sprite_id not in BUILTIN_RAIN_SPRITE_IDS:
        sprite_id = "drop"
    size = max(24, int(size))
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    fill = (255, 255, 255, 236)
    inset = size * 0.18
    if sprite_id == "circle":
        draw.ellipse((inset, inset, size - inset, size - inset), fill=fill)
        return _finalize_sprite(img, blur_radius=size / 120.0)
    if sprite_id == "square":
        draw.rectangle((inset, inset, size - inset, size - inset), fill=fill)
        return _finalize_sprite(img, blur_radius=size / 140.0)
    if sprite_id == "star":
        cx = size / 2.0
        cy = size / 2.0
        outer = size * 0.34
        inner = outer * 0.46
        points = []
        for idx in range(10):
            angle = -math.pi / 2.0 + idx * math.pi / 5.0
            radius = outer if idx % 2 == 0 else inner
            points.append((cx + math.cos(angle) * radius, cy + math.sin(angle) * radius))
        draw.polygon(points, fill=fill)
        return _finalize_sprite(img, blur_radius=size / 150.0)
    # Blend the tip into the body with overlapping fills so the built-in drop reads as one shape.
    draw.polygon(
        [
            (size * 0.50, size * 0.06),
            (size * 0.65, size * 0.24),
            (size * 0.61, size * 0.42),
            (size * 0.39, size * 0.42),
            (size * 0.35, size * 0.24),
        ],
        fill=(255, 255, 255, 240),
    )
    draw.ellipse(
        (
            size * 0.31,
            size * 0.20,
            size * 0.69,
            size * 0.60,
        ),
        fill=(255, 255, 255, 238),
    )
    draw.ellipse(
        (
            size * 0.22,
            size * 0.32,
            size * 0.78,
            size * 0.94,
        ),
        fill=(255, 255, 255, 234),
    )
    return _finalize_sprite(img, blur_radius=size / 96.0)
