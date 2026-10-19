"""Read pixel properties from the actual image rather than model predictions."""

from io import BytesIO

from PIL import Image


def image_properties(data: bytes) -> dict:
    with Image.open(BytesIO(data), formats=("PNG", "JPEG", "WEBP")) as image:
        width, height = image.size
        transparency = image.convert("RGBA").getchannel("A").getextrema()[0] < 255
        return {"width": width, "height": height, "transparency": transparency}
