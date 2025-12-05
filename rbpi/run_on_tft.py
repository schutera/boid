"""
If not yet done please install the luma library by
    # git clone https://github.com/rm-hull/luma.examples.git
    # cd luma.examples
    # sudo -H pip3 install -e .
(mind the dot at the end of the pip command)
"""

import importlib
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # render off-screen for TFT output

# Prefer the lgpio backend by default to avoid RPi.GPIO on Ubuntu.
os.environ.setdefault("LUMA_GPIO_INTERFACE", "lgpio")
os.environ.setdefault("LUMA_GPIO_CHIP", "/dev/gpiochip0")

from luma.core.interface.serial import spi, noop
from luma.lcd.device import st7735
from luma.core.render import canvas
from PIL import Image, ImageFont

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

import boid_minimal as minimal  # noqa: E402

FONT_PATH = Path("/home/luma.examples/examples/fonts/ChiKareGo.ttf")
DEFAULT_FONT = ImageFont.load_default()
FIGSIZE = (2.0, 1.6)  # approximately matches 160x128 aspect ratio
EDGE_BUFFER = 5
FRAME_DELAY = 0.05  # seconds between updates (~20 FPS)


def _resolve_lgpio_gpio():
    """Try to locate the lgpio helper class exposed by luma.core."""
    module_candidates = (
        "luma.core.lib.lgpio",
        "luma.core._lib.lgpio",
    )
    for module_name in module_candidates:
        spec = importlib.util.find_spec(module_name)
        if spec is None:
            continue
        module = importlib.import_module(module_name)
        gpio_cls = getattr(module, "GPIO", None)
        if gpio_cls is not None:
            return gpio_cls
    raise ImportError("lgpio GPIO helper not present in the installed luma.core package")


def load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype(str(FONT_PATH), size)
    except (OSError, IOError):
        return DEFAULT_FONT


def initialize_display():
    """Initialise the SPI TFT device, preferring the lgpio backend."""
    gpio = None
    try:
        GPIO = _resolve_lgpio_gpio()
        gpio = GPIO()
        serial = spi(port=0, device=0, cs_high=True, gpio=gpio, gpio_DC=23, gpio_RST=24)
    except (ImportError, RuntimeError) as exc:
        print(
            "lgpio backend not available; falling back to no-op GPIO.\n"
            "Install python3-lgpio (sudo apt install python3-lgpio) if your panel "
            "needs DC/RST control."
        )
        print(exc)
        os.environ["LUMA_GPIO_INTERFACE"] = "noop"
        serial = spi(port=0, device=0, cs_high=True, gpio=noop())
    device = st7735(
        serial,
        rotate=0,
        width=160,
        height=128,
        h_offset=0,
        v_offset=0,
        bgr=False,
    )
    return device, gpio


def show_text(device, headline: str, subline: str = ""):
    with canvas(device) as draw:
        draw.rectangle(device.bounding_box, outline="white", fill="black")
        draw.text((10, 40), headline, font=load_font(18), fill="red")
        if subline:
            draw.text((10, 70), subline, font=load_font(14), fill="white")


def run_boids(device):
    fig, ax = minimal.create_figure(figsize=FIGSIZE, edge_buffer=EDGE_BUFFER)
    print("[Press CTRL + C to end the script!]")
    try:
        while True:
            frame = minimal.render_frame(
                fig,
                ax,
                edge_buffer=EDGE_BUFFER,
                auto_step=True,
            )
            frame = frame.resize((device.width, device.height), Image.LANCZOS)
            device.display(frame.convert("RGB"))
            time.sleep(FRAME_DELAY)
    except KeyboardInterrupt:
        show_text(device, "Stopped", "CTRL + C detected")
        raise


def main():
    device, gpio = initialize_display()
    show_text(device, "Boid running!", "Initialising simulation")
    time.sleep(1.0)

    try:
        run_boids(device)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            device.clear()
            device.show()
        except Exception:
            pass
        if gpio is not None:
            try:
                gpio.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
