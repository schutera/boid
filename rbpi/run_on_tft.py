"""
If not yet done please install the luma library by
    # git clone https://github.com/rm-hull/luma.examples.git
    # cd luma.examples
    # sudo -H pip3 install -e .
(mind the dot at the end of the pip command)
"""

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # render off-screen for TFT output

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
EDGE_BUFFER = 1
FRAME_DELAY = 0.035  # seconds between updates (~28 FPS)

def _init_gpio_backend():
    """Load a GPIO backend compatible with the RPi.GPIO API."""
    # Tip: if lgpio keeps failing, running 'python3 -m pip install rpi-lgpio' rescued this setup.
    try:
        import RPi.GPIO as GPIO
        GPIO.setmode(GPIO.BCM)
        return GPIO, None
    except (ImportError, RuntimeError) as exc:
        return None, exc


def load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype(str(FONT_PATH), size)
    except (OSError, IOError):
        return DEFAULT_FONT


def initialize_display():
    """Initialise the SPI TFT device, preferring a GPIO backend when available."""
    gpio_backend, gpio_error = _init_gpio_backend()

    if gpio_backend is not None:
        serial = spi(port=0, device=0, cs_high=True, gpio=gpio_backend, gpio_DC=23, gpio_RST=24)
    else:
        print(
            "GPIO backend not available; falling back to no-op GPIO.\n"
            "Install rpi-lgpio (pip install rpi-lgpio) or python3-lgpio if your panel "
            "needs DC/RST control."
        )
        if gpio_error is not None:
            print(gpio_error)
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
    return device, gpio_backend


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
    device, gpio_backend = initialize_display()
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
        if gpio_backend is not None:
            try:
                gpio_backend.cleanup()
            except Exception:
                pass


if __name__ == "__main__":
    main()
