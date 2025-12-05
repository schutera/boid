import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # render off-screen for TFT output

from luma.core.interface.serial import spi
from luma.lcd.device import st7735
from luma.core.render import canvas
from luma.core.lib.gpio import GPIO
from PIL import Image

# Ensure local modules are importable when script runs from other directories
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

import boid_minimal as minimal

# SPI and display configuration (adjust GPIO pins or rotation if your panel differs)
gpio = GPIO(chip="/dev/gpiochip0")
serial = spi(port=0, device=0, gpio=gpio, gpio_DC=23, gpio_RST=24)
device = st7735(serial, width=128, height=160, rotate=0)

# Optional: show a splash message before starting the simulation
with canvas(device) as draw:
    draw.rectangle(device.bounding_box, outline="white", fill="black")
    draw.text((10, 70), "Boids starting", fill="white")

# Create the matplotlib figure used for generating frames
# A smaller figsize speeds up rendering; edge_buffer keeps boids away from the borders
FIGSIZE = (2, 2)
EDGE_BUFFER = 5

fig, ax = minimal.create_figure(figsize=FIGSIZE, edge_buffer=EDGE_BUFFER)

FRAME_DELAY = 0.05  # seconds between updates (~20 FPS)

try:
    while True:
        frame = minimal.render_frame(
            fig,
            ax,
            edge_buffer=EDGE_BUFFER,
            auto_step=True,
            resize=(device.width, device.height),
        )
        # Ensure image matches the panel's expected color mode
        device.display(frame.convert("RGB"))
        time.sleep(FRAME_DELAY)
except KeyboardInterrupt:
    with canvas(device) as draw:
        draw.rectangle(device.bounding_box, outline="white", fill="black")
        draw.text((20, 70), "Stopped", fill="white")
finally:
    gpio.close()
```