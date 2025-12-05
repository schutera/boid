import importlib
import os


def _resolve_lgpio_gpio():
    """Try the known module paths that expose an lgpio-compatible GPIO class."""
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

# Ensure luma picks the lgpio backend first to avoid RPi.GPIO imports
os.environ.setdefault("LUMA_GPIO_INTERFACE", "lgpio")
os.environ.setdefault("LUMA_GPIO_CHIP", "/dev/gpiochip0")

from luma.core.interface.serial import spi, noop
from luma.lcd.device import st7735
from luma.core.render import canvas

try:
    GPIO = _resolve_lgpio_gpio()
    gpio = GPIO()
    serial = spi(port=0, device=0, gpio=gpio, gpio_DC=23, gpio_RST=24)
except (ImportError, RuntimeError) as exc:
    gpio = None
    # Force luma to skip lgpio/RPi-specific handlers when we cannot load lgpio.
    os.environ["LUMA_GPIO_INTERFACE"] = "noop"
    print(
        "lgpio backend not available; continuing without explicit DC/RST control.\n"
        "Install python3-lgpio (sudo apt install python3-lgpio) if your panel "
        "needs those lines."
    )
    print(f"Details: {exc}")
    serial = spi(port=0, device=0, gpio=noop())

device = st7735(serial, width=128, height=160)

with canvas(device) as draw:
    draw.text((0,0), "TFT is ready!", fill="white")

if gpio is not None:
    try:
        gpio.close()
    except Exception:
        pass
