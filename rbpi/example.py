from luma.core.interface.serial import spi, noop
from luma.lcd.device import st7735
from luma.core.render import canvas

def _init_gpio_backend():
    """Load a GPIO backend compatible with the RPi.GPIO API."""
    try:
        import RPi.GPIO as GPIO
        GPIO.setmode(GPIO.BCM)
        return GPIO, None
    except (ImportError, RuntimeError) as exc:
        return None, exc


gpio_backend, gpio_error = _init_gpio_backend()

if gpio_backend is not None:
    serial = spi(port=0, device=0, gpio=gpio_backend, gpio_DC=23, gpio_RST=24)
else:
    print(
        "GPIO backend not available; continuing without explicit DC/RST control.\n"
        "Install rpi-lgpio (pip install rpi-lgpio) or python3-lgpio if your panel "
        "needs those lines."
    )
    if gpio_error is not None:
        print(f"Details: {gpio_error}")
    serial = spi(port=0, device=0, gpio=noop())

device = st7735(serial, width=160, height=128)

with canvas(device) as draw:
    draw.text((0,0), "TFT is ready!", fill="white")

if gpio_backend is not None:
    try:
        gpio_backend.cleanup()
    except Exception:
        pass
