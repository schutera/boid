from luma.core.interface.serial import spi, noop
from luma.lcd.device import st7735
from luma.core.render import canvas

try:
    from luma.core.lib.lgpio import GPIO  # type: ignore

    gpio = GPIO()
    serial = spi(port=0, device=0, gpio=gpio, gpio_DC=23, gpio_RST=24)
except (ImportError, RuntimeError) as exc:
    gpio = None
    print(
        "lgpio backend not available; continuing without explicit DC/RST control.\n"
        "Install python3-lgpio (sudo apt install python3-lgpio) if your panel "
        "needs those lines."
    )
    print(exc)
    serial = spi(port=0, device=0, gpio=noop())

device = st7735(serial, width=128, height=160)

with canvas(device) as draw:
    draw.text((0,0), "TFT is ready!", fill="white")

if gpio is not None:
    try:
        gpio.close()
    except Exception:
        pass
