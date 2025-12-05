from luma.core.interface.serial import spi
from luma.lcd.device import st7735
from luma.core.render import canvas
from gpiozero import Device
from gpiozero.pins.lgpio import LGPIOFactory

Device.pin_factory = LGPIOFactory()  # use lgpio backend

serial = spi(port=0, device=0, gpio_DC=23, gpio_RST=24)
device = st7735(serial, width=128, height=160)

with canvas(device) as draw:
    draw.text((0,0), "TFT is ready!", fill="white")
