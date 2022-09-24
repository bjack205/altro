# altro
A Fast Solver for Constrained Trajectory Optimization

# Installation

## Arduino
This shows how to compile the code to use on a Teensy microcontroller.

1. Install Arduino CLI
```shell
cd ~/.local
curl -fsSL https://raw.githubusercontent.com/arduino/arduino-cli/master/install.sh | sh
```
2. Change directory back to altro root
```shell
cd <altro/root/directory>
```
3. Install Teensy Arduino libraries and Teensy rules
```shell
sudo cp resources/00-teensy.rules /etc/udev/rules.d/
arduino-cli core install teensy:avr --additional-urls https://www.pjrc.com/teensy/td_156/package_teensy_index.json
```
4.  If needed, add yourself to the dialout and tty groups
```shell
sudo usermod -a -G tty $USER 
sudo usermod -a -G dialout $USER 
```

