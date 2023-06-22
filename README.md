# neural_collision
## Build 
You must have the nvidia tololkit installed to compile the cuda libraries. You can test your installation by running `nvcc --version`. 

```
mkdir -p ros_ws/src
cd ros_ws/src
git clone https://github.com/olaghattas/neural_collision.git
sudo apt update
rosdep install --from-paths src --ignore-src -y
colcon build

```

collision package:
contains config.json where the architecture is defined.
