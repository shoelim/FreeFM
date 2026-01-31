# Minimal Implementation of the Proposed Training-Free ODE Model for Probabilistic Forecasting

## **Setup**

First, build the container image from `ours.def` (installs packages from `requirements.txt`):
```
apptainer build ours.sif ours.def
```

## **Running Example**
Once you have created the directories for checkpoints and results, you can then run simple examples inside the container:

```
apptainer exec ours.sif python main.py --system dho --mode topk --device cuda:0 --integrator euler --ode_steps 100 --topk 64
```

```
apptainer exec ours.sif python main.py --system lorenz63 --mode topk --device cuda:0 --integrator euler --ode_steps 100 --topk 256
```
