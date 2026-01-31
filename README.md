# A Minimal Implementation of a Flow Matching Inspired Training-Free Model for Probabilistic Forecasting

## **Running Examples**
First, install packages from `requirements.txt`. Then we can run the code with simple examples below.

```
python main.py --system dho --mode topk --device cuda:0 --integrator euler --ode_steps 100 --topk 64
```

```
python main.py --system lorenz63 --mode topk --device cuda:0 --integrator euler --ode_steps 100 --topk 256

```

Work in progress. More details coming soon!

