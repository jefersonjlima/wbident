# PSO Method for WhiteBox Identification System

## Overview
This repository contains the source code of WhiteBox Identification System


## USAGE

```bash
$ python3 setup.py install
```

```bash
$ python3 tests/linearfunction.py
```

Dynamic Model is define by:

```python
class EqSystem(Model):
    def __init__(self, params=None):
        super(EqSystem, self).__init__(params)
        self._params = params

    def model(self, t, x):
        # unknown coefficients
        k = self.unknown_const
        ks   = k[0]
        c    = k[1]
        # some constants
        m    = 1
        wn   = np.sqrt(k[0]/m)
        zeta = k[1]/(2*m*wn)
        dx = torch.zeros(len(self.x0),)
        # state space model
        dx[0] = x[1]
        dx[1] = -2 * zeta * wn * x[1] - wn ** 2 * x[0] + np.sin(2*np.pi*0.5*t)
        return dx
```


# Bugs & Feature Requests
Please report bugs and request features using the [issues](https://gitlab.com/limajj_articles/core/wbident/-/issues)
