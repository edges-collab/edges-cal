## Combining Linear Models

There are two ways to combine linear models. The result of combining linear models must be another linear model. The two ways to combine linear models are:

1. Appending the models: i.e. the resulting model has a number of terms `n1 + n2 + ...`
   where `n1` is the number of terms in the first "sub-model" etc.
2. Cross-multiplying the models. If each sub-model has the same number of terms, then
   we can combine models by cross-multiplying the basis functions. Letting the first
   models' basis functions be `f1, f2, ..., fn` and the second models' basis functions
   be `g1, g2, ..., gn`, then the resulting model has basis functions
   `f1*g1, f1*g2, ..., fn*gn`.

In the rest of this tutorial, we will demonstrate how to combine linear models using the
`modelling` package in both of these ways.

### Appending Models

To append models, we use the `CompositeModel` class. The `CompositeModel` class takes a dict of models (with keys being names of each submodel) as input. The `CompositeModel` class has the same look and feel as  the `Model` class, so it can be used in the same way as any other model.

For example, let's combine two linear models:

```python
from edges_cal.modelling import Polynomial, LinLog, CompositeModel

model_a = Polynomial(n_terms=2)
model_b = LinLog(n_terms=3)

composite_model = CompositeModel({
    'model_a': model_a,
    'model_b': model_b
})
```

Now the composite model has 5 terms in total. We can fit the model to some data in the same way as we would fit a single model:

```python
import numpy as np

x = np.linspace(0, 1, 100)
y = 2*x**2 + 3*np.log(x) + 1

fit = composite_model.fit(x, y)

print(fit.parameters)
```

Not only that, but we can also access the submodels and their parameters. The
`fit` we just created is a normal `ModelFit` object. The `model` attribute of the fit is an instance of a `FixedLinearModel` object, with the parameters set to
the best-fit parameters of the fit. The `model` attribute of the `FixedLinearModel` is a `CompositeModel` object, with the submodels as attributes. The parameters of the submodels are stored in the `parameters` attribute of the `CompositeModel` object.

```python
best_model = fit.model.model
print(best_model['model_a'].parameters)
print(best_model['model_a'](x=x))
```

Also, note that each of the sub-models can have different transformations applied to them -- either x-transforms or basis-scaling.

```python
model_c = Polynomial(n_terms=2, basis_scaler=lambda x: x**2)
model_d = LinLog(n_terms=3, basis_scaler=lambda x: np.cos(x))

composite_model = CompositeModel({
    'model_c': model_c,
    'model_d': model_d
})
```

You can then evaluate the sub-models with or without the transformations applied:

```python
print(composite_model['model_c'].get_basis_term(indx=0, x=x))
```
