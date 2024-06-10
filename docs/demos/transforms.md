## Transformations

`modelling` allows for 3 kinds of transformations:

1. Transformations of the response variable (`x`)
2. Scaling of the basis functions by a function of the input (`s(x)`)
3. Transformations of the data (`t(d)`).

The first two are implemented at the `Model` level -- i.e. they are applied to the response variable and *all* the basis functions before fitting the model.
The third is implemented at the top level -- i.e. *either* on a `Model` or a
`CompositeModel`, whichever is the higher-level. Only one data-transformation can be
applied to any given data/model pairing, but in principle each sub-model might have
a different `x`-transformation and `s(x)`-transformation.
