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
