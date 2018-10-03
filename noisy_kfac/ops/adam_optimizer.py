from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.training import gradient_descent
from . import estimator as est


class AdamOptimizer(gradient_descent.GradientDescentOptimizer):
    """
    Noisy Adam Optimizer
    """

    def __init__(self,
                 learning_rate,
                 damping_int,
                 damping_ext,
                 param_dict,
                 fisher_momentum=0.,
                 momentum=0.,
                 name="adam"):
        """
        Params:
          param_dict: A dictionary {sample_weight => (mean, fisher_diag)}. All
            of the keys are Tensors, and all values are tuples of Variables. The
            keys of this dictionary becomes the list of Tensors that gradients
            of loss are taken with respect to. `mean` is the mean of the
            Gaussian distribution from which the sample_weight is sampled.
            `fisher_diag` is the diagonal of the Fisher matrix associated with
            this sampled weight. It is required for calculating the variance of
            Gaussian distribution from which the sample_weight is sampled.
        """

        self.param_dict = param_dict
        self.variables = list(param_dict.keys())
        # TODO: define damping_ext and damping_int.
        self.damping_int = damping_int
        self.damping_ext = damping_ext
        self.damping_total = self.damping_ext + self.damping_int

        self._momentum = momentum
        self._fisher_momentum = fisher_momentum

        super().__init__(learning_rate, name=name)

    def minimize(self, *args, **kwargs):
        kwargs["var_list"] = kwargs.get("var_list") or self.variables
        if set(kwargs["var_list"]) != set(self.variables):
            raise ValueError("var_list doesn't match with set of Fisher-estimating "
                             "variables.")
        return super().minimize(*args, **kwargs)

    def compute_gradients(self, loss, *args, **kwargs):
        # args[1] could be our var_list
        if len(args) > 1:
            var_list = args[1]
        else:
            kwargs["var_list"] = kwargs.get("var_list") or self.variables
            var_list = kwargs["var_list"]
        if set(var_list) != set(self.variables):
            raise ValueError("var_list doesn't match with set of Fisher-estimating "
                             "variables.")

        # # In case we aren't allowed to calculate Tensor derivatives here.
        # grads = tf.gradients(-loss, var_list)
        # grads_and_vars = list(zip(grads, var_list))
        # return grads_and_vars
        return super().compute_gradients(loss, *args, **kwargs)

    def apply_gradients(self, grads_and_vars, *args, **kwargs):
        """
        Apply gradient updates to the variational mean and variance parameters
        of each sample weight, given a list of gradients of each weight with
        respect to the loss.

        Params:
        grads_and_vars -- a list of (weight_grad, weight_tensor) pairs.
        """
        grads_and_vars = list(grads_and_vars)
        # A list of gradients and variables.
        steps_and_vars = self._compute_update_steps(grads_and_vars)
        return super().apply_gradients(steps_and_vars, *args, **kwargs)

    def _compute_update_steps(self, grads_and_vars):
        """
        Convert a list of (weight_grad, weight_tensor) pairs to a list of
        (mu_grad, mu_tensor). Mu is the mean of the variational distribution
        from which a weight is drawn.

        Also update fisher (which determines the variance of the variational
        distribution) and Adam momentum.
        """
        # We actually have a list of grads and tensors. We'll convert to
        # grads and vars along the way.
        grads_and_tensors = grads_and_vars

        # Produce fisher update operations.
        fisher_updates = self._update_fisher(grads_and_tensors,
                decay=self._fisher_momentum)

        # Calculate mu updates. As a side effect, update momentum.
        grad_mu_wb_list = self._update_momentum_and_mu(
                grads_and_vars, fisher_updates, decay=self._momentum)

        # Each mu is a concatentation of weight and bias variables.
        # Split the gradients and mus.
        grads_and_vars = self._split_grads_and_mus(grad_mu_wb_list)
        return grads_and_vars


    def _update_fisher(self, grads_and_vars, decay):
        """
        Takes the gradients of each weight with respect to loss,
        and updates the the fisher matrices associated with each weight.

        Params:
        grads_and_vars -- List (grad, w_var) pairs.

        beta2 -- A scalar Tensor or a scalar. The rate at which old Fisher
          diagonals decay.

        Return:
        fisher_updates -- A dictionary {w_var => new_f} mapping each weight
            vector to its newly assigned fisher diagonal.
        """
        beta2 = decay

        def _update(grad, w_var):
            # XXX: Ugly hack to change
            # (n_particles, n_in, n_out) => (n_in, n_out). Otherwise
            # the assignment won't work. In reality, n_particles should be 1
            # here, so this will have no effect.
            # TODO: assert n_particles==1.
            grad = tf.reduce_mean(grad, axis=0, name="avg_grad")

            f = self.param_dict[w_var]._f
            new_f = (beta2 - 1) * f + (1 - beta2) * tf.square(grad)
            return f.assign(new_f)
        update_dict = {w: _update(grad, w) for grad, w in grads_and_vars}
        return update_dict


    def _update_momentum_and_mu(self, grads_and_vars, fisher_updates, decay):
        """
        Use most recent velocity to update momentum. Then return the
        gradient updates to apply to each mu vector.

        This function assumes that fisher matrices have already been updated
        for this optimization step.

        Params:
        grads_and_vars -- List (grad, w_var) pairs. Gradients are
          Tensors that should be added to each var. Or None if
          the gradient shouldn't be applied.
        fisher_updates -- A dictionary {w_var => new_f} mapping each weight
            vector to its newly assigned fisher diagonal.
        decay -- A scalar Tensor or a scalar. The rate at which old momentum
          decays.

        Returns:
        vel_mu_wb_list -- List of (vel, mu_tensor, wb) triples.
        """
        def _update(grad, var):
            # XXX: Hack. Instead, I should assert that zero-th dimension
            # is 1, and then squeeze that dimension.
            grad = tf.reduce_mean(grad, axis=0, name="avg_grad")

            beta1 = decay
            zero = tf.zeros(shape=tf.shape(grad), dtype=float,
                    name="momentum_init")
            m = self._get_or_make_slot(var=var, val=zero, slot_name="momentum",
                    op_name="momentum")
            with ops.colocate_with(m):

                # Compute the new momentum
                new_m = beta1 * m + (1 - beta1) * grad
                new_m = tf.identity(m.assign(new_m), name="new_momentum")

                wb = self.param_dict[var]
                mu = wb._mean
                f = fisher_updates[var]
                vel = new_m / (1 - beta1) / (f + self.damping_total)
                return vel, mu, wb

        return [_update(grad, w) for grad, w in grads_and_vars]

    def _split_grads_and_mus(self, grad_mu_wb_list):
        """
        Params:
          grad_mu_wb_list: A list of (gradient, mean, wb) tuples.
        Returns:
          grads_and_vars: A list of (split_gradient, split_var) pairs,
            where each split_gradient and split_var is formed from
            unconcatentating gradient and mean Tensors.
        """
        res = []
        for grad, _, wb in grad_mu_wb_list:
            weight_grad_flat, bias_grad_expand = tf.split(grad,
                    [wb._n_in - 1, 1], axis=0)
            weight_grad = tf.reshape(weight_grad_flat, wb._shape,
                    name="weight_grad")
            bias_grad = tf.reshape(bias_grad_expand, [wb._n_out],
                    name="bias_grad")
            res.append((weight_grad, wb._weight))
            res.append((bias_grad, wb._bias))
        return res
