from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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

    def compute_gradients(self, *args, **kwargs):
        # args[1] could be our var_list
        if len(args) > 1:
            var_list = args[1]
        else:
            kwargs["var_list"] = kwargs.get("var_list") or self.variables
            var_list = kwargs["var_list"]
        if set(var_list) != set(self.variables):
            raise ValueError("var_list doesn't match with set of Fisher-estimating "
                             "variables.")
        return super().compute_gradients(*args, **kwargs)

    def apply_gradients(self, grads_and_vars, *args, **kwargs):
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
        # Update fisher with a session.run().
        fisher_updates = self._update_fisher(grads_and_vars,
                decay=self._fisher_momentum)

        with tf.control_dependencies(fisher_updates):
            # Calculate mu gradients. As a side effect, update momentum.
            mu_vel_and_mu_vars = self._update_momentum_and_mu(
                    grads_and_vars, decay=self._momentum)
        return mu_vel_and_mu_vars


    def _update_fisher(self, grads_and_vars, decay):
        """
        Takes the gradients of each weight with respect to loss,
        and updates the the fisher matrices associated with each weight.

        Params:
        grads_and_vars -- List (grad, w_var) pairs.

        beta2 -- A scalar Tensor or a scalar. The rate at which old Fisher
          diagonals decay.

        Return:
        update_list -- A list of all the the update operations.
        """
        beta2 = decay

        for grad, w_var in grads_and_vars:
            _, f = self.param_dict[var]
            new_f = (beta2 - 1) * f + (1 - beta2) * tf.square(grad)
            return f.assign(new_f)
        updates = [_update(grad, w) for grad, w in grads_and_vars]
        return updates


    def _update_momentum_and_mu(self, grads_and_vars, decay):
        """
        Use most recent velocity to update momentum. Then return the
        gradient updates to apply to each mu vector.

        This function assumes that fisher matrices have already been updated
        for this optimization step.

        Params:
        grads_and_vars -- List (grad, w_var) pairs. Gradients are
          Tensors that should be added to each var. Or None if
          the gradient shouldn't be applied.
        decay -- A scalar Tensor or a scalar. The rate at which old momentum
          decays.

        Returns:
        grads_and_mu_vars -- List of (grad, mu_var) pairs. Gradients are the
          Tensors that should be applied to each
        """
        beta1 = decay

        def _update(grad, var):
            m = self._zeros_slot(var, "momentum", self._name)
            with ops.colocate_with(m):
                # Compute the new momentum
                new_m = beta1 * m + (1 - beta1) * grad

                # TODO: Try this guy without array_ops.identity later.
                # I'm only using this out of superstition/tradition.
                new_m = array_ops.identity(m.assign(new_m), name="assign_m")

                mu, f = self.param_dict[var]
                vel = new_m / (1 - beta1) / (f + self.damping_total)
                return vel, mu

        return [_update(grad, w) for grad, w in grads_and_vars]
