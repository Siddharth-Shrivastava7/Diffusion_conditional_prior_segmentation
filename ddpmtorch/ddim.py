"""
Use the deterministic generative process proposed by Song et al. (2020) [1]
[1] Song, Jiaming, Chenlin Meng, and Stefano Ermon. "Denoising Diffusion Implicit Models." International Conference on Learning Representations. 2020.
source file: https://github.com/ermongroup/ddim/blob/main/runners/diffusion.py, Ln 342-356
"""  # noqa
import math
import torch
import ddpm_torch


# def get_selection_schedule(schedule, size, timesteps):
#     """
#     :param schedule: selection schedule
#     :param size: length of subsequence
#     :param timesteps: total timesteps of pretrained ddpm model
#     :return: a mapping from subsequence index to original one
#     """
#     assert schedule in {"linear", "quadratic"}
#     power = 1 if schedule == "linear" else 2
#     c = timesteps / size ** power
#
#     def subsequence(t: np.ndarray):
#         return np.floor(c * np.power(t + 1, power) - 1).astype(np.int64)
#     return subsequence


def get_selection_schedule(schedule, size, timesteps):
    """
    :param schedule: selection schedule
    :param size: length of subsequence
    :param timesteps: total timesteps of pretrained ddpm model
    :return: subsequence
    """
    assert schedule in {"linear", "quadratic"}

    if schedule == "linear":
        subsequence = torch.arange(0, timesteps, timesteps // size)
    else:
        subsequence = torch.pow(torch.linspace(0, math.sqrt(timesteps * 0.8), size), 2).round().to(torch.int32)  # noqa

    return subsequence


class DDIM(ddpm_torch.GaussianDiffusion):
    def __init__(self, betas, model_mean_type, model_var_type, loss_type, eta, subsequence):
        super().__init__(betas, model_mean_type, model_var_type, loss_type)
        self.eta = eta  # coefficient between [0, 1] that decides the behavior of generative process
        self.subsequence = subsequence  # subsequence of the accelerated generation

        eta2 = eta ** 2
        assert not (eta2 != 1. and model_var_type != "fixed-small"), \
            'Cannot use DDIM (eta < 1) with var type other than "fixed-small"'

        self.alphas_bar = self.alphas_bar[subsequence]
        self.alphas_bar_prev = torch.cat([torch.ones(1, dtype=torch.float64), self.alphas_bar[:-1]], dim=0)
        self.alphas = self.alphas_bar / self.alphas_bar_prev
        self.betas = 1. - self.alphas
        self.sqrt_alphas_bar_prev = torch.sqrt(self.alphas_bar_prev)

        # q(x_t|x_0)
        # re-parameterization: x_t(x_0, \epsilon_t)
        self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar)
        self.sqrt_one_minus_alphas_bar = torch.sqrt(1. - self.alphas_bar)

        self.posterior_var = self.betas * (1. - self.alphas_bar_prev) / (1. - self.alphas_bar) * eta2
        self.posterior_logvar_clipped = torch.log(torch.cat([
            self.posterior_var[[1]], self.posterior_var[1:]]).clip(min=1e-20))

        # coefficients to recover x_0 from x_t and \epsilon_t
        self.sqrt_recip_alphas_bar = torch.sqrt(1. / self.alphas_bar)
        self.sqrt_recip_m1_alphas_bar = torch.sqrt(1. / self.alphas_bar - 1.)

        # coefficients to calculate E[x_{t-1}|x_0, x_t]
        self.posterior_mean_coef2 = torch.sqrt(
            1 - self.alphas_bar - eta2 * self.betas
        ) * torch.sqrt(1 - self.alphas_bar_prev) / (1. - self.alphas_bar)
        self.posterior_mean_coef1 = self.sqrt_alphas_bar_prev * \
                                    (1. - torch.sqrt(self.alphas) * self.posterior_mean_coef2)

        # for fixed model_var_type's
        self.fixed_model_var, self.fixed_model_logvar = {
            "fixed-large": (
                self.betas, torch.log(
                    torch.cat([self.posterior_var[[1]], self.betas[1:]]).clip(min=1e-20))),
            "fixed-small": (self.posterior_var, self.posterior_logvar_clipped)
        }[self.model_var_type]

        self.subsequence = torch.as_tensor(subsequence)

    @torch.inference_mode()
    def p_sample(self, denoise_fn, shape, device=torch.device("cpu"), noise=None):
        S = len(self.subsequence)
        B, *_ = shape
        subsequence = self.subsequence.to(device)
        _denoise_fn = lambda x, t: denoise_fn(x, subsequence.gather(0, t))
        t = torch.empty((B, ), dtype=torch.int64, device=device)
        if noise is None:
            x_t = torch.randn(shape, device=device)
        else:
            x_t = noise.to(device)
        for ti in range(S - 1, -1, -1):
            t.fill_(ti)
            x_t = self.p_sample_step(_denoise_fn, x_t, t)
        return x_t

    @staticmethod
    def from_ddpm(diffusion, eta, subsequence):
        return DDIM(**{
            k: diffusion.__dict__.get(k, None)
            for k in ["betas", "model_mean_type", "model_var_type", "loss_type"]
        }, eta=eta, subsequence=subsequence)


if __name__ == "__main__":
    from ddpm_torch import GaussianDiffusion, get_beta_schedule

    subsequence = get_selection_schedule("linear", 10, 1000)
    print(subsequence)
    betas = get_beta_schedule("linear", 0.0001, 0.02, 1000)
    diffusion = GaussianDiffusion(betas, "eps", "fixed-small", "mse")
    print(diffusion.__dict__)
    print(DDIM.from_ddpm(diffusion, eta=0., subsequence=subsequence).__dict__)
