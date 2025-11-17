import torch
import matplotlib.pyplot as plt


class LinearScheduler:

    def __init__(self, num_steps, beta_start, beta_end):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_steps = num_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.betas = torch.linspace(beta_start, beta_end, num_steps).to(self.device)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0).to(self.device)
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod).to(self.device)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alpha_cumprod).to(
            self.device
        )

    def add_noise(self, original, t, noise):
        original = original.to(self.device)
        noise = noise.to(self.device)
        
        original_shape = original.shape
        batch_size = original_shape[0]

        sqrt_alpha_cumprod = (
            self.sqrt_alpha_cumprod[t].reshape(batch_size)
        )
        sqrt_one_minus_alpha_cumprod = (
            self.sqrt_one_minus_alpha_cumprod[t].reshape(batch_size)
        )

        for _ in range(len(original_shape) - 1):
            sqrt_alpha_cumprod = sqrt_alpha_cumprod.unsqueeze(1)
            sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprod.unsqueeze(1)

        noisy = sqrt_alpha_cumprod * original + sqrt_one_minus_alpha_cumprod * noise
        return noisy

    def get_prev_timestep(self, xt, noise_prediction, timestep):
        xt = xt.to(self.device)
        noise_prediction = noise_prediction.to(self.device)
        
        x0 = (
            xt - self.sqrt_one_minus_alpha_cumprod[timestep] * noise_prediction
        ) / self.sqrt_alpha_cumprod[timestep]
        x0 = torch.clamp(x0, -1.0, 1.0)

        mean = (
            xt
            - self.betas[timestep]
            * noise_prediction
            / self.sqrt_one_minus_alpha_cumprod[timestep]
        )

        mean = mean / torch.sqrt(self.alphas[timestep])

        if timestep == 0:
            return mean, x0
        else:
            variance = (
                (1.0 - self.alpha_cumprod[timestep - 1])
                / (1.0 - self.alpha_cumprod[timestep])
                * self.betas[timestep]
            )
            sigma = torch.sqrt(variance)
            z = torch.randn(xt.shape, device=self.device)
            return mean + sigma * z, x0


if __name__ == "__main__":
    x = torch.from_numpy(plt.imread("train_image_0.png"))
    x = x[:, :, 0]
    x = x.reshape(1, 1, 28, 28)

    scheduler = LinearScheduler(1000, 0.0001, 0.02)
    noise = torch.randn(x.shape).to(x.device)
    noisy = scheduler.add_noise(x, 60, noise)
    print(noisy.shape)

    # save the noisy image to disk for debugging
    plt.imsave("noisy_image.png", noisy[0, 0, :, :].cpu().detach().numpy(), cmap=plt.cm.gray)
