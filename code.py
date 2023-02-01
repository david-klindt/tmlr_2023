import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float
device = torch.device("mps")
print(device, dtype)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def neg_log_dist(y, y_):
    out = (y - y_)**2
    perfect_ind = out == 0.0
    # set perfect predictions to minimum error
    out[perfect_ind] = np.min(out[np.logical_not(perfect_ind)])
    out = np.log(out)
    assert not np.any(np.isnan(out))
    assert not np.any(np.isinf(out))
    return - out


class Data:
    def __init__(
            self,
            setting,
            batch_size=256,
            train_batch=np.inf,
            test_batch=1024,
            seed=42,
            num_grid=512,  # gets squared
            grid_limit=np.pi / 2,
            noise_level=1.0,
    ):
        self.setting = setting
        self.batch_size = batch_size
        self.train_batch = train_batch
        self.test_batch = test_batch
        self.seed = seed
        self.num_grid = num_grid
        self.grid_limit = grid_limit
        self.noise_level = noise_level
        self.x_grid, self.y_grid = self.make_grid()
        self.grid_batch = int(np.ceil(self.x_grid.shape[0] / self.batch_size))

        # initialize batch count for train
        self.batch_count = 0

    def get_train(self):
        self.batch_count += 1
        if self.batch_count >= self.train_batch:
            self.batch_count = 0
        # take every second seed
        return self.get_data(self.batch_size, self.seed + self.batch_count * 2)

    def get_test(self, ind):
        assert 0 <= ind < self.test_batch
        # take every second seed + 1
        return self.get_data(self.batch_size, self.seed + 1 + ind * 2)

    def get_grid(self, ind):
        assert 0 <= ind < self.grid_batch
        x = self.x_grid[ind * self.batch_size:(ind + 1) * self.batch_size].copy()
        y = self.y_grid[ind * self.batch_size:(ind + 1) * self.batch_size].copy()
        return x, y

    def make_grid(self, num_sample=None, limit=None):
        if num_sample is None:
            num_sample = self.num_grid
        if limit is None:
            limit = self.grid_limit
        x = np.linspace(-limit, limit, num_sample)
        if self.setting == 'sine':  # 1D regression
            y = np.sin(x)
            x = x[:, None]
        else:  # 2D addition
            x = np.stack(np.meshgrid(x, x), 2).reshape(-1, 2)
            y = x.sum(1)
        return x, y[:, None]

    def get_data(self, num_sample, seed):
        np.random.seed(seed)
        if self.setting == 'sine':
            x = np.random.uniform(-2 * np.pi, 2 * np.pi, num_sample)[:, None]
            y = np.sin(x) + np.random.normal(0, self.noise_level, x.shape)
        elif self.setting == 'gauss':
            x = np.random.normal(0, 1, (num_sample, 2))
            y = np.sum(x, axis=1, keepdims=True)
        elif self.setting == 'disc':
            angle = np.random.uniform(0, 2 * np.pi, num_sample)
            norm = np.random.uniform(0, 1, num_sample)**.5  # uniform sampling on disc
            x = np.stack([np.sin(angle), np.cos(angle)], 1) * norm[:, None]
            y = np.sum(x, axis=1, keepdims=True)
        elif self.setting == 'annulus':
            angle = np.random.uniform(0, 2 * np.pi, num_sample)
            norm = np.random.uniform(.25, 1, num_sample)**.5  # uniform sampling on annulus
            x = np.stack([np.sin(angle), np.cos(angle)], 1) * norm[:, None]
            y = np.sum(x, axis=1, keepdims=True)
        return x, y

    def get_ind(self):
        """indices of grid that is in (i.i.d.) and out (o.o.d.) of training domain."""
        if self.setting == 'disc':
            ind_iid = np.linalg.norm(self.x_grid, axis=1) <= 1  # in domain
        elif self.setting == 'annulus':
            ind_iid = np.logical_and(  # in domain
                .5 <= np.linalg.norm(self.x_grid, axis=1),
                np.linalg.norm(self.x_grid, axis=1) <= 1
            )
        elif self.setting == 'sine':
            ind_iid = abs(self.x_grid) < 2 * np.pi
        else:
            raise ValueError("iid/ood indices not defined for data domain: %s" % self.setting)
        ind_ood = np.logical_not(ind_iid)  # out of domain
        return ind_iid.flatten(), ind_ood.flatten()


class NormalizedLinear(torch.nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            sigma,
            normalization='std',
            use_bias=True,
            seed=42,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma = sigma
        self.normalization = normalization
        self.use_bias = use_bias
        torch.manual_seed(seed)
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        if self.normalization == 'std':
            # weights
            if self.in_features > 1 and self.out_features > 1:
                # w_std = torch.std(self.weight, dim=1, keepdim=True)
                # w_std = torch.std(self.weight, dim=0, keepdim=True)
                w_std = torch.std(self.weight)
            else:
                w_std = torch.std(self.weight)
            weight = self.weight / w_std
            # biases
            if self.out_features > 1:
                b_std = torch.std(self.bias)
                bias = self.bias / b_std
            else:
                bias = self.bias
        elif self.normalization == 'tanh':
            weight = torch.tanh(self.weight)
            bias = torch.tanh(self.bias)

        weight = weight * self.sigma / self.in_features ** .5
        bias = bias * self.sigma

        if not self.use_bias:
          bias = None

        return torch.nn.functional.linear(x, weight, bias=bias)


class Calculator(torch.nn.Module):
    def __init__(
            self,
            num_input,
            num_output,
            normalized,
            num_hidden,
            num_layer,
            sigma,  # different for layers?
            nonlinearity=torch.nn.Tanh,
            normalization='std',
            init='default',
            use_bias=True,
            seed=42,
    ):
        super().__init__()
        self.num_layer = num_layer
        self.sigma = sigma
        self.init = init
        torch.manual_seed(seed)
        self.layers = []
        for i in range(num_layer):
            if i == 0:  # first layer
                num_in, num_out = num_input, num_hidden
            elif i < num_layer - 1:  # hidden layers
                num_in, num_out = num_hidden, num_hidden
            else:  # last layer
                num_in, num_out = num_hidden, num_output
            if normalized:
                self.layers.append(
                    NormalizedLinear(num_in, num_out, sigma, normalization, 
                        seed=seed, use_bias=use_bias)
                )
            else:
                self.layers.append(torch.nn.Linear(
                    num_in, num_out, bias=use_bias))
            if i < num_layer - 1:  # no nonlinearity on last layer
                self.layers.append(nonlinearity())
        self.layers = torch.nn.Sequential(*self.layers)
        if init != 'default':
            self.layers.apply(self.init_weights)
        self.to(device)

    def init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.bias, mean=0.0, std=self.sigma)
            sigma = self.sigma
            if self.init == 'lecun':
                sigma /= m.weight.shape[1] ** .5  # sqrt(num_input)
            else:
                assert self.init == 'normal'
            torch.nn.init.normal_(m.weight, mean=0.0, std=sigma)


    def forward(self, x, get_hidden=False):
        if get_hidden:
            hidden_activations = [x]
            for l in self.layers:
                hidden_activations.append(l(hidden_activations[-1]))
            return hidden_activations
        else:
            return self.layers(x)


def train(model, data, num_step=10000, lr=1e-3, decayRate=1.0, 
          optimizer=torch.optim.Adam, plot_every=[], log_dir=''):
    #precise: num_step=50000, lr=1e-4, decayRate=0.9):
    if type(optimizer) == torch.optim.Adam:
      pass
    else:
      optimizer = optimizer(model.parameters(), lr=lr)
      #optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    if decayRate < 0:
        my_lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer=optimizer, base_lr=lr/10, max_lr=lr*10, step_size_up=2000, cycle_momentum=False)
    elif decayRate < 1:
        my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer, gamma=decayRate)
    run_loss = []
    train_log = {'mse_train': [], 'mse_test': []}
    if len(plot_every) > 0:
        train_log['filenames'] = []
        #train_log['nlds'] = []
    for l in range(model.num_layer):
        train_log['layer=%s_weight_std' % l] = []
        train_log['layer=%s_bias_std' % l] = []
    model.train()
    pbar = tqdm(np.arange(num_step + 1))
    for i in pbar:
        #for i in range(num_step + 1):
        x, y = data.get_train()
        x = torch.tensor(x, device=device, dtype=dtype)
        y = torch.tensor(y, device=device, dtype=dtype)
        optimizer.zero_grad()
        y_ = model(x)
        loss = torch.mean((y - y_) ** 2)
        loss.backward()
        optimizer.step()
        run_loss.append(loss.item())

        if i > 0:
            model.eval()
            if decayRate < 0:
                my_lr_scheduler.step()
            elif decayRate < 1 and not (i % 500):
                my_lr_scheduler.step()
            if not (i % 1000):
                x_test, y_test, y_test_, mse_test = evaluate(model, data, mode='test')
                train_log['mse_train'].append(np.mean(run_loss))
                train_log['mse_test'].append(mse_test)
                for l in range(model.num_layer):
                    train_log['layer=%s_weight_std' % l].append(model.layers[l * 2].weight.std().item())
                    try:
                      train_log['layer=%s_bias_std' % l].append(model.layers[l * 2].bias.std().item())
                    except AttributeError:
                      pass
                log = 'iteration %s, losses: train=%.4e, test=%.4e' % (
                    i, np.mean(run_loss), mse_test)
                if 'layer=0_weight_std' in train_log:
                    log += ', layer0_weight_std=%.4e' % train_log['layer=0_weight_std'][-1]
                #print(log)
                pbar.set_description(log)
                run_loss = []
            if i in plot_every:
                file = log_dir + 'fig_%s.png' % i
                train_log['filenames'].append(file)
                make_plot(model, data, file)
                #train_log['nlds'].append(make_plot(model, data, file))
            model.train()
    return train_log


def evaluate(model, data, mode='train', max_train_batch=1024):
    model.eval()
    if mode == 'train':
        num_batch = int(np.min([data.train_batch, max_train_batch]))
        data.batch_count = 0
    elif mode == 'test':
        num_batch = data.test_batch
    elif mode == 'grid':
        num_batch = data.grid_batch
    X, Y, Y_ = [], [], []
    for i in range(num_batch):
        if mode == 'train':
            x, y = data.get_train()
        elif mode == 'test':
            x, y = data.get_test(i)
        elif mode == 'grid':
            x, y = data.get_grid(i)
        X.append(x.copy())
        Y.append(y.copy()[:, 0])
        y_ = model(
            torch.tensor(x, device=device, dtype=dtype)
        ).detach().cpu().numpy().copy()[:, 0]
        Y_.append(y_)
    X = np.concatenate(X)
    Y = np.concatenate(Y)
    Y_ = np.concatenate(Y_)
    mse = np.mean((Y - Y_) ** 2)
    return X, Y, Y_, mse


def make_plot(model, data, file):
    x, y, y_, _ = evaluate(model, data, mode='grid')
    nld = neg_log_dist(y, y_)
    #return nld
    plt.figure(figsize=(6, 6))
    plt.scatter(*x.T, c=nld, s=1)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(file, dpi=600, bbox_inches='tight', pad_inches=-0.1)
    plt.clf()


### GP MODELS ###

def train_gp(
        data,
        training_iter=1000,
        lr=1e-1,
        verbose=False,
        lengthscale=1.0,  # learnable if None
        framework='sklearn',
        seed=42,
        grid_batch_size=10000,
        n_restarts_optimizer=100,
        noise_kernel=False,
):
    # prepare data
    x_train, y_train = data.get_train()
    x_test, y_test = data.get_test(0)
    x_grid, y_grid = data.make_grid()

    # initialize model and likelihood
    if framework == 'sklearn':
        if lengthscale == None:
            kernel = 1.0 * RBF()
        else:
            kernel = 1.0 * RBF(length_scale=lengthscale, length_scale_bounds='fixed')
        if noise_kernel:
            kernel = kernel + WhiteKernel()
        gpr = GaussianProcessRegressor(
            kernel=kernel,
            random_state=seed,
            n_restarts_optimizer=n_restarts_optimizer
        )
        if verbose:
            print(f"Kernel parameters before fit:\n{kernel})")

    # Find optimal model hyperparameters
    if framework == 'sklearn':
        gpr.fit(x_train, y_train)
        if verbose:
            print(f"Kernel parameters after fit: \n{gpr.kernel_} \n"
              f"Log-likelihood: {gpr.log_marginal_likelihood(gpr.kernel_.theta):.3f}")

    # make predictions
    if framework == 'sklearn':
        y_train_ = gpr.predict(x_train, return_std=False)
        y_test_ = gpr.predict(x_test, return_std=False)
        # grid
        num_batch = int(np.ceil(x_grid.shape[0] / grid_batch_size))
        y_grid_, std_grid = [], []
        for i in range(num_batch):
            y_, std = gpr.predict(x_grid[i * grid_batch_size:(i + 1) * grid_batch_size], return_std=True)
            y_grid_.append(y_[:, 0])
            std_grid.append(std)
        model = gpr
        output = {
            'llh_train': gpr.log_marginal_likelihood(gpr.kernel_.theta),
            'mse_train': np.mean((y_train - y_train_) ** 2),
            'mse_test': np.mean((y_test - y_test_) ** 2),
            'y_grid_': np.concatenate(y_grid_, 0),
            'std_grid': np.concatenate(std_grid, 0),
        }

    return model, output
