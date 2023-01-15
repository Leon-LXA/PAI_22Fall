import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
from matplotlib import pyplot as plt

domain = np.array([[0, 5]])
SAFETY_THRESHOLD = 1.2
SEED = 1
simplefilter("ignore", category=ConvergenceWarning)
""" Solution """
np.random.seed(SEED)

class BO_algo():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration. """

        # TODO: enter your code here
        self.kernel = 0.5 * Matern(length_scale=0.5, nu=2.5)+WhiteKernel(noise_level=0.15)
        self.kernel_v = 1.5 + np.sqrt(2) * Matern(length_scale=0.5, nu=2.5)+WhiteKernel(noise_level=0.0001)
        # self.model = GaussianProcessRegressor(kernel=self.kernel)
        self.model = GaussianProcessRegressor(self.kernel)
        self.model_v = GaussianProcessRegressor(self.kernel_v)

        self.beta = 0.8
        self.X = []
        self.y = []
        self.v = []
        pass


    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: np.ndarray
            1 x domain.shape[0] array containing the next point to evaluate
        """

        # TODO: enter your code here
        # In implementing this function, you may use optimize_acquisition_function() defined below.
        x_next = self.optimize_acquisition_function()
        return x_next
        # raise NotImplementedError


    def optimize_acquisition_function(self):
        """
        Optimizes the acquisition function.

        Returns
        -------
        x_opt: np.ndarray
            1 x domain.shape[0] array containing the point that maximize the acquisition function.
        """

        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick best solution
        for _ in range(20):
            x0 = domain[:, 0] + (domain[:, 1] - domain[:, 0]) * \
                 np.random.rand(domain.shape[0])
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=domain,
                                   approx_grad=True)
            x_values.append(np.clip(result[0], *domain[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        return np.atleast_2d(x_values[ind])

    def acquisition_function(self, x):
        """
        Compute the acquisition function.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f

        Returns
        ------
        af_value: float
            Value of the acquisition function at x
        """

        # TODO: enter your code here
        mean, cov = self.model.predict([x], return_std=True)
        v = self.model_v.predict([x])
        k = 0.1
        if v > SAFETY_THRESHOLD:
            return mean[0][0] + self.beta * np.sqrt(cov[0])
        else:
            # return - 10 + k*(v[0][0]-SAFETY_THRESHOLD)
            return - 10

        # return mean[0][0] + self.beta * np.sqrt(cov[0]) + k * v[0][0]
        # raise NotImplementedError


    def add_data_point(self, x, f, v):
        """
        Add data points to the model.

        Parameters
        ----------
        x: np.ndarray
            Hyperparameters
        f: np.ndarray
            Model accuracy
        v: np.ndarray
            Model training speed
        """
        x = np.atleast_2d(x)
        f = np.atleast_2d(f)
        v = np.atleast_2d(v)
        # TODO: enter your code here
        if not any(self.X):
            self.X = np.empty((0, np.size(x, 1)), float)
            self.y = np.empty((0, np.size(f, 1)), float)
            self.v = np.empty((0, np.size(v, 1)), float)
        self.X = np.vstack((self.X, x))
        self.y = np.vstack((self.y, f))
        self.v = np.vstack((self.v, v))
        # self.X.append(x)
        # self.y.append(f)
        self.model.fit(self.X, self.y)
        self.model_v.fit(self.X, self.v)
        # print("here")

        # raise NotImplementedError

    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: np.ndarray
            1 x domain.shape[0] array containing the optimal solution of the problem
        """

        # TODO: enter your code here
        x_domain = np.linspace(*domain[0], 4000)[:, None]
        y_max = -np.Inf
        i_max = 0
        y_list = []
        # mean_list = []
        # std_list = []
        for i, x in enumerate(x_domain):
            mean, std = self.model.predict([x], return_std=True)
            v = self.model_v.predict([x])
            y_new = mean + self.beta * np.sqrt(std)
            # y_list.append(y_new[0][0])
            # mean_list.append(mean[0][0])
            # std_list.append(std[0])
            if y_new > y_max and v > SAFETY_THRESHOLD:
                y_max = y_new
                i_max = i
        # mu,cov = self.model.predict(x_domain, return_std=True)
        # test_y = mu.ravel()
        # uncertainty = 1 * np.sqrt(cov)
        # plt.figure()
        # plt.fill_between(x_domain.ravel(), test_y + uncertainty, test_y - uncertainty, alpha=0.1)
        # plt.plot(x_domain, test_y, label="predict")
        # # plt.scatter(train_X, train_y, label="train", c="red", marker="x")
        # plt.legend()
        # plt.show()
        #
        # mu = self.model_v.predict(x_domain)
        # test_y = mu.ravel()
        # plt.figure(2)
        # plt.plot(x_domain, test_y, label="predict")
        # plt.legend()
        # plt.show()

        # mean, std = self.model.predict(x_domain, return_std=True)
        # std = std.reshape(4000, 1)
        # y = mean + self.beta * std
        # i_max = np.argmax(y)

        return x_domain[i_max]

        # raise NotImplementedError


""" Toy problem to check code works as expected """

def check_in_domain(x):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= domain[None, :, 0]) and np.all(x <= domain[None, :, 1])


def f(x):
    """Dummy objective"""
    mid_point = domain[:, 0] + 0.5 * (domain[:, 1] - domain[:, 0])
    return - np.linalg.norm(x - mid_point, 2)  # -(x - 2.5)^2


def v(x):
    """Dummy speed"""
    return 2.0

def get_initial_safe_point():
    """Return initial safe point"""
    x_domain = np.linspace(*domain[0], 4000)[:, None]
    c_val = np.vectorize(v)(x_domain)
    x_valid = x_domain[c_val > SAFETY_THRESHOLD]
    np.random.seed(SEED)
    np.random.shuffle(x_valid)
    x_init = x_valid[0]
    return x_init



def main():
    # Init problem
    agent = BO_algo()

    # Add initial safe point
    x_init = get_initial_safe_point()
    obj_val = f(x_init)
    cost_val = v(x_init)
    agent.add_data_point(x_init, obj_val, cost_val)


    # Loop until budget is exhausted
    for j in range(20):
        # Get next recommendation
        x = agent.next_recommendation()

        # Check for valid shape
        assert x.shape == (1, domain.shape[0]), \
            f"The function next recommendation must return a numpy array of " \
            f"shape (1, {domain.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x)
        cost_val = v(x)
        agent.add_data_point(x, obj_val, cost_val)

    # Validate solution
    solution = np.atleast_2d(agent.get_solution())
    assert solution.shape == (1, domain.shape[0]), \
        f"The function get solution must return a numpy array of shape (" \
        f"1, {domain.shape[0]})"
    assert check_in_domain(solution), \
        f'The function get solution must return a point within the ' \
        f'domain, {solution} returned instead'

    # Compute regret
    # print(v(solution))
    if v(solution) < 1.2:
        regret = 1
    else:
        regret = (0 - f(solution))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret{regret}')


if __name__ == "__main__":
    main()