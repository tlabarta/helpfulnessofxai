import numpy as np
import maxi
import matplotlib.pyplot as plt
from mnist_tf.utils import setup_mnist


def main():
    # load the mnist data, model and autoencoder
    data, model, AE = setup_mnist(
        model_path="./experiments/mnist/models/mnist",
        ae_path="./experiments/mnist/models/AE_codec",
    )

    # chose an image to be explained
    inputs = np.array([data.test_data[10]])

    # chose desired component classes for the loss, optimizer and gradient
    loss_class = maxi.CEMLoss
    optimizer_class = maxi.SpectralAoExpGradOptimizer
    gradient_class = maxi.USRVGradientEstimator

    # specify the configuration for the components
    loss_kwargs = {"mode": "PP", "c": 1, "gamma": 3, "K": 20, "AE": AE}
    optimizer_kwargs = {"l1": 0.025, "l2": 0.000025, "eta": 1.0, "channels_first": False}
    gradient_kwargs = {"mu": None}

    # instantiate the "ExplanationGenerator" with our settings
    cem = maxi.ExplanationGenerator(
        loss=loss_class,
        optimizer=optimizer_class,
        gradient=gradient_class,
        num_iter=1000,
        loss_kwargs=loss_kwargs,
        optimizer_kwargs=optimizer_kwargs,
        gradient_kwargs=gradient_kwargs,
        save_freq=250,
        verbose=True,
    )

    # start the explanation procedure and retrieve the results
    results = cem.run(image=inputs, inference_call=model)

    # visualize the savepoints
    f, axarr = plt.subplots(1, len(results))
    for i, (iter_, result) in enumerate(results.items()):
        axarr[i].title.set_text("Iteration: " + iter_)
        axarr[i].imshow(result.squeeze(axis=-1).squeeze(axis=0), cmap="gray", vmin=-0.5, vmax=0.5)


if __name__ == "__main__":
    main()
