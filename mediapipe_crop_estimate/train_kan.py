# Based on https://github.com/KindXiaoming/pykan/blob/master/hellokan.ipynb

from kan import KAN
from matplotlib import pyplot as plt

from mediapipe_crop_estimate.train_dataset import get_dataset

dataset = get_dataset()

train_features = dataset["train_input"].shape[-1]
label_features = dataset["train_label"].shape[-1]

for i in range(label_features):
    feature_dataset = {
        "train_input": dataset["train_input"],
        "train_label": dataset["train_label"][:, [i]],
        "test_input": dataset["test_input"],
        "test_label": dataset["test_label"][:, [i]]
    }

    train_losses = []
    test_losses = []

    last_model = None

    # for grid in [5, 10, 20]:  # Train and refine the grid
    for grid in [5]:  # Train and refine the grid
        model = KAN(width=[train_features, 5, 1],  # Features input, 5 hidden neurons, 1D output
                    grid=grid,  # 5 grid intervals
                    k=3)  # cubic spline
        if last_model is not None:
            model = model.initialize_from_another_model(last_model, feature_dataset["train_input"])
        last_model = model

        results = model.train(dataset, opt="LBFGS", steps=50, stop_grid_update_step=30)
        train_losses += results['train_loss']
        test_losses += results['test_loss']

    plt.plot(train_losses)
    plt.plot(test_losses)
    plt.legend(['train', 'test'])
    plt.ylabel('RMSE')
    plt.xlabel('step')
    plt.yscale('log')
    plt.show()

    model = model.prune()
    model(dataset['train_input'])
    model.plot()
    plt.show()

    lib = ['x', 'x^2', 'exp', 'log', 'sqrt', 'sin']
    model.auto_symbolic(lib=lib)

    model.train(dataset, opt="LBFGS", steps=50)

    print(model.symbolic_formula()[0][0])
