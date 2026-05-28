import marimo

__generated_with = "0.11.5"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import torch
    return (
        accuracy_score,
        classification_report,
        confusion_matrix,
        mo,
        nn,
        np,
        optim,
        pd,
        plt,
        sns,
        torch,
        train_test_split,
    )


@app.cell
def _():
    import xgboost as xgb
    from sklearn.multioutput import MultiOutputClassifier
    return MultiOutputClassifier, xgb


@app.cell
def _(np):
    def generate_correlated_data(n_samples = 2000):
        rng = np.random.default_rng(123)

        X= rng.normal(
            size = (n_samples, 2)
        )
        logit_y1 = 1.5*X[:,0]+5*X[:,1]
        prob_y1= 1./(1.+np.exp(-logit_y1))
        y1 = np.random.binomial(1, prob_y1)

        logit_y2 = 2.0*X[:,0] - 3.0*X[:,1] +0.8*y1
        prob_y2 = 1./(1.+np.exp(-logit_y2))
        y2 = np.random.binomial(1, prob_y2)
        Y = np.column_stack((y1, y2))
        return X,Y
    return (generate_correlated_data,)


@app.cell
def _(generate_correlated_data):
    X,Y= generate_correlated_data()
    return X, Y


@app.cell
def _(X, Y, train_test_split):
    # Train-test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    print("Shapes:")
    print("X_train:", X_train.shape, "Y_train:", Y_train.shape)
    print("X_test:", X_test.shape, "Y_test:", Y_test.shape)
    return X_test, X_train, Y_test, Y_train


@app.cell
def _(Y_train, pd):
    # %% [code]
    df_train = pd.DataFrame({"y1": Y_train[:,0], "y2": Y_train[:,1]})
    cont_table = pd.crosstab(df_train["y1"], df_train["y2"], rownames=["y1"], colnames=["y2"])
    print("Contingency table (training):")
    print(cont_table)
    return cont_table, df_train


@app.cell
def _(df_train, plt, sns):
    # We can also look at correlation coefficient:
    corr = df_train.corr()
    print("\nCorrelation matrix:")
    print(corr)

    # Visualize it
    sns.heatmap(corr, annot=True, cmap="Blues", vmin=-1, vmax=1)
    plt.title("Label Correlation (Training)")
    plt.show()
    return (corr,)


@app.cell
def _(X_test, X_train, Y_test, Y_train, accuracy_score, nn, optim, torch):
    # %% [code]
    class MultiLabelMLP(nn.Module):
        def __init__(self, input_dim=2, hidden_dim=16, output_dim=2):
            super(MultiLabelMLP, self).__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.Sigmoid(),

                nn.Linear(hidden_dim, output_dim),
            )

        def forward(self, x):
            # Output dimension = 2
            # We'll apply sigmoid later for probability
            return self.net(x)

    # Prepare data for PyTorch
    X_train_t = torch.from_numpy(X_train).float()
    Y_train_t = torch.from_numpy(Y_train).float()
    X_test_t  = torch.from_numpy(X_test).float()
    Y_test_t  = torch.from_numpy(Y_test).float()

    model = MultiLabelMLP(input_dim=2, hidden_dim=16, output_dim=2)
    criterion = nn.BCEWithLogitsLoss()  # combines sigmoid + BCELoss
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    n_epochs = 50
    batch_size = 64

    for epoch in range(n_epochs):
        # Shuffle
        idx = torch.randperm(X_train_t.size(0))
        X_train_t = X_train_t[idx]
        Y_train_t = Y_train_t[idx]

        # Mini-batch
        for i in range(0, X_train_t.size(0), batch_size):
            x_batch = X_train_t[i:i+batch_size]
            y_batch = Y_train_t[i:i+batch_size]

            # Forward
            logits = model(x_batch)
            loss = criterion(logits, y_batch)


            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate
    model.eval()
    with torch.no_grad():
        logits_test = model(X_test_t)
        preds_test = torch.sigmoid(logits_test)
        # Round
        preds_test_bin = (preds_test.numpy() > 0.5).astype(int)

    acc_y1 = accuracy_score(Y_test[:,0], preds_test_bin[:,0])
    acc_y2 = accuracy_score(Y_test[:,1], preds_test_bin[:,1])
    print("PyTorch Multi-Label MLP Accuracy:")
    print(f"  y1 accuracy: {acc_y1:.3f}")
    print(f"  y2 accuracy: {acc_y2:.3f}")
    return (
        MultiLabelMLP,
        X_test_t,
        X_train_t,
        Y_test_t,
        Y_train_t,
        acc_y1,
        acc_y2,
        batch_size,
        criterion,
        epoch,
        i,
        idx,
        logits,
        logits_test,
        loss,
        model,
        n_epochs,
        optimizer,
        preds_test,
        preds_test_bin,
        x_batch,
        y_batch,
    )


@app.cell
def _(
    X_test_t,
    X_train_t,
    Y_test,
    Y_train_t,
    accuracy_score,
    nn,
    optim,
    torch,
):
    # ame inputs \\((x_1, x_2)\\) but doesn't directly see the other label.



    # %% [code]
    class SingleLabelMLP(nn.Module):
        def __init__(self, input_dim=2, hidden_dim=16):
            super(SingleLabelMLP, self).__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.Sigmoid(),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, x):
            return self.net(x)

    def train_single_label_model(X_train_t, y_train_t, n_epochs=50, batch_size=64, lr=0.01):
        model = SingleLabelMLP(input_dim=2, hidden_dim=16)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        for epoch in range(n_epochs):
            idx = torch.randperm(X_train_t.size(0))
            X_train_t = X_train_t[idx]
            y_train_t = y_train_t[idx]

            for i in range(0, X_train_t.size(0), batch_size):
                x_batch = X_train_t[i:i+batch_size]
                y_batch = y_train_t[i:i+batch_size]

                logits = model(x_batch)
                loss = criterion(logits, y_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return model

    # Train models for y1 and y2
    model_y1 = train_single_label_model(X_train_t, Y_train_t[:,0].view(-1,1))
    model_y2 = train_single_label_model(X_train_t, Y_train_t[:,1].view(-1,1))

    # Evaluate
    def predict_single_label(model, X_test_t):
        model.eval()
        with torch.no_grad():
            logits = model(X_test_t)
            probs = torch.sigmoid(logits)
        return (probs.numpy() > 0.5).astype(int)

    pred_y1 = predict_single_label(model_y1, X_test_t).ravel()
    pred_y2 = predict_single_label(model_y2, X_test_t).ravel()

    acc_y1_sep = accuracy_score(Y_test[:,0], pred_y1)
    acc_y2_sep = accuracy_score(Y_test[:,1], pred_y2)
    print("PyTorch Multiple Single-Label Models Accuracy:")
    print(f"  y1 accuracy: {acc_y1_sep:.3f}")
    print(f"  y2 accuracy: {acc_y2_sep:.3f}")
    return (
        SingleLabelMLP,
        acc_y1_sep,
        acc_y2_sep,
        model_y1,
        model_y2,
        pred_y1,
        pred_y2,
        predict_single_label,
        train_single_label_model,
    )


@app.cell
def _(
    MultiOutputClassifier,
    X_test,
    X_train,
    Y_test,
    Y_train,
    accuracy_score,
    xgb,
):
    # %% [code]
    # MultiOutputClassifier approach (internally it creates one classifier per label 
    # but we can treat it as a single multi-label pipeline).
    multi_xgb = MultiOutputClassifier(xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss"))
    multi_xgb.fit(X_train, Y_train)

    preds_multi = multi_xgb.predict(X_test)
    acc_y1_xgb_multi = accuracy_score(Y_test[:,0], preds_multi[:,0])
    acc_y2_xgb_multi = accuracy_score(Y_test[:,1], preds_multi[:,1])
    print("XGBoost via MultiOutputClassifier Accuracy:")
    print(f"  y1 accuracy: {acc_y1_xgb_multi:.3f}")
    print(f"  y2 accuracy: {acc_y2_xgb_multi:.3f}")

    # Two separate XGBoost classifiers
    xgb_y1 = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    xgb_y2 = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")

    xgb_y1.fit(X_train, Y_train[:,0])
    xgb_y2.fit(X_train, Y_train[:,1])

    pred_y1_xgb_sep = xgb_y1.predict(X_test)
    pred_y2_xgb_sep = xgb_y2.predict(X_test)

    acc_y1_xgb_sep = accuracy_score(Y_test[:,0], pred_y1_xgb_sep)
    acc_y2_xgb_sep = accuracy_score(Y_test[:,1], pred_y2_xgb_sep)
    print("XGBoost Separate Classifiers Accuracy:")
    print(f"  y1 accuracy: {acc_y1_xgb_sep:.3f}")
    print(f"  y2 accuracy: {acc_y2_xgb_sep:.3f}")
    return (
        acc_y1_xgb_multi,
        acc_y1_xgb_sep,
        acc_y2_xgb_multi,
        acc_y2_xgb_sep,
        multi_xgb,
        pred_y1_xgb_sep,
        pred_y2_xgb_sep,
        preds_multi,
        xgb_y1,
        xgb_y2,
    )


@app.cell
def _(
    pd,
    pred_y1,
    pred_y1_xgb_sep,
    pred_y2,
    pred_y2_xgb_sep,
    preds_multi,
    preds_test_bin,
):
    # %% [code]
    print("PyTorch Multi-Label MLP contingency table of predictions on test set:")
    df_pred_multi_pt = pd.DataFrame({
        "y1_pred": preds_test_bin[:,0], 
        "y2_pred": preds_test_bin[:,1]
    })
    print(pd.crosstab(df_pred_multi_pt["y1_pred"], df_pred_multi_pt["y2_pred"]))

    print("\nPyTorch Separate Models contingency table of predictions on test set:")
    df_pred_sep_pt = pd.DataFrame({
        "y1_pred": pred_y1, 
        "y2_pred": pred_y2
    })
    print(pd.crosstab(df_pred_sep_pt["y1_pred"], df_pred_sep_pt["y2_pred"]))

    # Similarly for XGBoost
    print("\nXGBoost MultiOutputClassifier contingency table:")
    df_pred_multi_xgb = pd.DataFrame({
        "y1_pred": preds_multi[:,0], 
        "y2_pred": preds_multi[:,1]
    })
    print(pd.crosstab(df_pred_multi_xgb["y1_pred"], df_pred_multi_xgb["y2_pred"]))

    print("\nXGBoost Separate Classifiers contingency table:")
    df_pred_sep_xgb = pd.DataFrame({
        "y1_pred": pred_y1_xgb_sep, 
        "y2_pred": pred_y2_xgb_sep
    })

    print(pd.crosstab(df_pred_sep_xgb["y1_pred"], df_pred_sep_xgb["y2_pred"]))
    return (
        df_pred_multi_pt,
        df_pred_multi_xgb,
        df_pred_sep_pt,
        df_pred_sep_xgb,
    )


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(r"")
    return


if __name__ == "__main__":
    app.run()
