import copy
import torch


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs=50,
    patience=7
):
    train_losses = []
    val_losses = []

    best_val_loss = float("inf")
    best_epoch = 0
    best_weights = None
    epochs_without_improvement = 0

    for epoch in range(num_epochs):

        # Training
        model.train()
        running_train_loss = 0.0

        for features, labels in train_loader:
            optimizer.zero_grad()

            outputs = model(features)
            loss = criterion(outputs, labels.view(-1, 1))

            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * features.size(0)

        epoch_train_loss = running_train_loss / len(
            train_loader.dataset
        )
        train_losses.append(epoch_train_loss)

        # Validation
        model.eval()
        running_val_loss = 0.0

        with torch.no_grad():
            for features, labels in val_loader:
                outputs = model(features)
                loss = criterion(outputs, labels.view(-1, 1))

                running_val_loss += loss.item() * features.size(0)

        epoch_val_loss = running_val_loss / len(
            val_loader.dataset
        )
        val_losses.append(epoch_val_loss)

        # Early stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_epoch = epoch + 1
            best_weights = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if (epoch + 1) % 5 == 0:
            print(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Train: {epoch_train_loss:.4f} | "
                f"Val: {epoch_val_loss:.4f}"
            )

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Restore the best weights
    model.load_state_dict(best_weights)
    model.eval()

    print(f"Best epoch: {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.4f}")

    return model, train_losses, val_losses