        print(f"Loss: {(sum(train_loss['loss_ab']) + sum(train_loss['loss_self'])) / (n_batches_ab + n_batches_self):.5f}")
        return train_loss, {
            "p_loss": (sum(train_loss["p_loss_ab"]) + sum(train_loss["p_loss_self"])) / (n_batches_ab + n_batches_self),
