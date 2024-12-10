import torch
import torch.nn as nn
import torch.nn.functional as F

class CorrelationPenaltyLoss(nn.Module):
    def __init__(self, model, base_loss_fn, layer_type, num_orthog_classes=None, one_hot=False, alpha=0.1, epsilon=1e-8):
        """
        CorrelationPenaltyLoss adds an additional loss scalar to the chosen base loss (e.g., Cross Entropy).
        It penalises the model for learning weights that correlate to an orthogonal task.
        The desired and orthogonal labels are provided in the forward call.

        Args:
            model: the model being trained
            base_loss_fn: the basis for calculating loss
            layer_type (string): the type of nn layer to apply the loss to: "nn.Linear", "nn.Conv2d" or "both"
            num_orthog_classes (int, optional): number of orthogonal classes if one-hot encoding is used
            one_hot (boolean, optional): whether to use one-hot encoding for the orthogonal classes
            alpha (float, optional): hyperparameter - scalar for the additional loss. Defaults to 0.1.
            epsilon (float, optional): added to std in batch normalization to avoid NaN outputs
        """
        super(CorrelationPenaltyLoss, self).__init__()
        self.model = model
        self.base_loss_fn = base_loss_fn
        self.layer_type = layer_type
        self.one_hot = one_hot
        self.num_orthog_classes = num_orthog_classes
        self.alpha = alpha
        self.epsilon = epsilon
        self.activations = {}
        self._register_hooks()

    # we first need to register hooks to get the activations
    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if self.layer_type == "nn.Linear":
                if isinstance(module, nn.Linear):
                    module.register_forward_hook(self._create_hook(name))
            elif self.layer_type == "nn.Conv2d":
                if isinstance(module, nn.Conv2d):
                    module.register_forward_hook(self._create_hook(name))
            elif self.layer_type == "both":
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                    module.register_forward_hook(self._create_hook(name))
            else:
                break

    def _create_hook(self, name):
        def hook(module, input, output):
            self.activations[name] = output
        return hook

    def forward(self, output, desired_labels, orthogonal_labels):
        # add our calculation to the forward pass
        
        # check if the activations are empty
        if self.activations is None:
            raise ValueError("Activations are none. Ensure the network has the given layer type")

        # initialise the loss
        base_loss = self.base_loss_fn(output, desired_labels)
        batch_size = output.size(0)
        corr_penalty = 0.0
        
        # if the orthogonal labels are categorical, we need one-hot encoding
        if self.one_hot:
            if self.num_orthog_classes is None:
                raise ValueError("num_orthog_classes must be specified for one-hot encoding.")
            orthogonal_labels = F.one_hot(orthogonal_labels, num_classes=self.num_orthog_classes).float()
        else:
            orthogonal_labels = orthogonal_labels.view(batch_size, -1).float()

        # loop through the dictionary items for all layers
        for name, activation in self.activations.items():
            # reshape for correlation calculation
            activation_flat = activation.view(batch_size, -1)

            # normalise activations and orthogonal labels
            activation_norm = (activation_flat - activation_flat.mean(dim=0)) / (activation_flat.std(dim=0) + self.epsilon)
            orthogonal_labels_norm = (orthogonal_labels - orthogonal_labels.mean(dim=0)) / (orthogonal_labels.std(dim=0) + self.epsilon)

            # calculate correlation
            corr_matrix = torch.mm(activation_norm.T, orthogonal_labels_norm) / (batch_size - 1)
            corr_penalty_for_layer = torch.norm(corr_matrix, p=2)

            # scale by the number of nodes
            num_nodes = activation_flat.size(1)
            corr_penalty += corr_penalty_for_layer / num_nodes

        # scale by the number of layers
        num_layers = len(self.activations)
        if num_layers > 0:
            corr_penalty /= num_layers
            
        # final calc incorporates the scalar alpha
        total_loss = base_loss + self.alpha * corr_penalty
        return total_loss
