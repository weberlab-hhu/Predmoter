# Weighting experiment
Weighting the ATAC- and ChIP-seq data to combat overfitting towards a NGS dataset,
i.e. H3K4me3 ChIP-seq data, was thought to improve the predictions. The necessary
[code changes](#code-changes) were implemented and three different setups with three
replicates each were trained with the exact same base setup and datasets used by the
combined model (see [pretrained models](https://github.com/weberlab-hhu/predmoter_models)
and the supplementary material of [this preprint](https://doi.org/10.1101/2023.11.03.565452)
for more information). The three setups applied weighting to the training loss
**only**. The tested weights were:
- 90 % ATAC-seq, 10 % ChIP-seq
- 80 % ATAC-seq, 20 % ChIP-seq
- 70 % ATAC-seq, 30 % ChIP-seq
    
The default 'loss weights' before defined that each dataset contributed towards the
training loss equally. If one dataset was overrepresented (more data of this dataset
was available and trained on) it contributed more towards the training loss, meaning
the network's predictions could skew towards the larger dataset.    
    
The weighting was implemented, so that the default behavior from before was 
loaded for older models, so that weighting would be backwards compatible and old models
could still be loaded and training for them could be resumed leading to the exact
same results as before. This **ONLY** applied to models using one dataset, where no
loss weight would be used regardless, and models using two datasets, where the loss
weight would then be 0.5 for both. Models training on an uneven number of datasets
(models with three datasets were tested) were **not** backwards compatible.
    
## Results
The ATAC-seq predictions were in some cases slightly improved, but the ChIP-seq
predictions worsened more than the ATAC-seq metrics improved. The tradeoff was
decided not to be worth it and the code changes weren't. The exact code changes are
documented below, in case weighting might become relevant again or someone likes
to implement the changes for themselves. The code changes were performed on the
GitHub commit version **3ddadf9** of Predmoter version 0.3.2.    
The metric displayed is the Pearson's correlation. The closer to 1 the value is
the better are the model's predictions. The validation species were
*Medicago truncatula* and *Spirodela polyrhiza*, the test species were
*Arabidopsis thaliana* and *Oryza sativa*.    
    
**Pearson’s correlation for ATAC-seq predictions per species:**
    
| Species                   | Combined | Combined 90/10 | Combined 80/20 | Combined 70/30 |
|:--------------------------|:---------|:---------------|:---------------|:---------------|
| *Arabidopsis thaliana*    | 0.6106   | 0.6219         | 0.5959         | 0.5945         |
| *Bigelowiella natans*     | 0.5608   | 0.6198         | 0.5407         | 0.5529         |
| *Brachypodium distachyon* | 0.7165   | 0.7428         | 0.7238         | 0.7247         |
| *Brassica napus*          | 0.4777   | 0.4887         | 0.4685         | 0.4656         |
| *Eragrostis nindensis*    | 0.4930   | 0.5273         | 0.4856         | 0.4909         |
| *Glycine max*             | 0.7174   | 0.7226         | 0.7126         | 0.7111         |
| *Malus domestica*         | 0.4801   | 0.5081         | 0.4646         | 0.4653         |
| *Marchantia polymorpha*   | 0.5965   | 0.6294         | 0.5894         | 0.5912         |
| *Medicago truncatula*     | 0.5583   | 0.5581         | 0.5494         | 0.5586         |
| *Oropetium thomaeum*      | 0.7600   | 0.7954         | 0.7403         | 0.7517         |
| *Oryza sativa*            | 0.4472   | 0.4736         | 0.4792         | 0.4651         |
| *Solanum lycopersicum*    | 0.4938   | 0.4914         | 0.5080         | 0.5100         |
| *Spirodela polyrhiza*     | 0.4964   | 0.5091         | 0.4785         | 0.4731         |
| *Zea mays*                | 0.5400   | 0.5546         | 0.5249         | 0.5331         |
    
**Pearson’s correlation for ChIP-seq predictions per species:**
    
| Species                     | Combined | Combined 90/10 | Combined 80/20 | Combined 70/30 |
|:----------------------------|:---------|:---------------|:---------------|:---------------|
| *Arabidopsis thaliana*      | 0.7641   | 0.7518         | 0.7538         | 0.7533         |
| *Brachypodium distachyon*   | 0.7871   | 0.7721         | 0.7700         | 0.7731         |
| *Brassica napus*            | 0.6222   | 0.5956         | 0.5978         | 0.6010         |
| *Brassica oleracea*         | 0.5893   | 0.5591         | 0.5602         | 0.5629         |
| *Brassica rapa*             | 0.6616   | 0.6312         | 0.6306         | 0.6360         |
| *Chlamydomonas reinhardtii* | 0.7988   | 0.7204         | 0.7303         | 0.7597         |
| *Eragrostis nindensis*      | 0.5603   | 0.5315         | 0.5359         | 0.5476         |
| *Glycine max*               | 0.5662   | 0.5337         | 0.5296         | 0.5377         |
| *Malus domestica*           | 0.4919   | 0.4594         | 0.4531         | 0.4551         |
| *Medicago truncatula*       | 0.3771   | 0.3702         | 0.3760         | 0.3708         |
| *Oryza brachyantha*         | 0.8372   | 0.8093         | 0.8064         | 0.8131         | 
| *Oryza sativa*              | 0.6160   | 0.6044         | 0.5949         | 0.6036         |
| *Prunus persica*            | 0.7055   | 0.6682         | 0.6706         | 0.6734         |
| *Pyrus x bretschneideri*    | 0.6328   | 0.6041         | 0.6036         | 0.6032         |
| *Sesamum indicum*           | 0.7895   | 0.7618         | 0.7602         | 0.7676         |
| *Setaria italica*           | 0.7296   | 0.6800         | 0.6824         | 0.7082         |
| *Solanum lycopersicum*      | 0.5699   | 0.5632         | 0.5776         | 0.5704         |
| *Spirodela polyrhiza*       | 0.4227   | 0.3909         | 0.3742         | 0.3853         |
| *Zea mays*                  | 0.3294   | 0.2971         | 0.2847         | 0.3100         | 
    
## Code changes <a id="code-changes"></a>
**File: Predmoter.py**
```python
def main():
    ...
    if args.resume_training or args.mode in ["test", "predict"]:
    # replace these two lines (153 and 154)
        args.datasets = torch.load(args.model,
                                   map_location=lambda storage, loc: storage)["hyper_parameters"]["datasets"]

    # with these lines
        hyperparameters = torch.load(args.model, map_location=lambda storage, loc: storage)["hyper_parameters"]
        args.datasets = hyperparameters["datasets"]
        if "loss_weights" in hyperparameters:
            args.loss_weights = hyperparameters["loss_weights"]
        else:
            # when an older model is used, load default loss weights (each dataset will contribute equally),
            # which will yield the same results as the code before the weighting implementation
            args.loss_weights = [round((1/len(args.datasets)), 2)] * len(args.datasets)
    
    if len(args.datasets) == 1:  # weights are not needed for one dataset
        args.loss_weights = None
    ...
# the number of indents match the required number in the actual file
```
    
**File: predmoter/core/parser.py**
```python
self.config_group.add_argument("--datasets", ...)
# add in these lines after line 84/after the dataset argument is defined:       
self.config_group.add_argument("--loss-weights", type=float, nargs="+", default=[0.5, 0.5],
                               help="loss weights of the different datasets, they need to be the same order "
                                    "as the datasets; they are only used to weight the training loss and no "
                                    "other metric")

# second code change:
def check_args(self, args):
    ...
    # add in these lines right before the model checks line (around line 210):
    if len(args.datasets) > 1:
        assert len(args.datasets) == len(args.loss_weights), "the same amount of loss weights as datasets have " \
                                                             "to be set when training on multiple datasets"
        assert round(sum(args.loss_weights), 1) == 1, "loss weights should sum up to 1 (100 %)"
    
    # Model checks
    # -------------
    ...
# the number of indents DON'T match the required number in the actual file
```
    
**File: predmoter/utilities/utils.py**
```python
# replace the old init_model function (line 192) with this one:
def init_model(args, seq_len, bases, load_from_checkpoint: bool):  # args = Predmoter args
    """Initialize the model."""
    if load_from_checkpoint:
        msg_start = "Model's"
        # if something ever changes in the nucleotide encoding,
        # bases of the input files == model input_size need to be checked here
        rank_zero_info(f"Chosen model checkpoint: {args.model}")
        device = args.device if args.device == "cpu" else "cuda"
        model = LitHybridNet.load_from_checkpoint(checkpoint_path=args.model, seq_len=seq_len,
                                                  loss_weights=args.loss_weights, map_location=torch.device(device))
    else:
        msg_start = "Chosen"
        model = LitHybridNet(args.model_type, args.cnn_layers, args.filter_size, args.kernel_size,
                             args.step, args.up, args.dilation, args.lstm_layers, args.hidden_size,
                             args.bnorm, args.dropout, args.learning_rate, seq_len, input_size=bases,
                             output_size=len(args.datasets), datasets=args.datasets, loss_weights=args.loss_weights)

    # models loss weights and datasets were preloaded beforehand and replaced the command line args
    msg = f"{msg_start} dataset(s): {', '.join(args.datasets)}."
    if len(args.datasets) > 1:
        msg += f" {msg_start} loss weights: {', '.join([str(w) for w in args.loss_weights])}."
    rank_zero_info(msg)
    rank_zero_info(f"\n\nModel summary (model type: {model.model_type}, dropout: {model.dropout})"
                   f"\n{ModelSummary(model=model, max_depth=-1)}\n")
    return model
```
    
**File: predmoter/prediction/HybridModel.py**
```python
# replace the old class LitHybridNet(pl.LightningModule)... with this:
class LitHybridNet(pl.LightningModule):
    def __init__(self, model_type, cnn_layers, filter_size, kernel_size, step, up, dilation,
                 lstm_layers, hidden_size, bnorm, dropout, learning_rate, seq_len,
                 input_size, output_size, datasets, loss_weights):
    super(LitHybridNet, self).__init__()
    ...
    # add in self.loss_weights after self.datasets=datasets
    self.loss_weights = loss_weights
    ...
    def step_fn(self, batch, prefix):
        ...
        # add in this line after the second line in the function (pred = self(X))
        batch_size = X.size(0)
        ...
        # replace line 175 to 182 with this (these lines start with the same comment)
        # mask filler if there is more than 1 ngs dataset
        # -------------------------------------------------
        if self.output_size > 1:
            mask = torch.where(Y[:, 0] == -3, False, True)  # False if -3 fill value in tensor
            loss_weights = self.create_loss_weights(batch_size, mask) if prefix == "train" else None
            # mask missing ngs data (not every file has every dataset)
            pred, Y = pred[mask], Y[mask]
        else:
            loss_weights = None

        loss = self.poisson_nll_loss(pred, Y, loss_weights, is_log=True)
        ...
    ...
    # replace the custom poisson_nll_loss function (line 313 to 331) with this one
    @staticmethod
    def poisson_nll_loss(prediction, target, weights=None, is_log=True):
        """Custom way of calculating Poisson negative log likelihood loss.

        The formula is adapted from Pytorch's Poisson loss function
        (see https://github.com/pytorch/pytorch/blob/main/torch/_refs/nn/functional/__init__.py).
        This is a simpler version, that uses torch.nanmean() as a reduction method to simulate masking,
        as Pytorch MaskedTensors are still in the prototype phase. It also contains weighting used when
        training on multiple datasets simultaneously.
        """
        dims = len(prediction.size())
        assert dims <= 2, f"can only calculate pearson's r for tensors with 1 or 2 dimensions, not {dims}"
        if is_log:
            loss = torch.exp(prediction) - target * prediction
        else:
            loss = prediction - target * torch.log(prediction + EPS)

        if dims > 1:
            loss = torch.nanmean(loss, dim=1)
            if weights is not None:
                loss = loss * weights
        return torch.nanmean(loss)
    ...
    # add in this new class function after the last one (def unpack()...)
        def create_loss_weights(self, batch_size, mask):
        """Compute the loss weight tensor. This tensor will be multiplied with the training loss and accuracy.

            Args:
                batch_size: 'int', the batch size BEFORE unpacking
                mask: 'torch.tensor', a tensor generated with 'compute_mask' to filter out the filler datasets
                      so the weights and computed losses/Pearson correlation coefficients have the same shape

            Returns:
                'torch.tensor', the weight per dataset row (see 'unpack')

        Example:
            The example weights generated for the example tensor in the 'unpack' function are shown.
            The loss weights are 0.7 for ATAC-seq and 0.3 for ChIP-seq.
            tensor([1.4, 0.6, 1.4, 0.6])
        """
        weight_tensors = [torch.full((batch_size,), weight * 2, device=self.device) for weight in self.loss_weights]
        loss_weights = torch.stack(weight_tensors, dim=1).view(batch_size * len(self.loss_weights), )
        return loss_weights[mask]
    
```
