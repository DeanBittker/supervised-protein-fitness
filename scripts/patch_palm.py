import os
import sys

# PALM's LightAttention predictor refuses regression in three places: it will not build
# the model, its forward pass thresholds against a classification cutoff, and after
# training it looks for an optimal cutoff. None of those are properties of the network,
# which is sigmoid bounded and trains happily against a continuous target in [0, 1]
# once the labels are min-max scaled. This applies the smallest change that lets it run
# and leaves the architecture and the loss alone.
#
#   python3 scripts/patch_palm.py ~/PALM

palm_root = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/PALM")
targets = {
    'predictors': f"{palm_root}/src/model/predictors.py",
    'scalers': f"{palm_root}/src/model/scalers.py",
    'light_attention': f"{palm_root}/src/helpers/pytorch/light_attention.py",
}

loss_patches = [
    (
        "let the loss be chosen at run time",
        """        self.loss_fxn = nn.BCELoss()""",
        """        # cross entropy against a soft target cannot fall below the entropy of
        # the targets themselves, which for values squashed into [0, 1] sits near 0.65
        # and leaves a few hundredths of range to read a curve in. Squared error has no
        # such floor, so PALM_LOSS=mse makes the curves and any comparison legible. The
        # head stays sigmoid bounded either way, which suits a target already in [0, 1]
        import os as _os
        self.loss_fxn = (nn.MSELoss()
                         if _os.environ.get("PALM_LOSS", "bce").lower() == "mse"
                         else nn.BCELoss())""",
    ),
]

scaler_patches = [
    (
        "keep the scaler filename inside the filesystem limit",
        """        if pred_model_name := self.cfg.predictor.model_name:
            return Path(f"{pred_model_name}_{self.name}.skops")""",
        """        if pred_model_name := self.cfg.predictor.model_name:
            # the model name is every hyperparameter joined together, which on a model
            # with this many of them runs past the 255 bytes a filename is allowed. the
            # readable part is kept and a digest of the whole preserves uniqueness
            if len(pred_model_name) > 120:
                import hashlib
                digest = hashlib.sha1(pred_model_name.encode()).hexdigest()[:10]
                pred_model_name = f"{pred_model_name[:120]}_{digest}"
            return Path(f"{pred_model_name}_{self.name}.skops")""",
    ),
    (
        "clip the scaled target into the range the loss accepts",
        """        _scaler = self.scaler_map[self.config_scaler_name]()""",
        """        # a scaler fitted on train and applied to validation can push a value
        # outside the range it learned: a variant more extreme than anything in
        # training scales below zero. BCE asserts its target lies in [0, 1], and on
        # a GPU that surfaces as an asynchronous device-side assert during the first
        # validation pass, pointing at teardown rather than at the loss
        _scaler = (MinMaxScaler(clip=True)
                   if self.config_scaler_name == "MinMaxScaler"
                   else self.scaler_map[self.config_scaler_name]())""",
    ),
]

patches = [
    (
        "build the model in regression mode",
        '''                if cfg.predictor.model_type == "regression":
                    raise NotImplementedError("Regression not implemented for LightAttention")
                elif cfg.predictor.model_type == "classification_binary":
                    # Lazy load (once we know the embedding dimension)
                    self.model = None''',
        '''                if cfg.predictor.model_type in ("regression", "classification_binary"):
                    # Lazy load (once we know the embedding dimension)
                    self.model = None''',
    ),
    (
        "return the bounded output rather than thresholding it",
        '''        if self.model_type == "classification_binary":
            with torch.inference_mode():
                predicted_probabilities = self.model.forward(padded_embeddings, mask)
                predicted_probabilities = self.model.convert_to_numpy(predicted_probabilities, mask)
            predictions = (
                predicted_probabilities >= self.optimal_cutoff if self.optimal_cutoff else None
            )
        else:
            raise NotImplementedError''',
        '''        with torch.inference_mode():
            predicted_probabilities = self.model.forward(padded_embeddings, mask)
            predicted_probabilities = self.model.convert_to_numpy(predicted_probabilities, mask)
        if self.model_type == "classification_binary":
            predictions = (
                predicted_probabilities >= self.optimal_cutoff if self.optimal_cutoff else None
            )
        else:
            # the head is sigmoid bounded, so the bounded output is the prediction
            # itself and there is no threshold to apply
            predictions = predicted_probabilities''',
    ),
    (
        "skip the cutoff search after training a regressor",
        '''        _, predicted_probabilities = self.forward(x_val)
        if self.residue_prediction_mode:''',
        '''        if self.model_type != "classification_binary":
            # an optimal cutoff is a classification idea, there is no threshold to find
            return
        _, predicted_probabilities = self.forward(x_val)
        if self.residue_prediction_mode:''',
    ),
]

applied, already = [], []
for key, patch_set in [('predictors', patches), ('scalers', scaler_patches),
                       ('light_attention', loss_patches)]:
    target = targets[key]
    if not os.path.exists(target):
        raise SystemExit(f"could not find {target} - pass the PALM checkout as the first argument")
    source = open(target).read()
    changed = False
    for label, old, new in patch_set:
        if new in source:
            already.append(label)
        elif old in source:
            source = source.replace(old, new, 1)
            applied.append(label)
            changed = True
        else:
            raise SystemExit(f"could not apply {label!r}: PALM has changed, patch by hand")
    if changed:
        backup = target + ".orig"
        if not os.path.exists(backup):
            open(backup, 'w').write(open(target).read())
            print(f"kept the original at {backup}")
        open(target, 'w').write(source)

for label in applied:
    print(f"  applied  {label}")
for label in already:
    print(f"  already  {label}")
print("\nnow run with dataset.task and predictor.model_type both set to regression")
